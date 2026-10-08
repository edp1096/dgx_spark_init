package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// QAD keeps the LLM resident. Image and speech/media jobs own the auxiliary
// processes exclusively; stopping containers also releases unified CPU memory.
func workloadGroup(id string) string {
	switch id {
	case "flux2", "qwim-mmh3":
		return "image"
	case "qwen3-tts":
		return "tts"
	case "nemotron-asr", "extra-media":
		return "speech"
	case "extra-documents":
		return "documents"
	case "extra-collector":
		return "collector"
	}
	return ""
}

func maxWorkloadBudget(groups map[string]float64) float64 {
	peak := 0.0
	for _, value := range groups {
		peak = max(peak, value)
	}
	return peak
}

// Resident weights accumulate, while the common lease serializes scratch
// space. Without a workspace profile, startup is a conservative weight floor.
func residentWorkloadMemory(x Component) float64 {
	if x.WorkspaceMemoryGiB > 0 {
		return max(0, x.MemoryGiB-x.WorkspaceMemoryGiB)
	}
	return min(x.MemoryGiB, x.startupMemoryGiB())
}

// Local shared support may be used even when it is not a model-set member.
// External/remote services remain under their owner's lifecycle management.
func (c *Controller) workloadBundle(b Bundle) Bundle {
	seen := map[string]bool{}
	for _, id := range b.Components {
		seen[id] = true
	}
	for _, x := range c.Catalog().SupportComponents(b.ID) {
		if !seen[x.ID] && workloadGroup(x.ID) != "" && x.Controller == "compose" && c.local(x) {
			b.Components = append(append([]string(nil), b.Components...), x.ID)
		}
	}
	return b
}

func (c *Controller) workloadBudget(b Bundle) float64 {
	groups := map[string]float64{}
	resident := 0.0
	for _, id := range b.Components {
		if g := workloadGroup(id); g != "" {
			x, _ := c.Catalog().ResolveComponent(b.ID, id)
			if x.IsSupport() && (x.Controller != "compose" || !c.local(x)) {
				continue
			}
			if x.KeepResident {
				fixed := residentWorkloadMemory(x)
				resident += fixed
				groups[g] += workloadAdditionalMemory(x, fixed)
			} else {
				groups[g] += x.MemoryGiB
			}
		}
	}
	return resident + maxWorkloadBudget(groups)
}

func (c *Controller) stopWorkloads(ctx context.Context, b Bundle) error {
	// Startup filters shared support out of b; ownership still includes Media.
	if full, ok := c.Catalog().Bundle(b.ID); ok {
		b = c.workloadBundle(full)
	}
	for _, id := range b.Components {
		if workloadGroup(id) == "" {
			continue
		}
		x, _ := c.Catalog().ResolveSupport(b.ID, id)
		// Resident members are already accounted for by bundleMemoryPlan.
		// Deferred resident members were stopped explicitly before this sweep;
		// keep other resident services, such as the shared QWIM/H3 DiTs, alive.
		if x.KeepResident {
			continue
		}
		if x.IsSupport() && (x.Controller != "compose" || !c.local(x)) {
			continue
		}
		state, err := c.inspectComponent(ctx, x)
		if err != nil {
			if isMissingContainer(err) {
				continue
			}
			return fmt.Errorf("%s 상태 확인 실패: %w", x.Name, err)
		}
		if state == "exited" || state == "created" || state == "dead" {
			continue
		}
		if state != "running" {
			return fmt.Errorf("%s 종료 전 상태 확인 필요: %s", x.Name, state)
		}
		if x.IsSupport() {
			idle, err := c.supportIdle(ctx, x)
			if err != nil {
				return err
			}
			if !idle {
				return fmt.Errorf("%s 작업이 진행 중이므로 기동용 회수를 중단합니다", x.Name)
			}
		}
		var stopErr error
		if isManagedTTS(x) {
			stopErr = c.stopIdleWorkload(ctx, x)
		} else {
			stopErr = c.stopComponent(ctx, x)
		}
		if err := stopErr; err != nil {
			return fmt.Errorf("%s 메모리 회수: %w", x.Name, err)
		}
		state, err = c.inspectComponent(ctx, x)
		if err != nil && !isMissingContainer(err) {
			return fmt.Errorf("%s 종료 확인: %w", x.Name, err)
		}
		if err == nil && state != "exited" && state != "dead" {
			return fmt.Errorf("%s 종료 확인 실패: %s", x.Name, state)
		}
	}
	return nil
}

// AcquireWorkload holds the runtime operation through response consumption and
// cleanup. A canceled waiter never starts a service. Cleanup uses its own context
// because request cancellation does not guarantee GPU/server work has stopped.
func (c *Controller) AcquireWorkload(ctx context.Context, bundleID, target string, reserve float64, requestedPeak ...float64) (func() error, error) {
	b, ok := c.Catalog().Bundle(bundleID)
	noop := func() error { return nil }
	if ok {
		if image, found := c.Catalog().ResolveComponent(b.ID, target); found && (isQwenImage21(image) || (isManagedTTS(image) && c.local(image))) && image.Controller == "compose" {
			return c.acquireManagedWorkload(ctx, c.workloadBundle(b), image, reserve, requestedPeak...)
		}
	}
	if !ok || !b.WorkloadSwap || workloadGroup(target) == "" {
		return noop, nil
	}
	b = c.workloadBundle(b)
	targetComponent, found := c.Catalog().ResolveSupport(b.ID, target)
	if found && targetComponent.IsSupport() && (targetComponent.Controller != "compose" || !c.local(targetComponent)) {
		return noop, nil
	}
	member := false
	for _, id := range b.Components {
		if id == target {
			member = true
		}
	}
	if !member {
		return nil, fmt.Errorf("%s는 현재 세트에 포함되지 않습니다", target)
	}
	requestedAt := time.Now()
	c.workloadOnce.Do(func() { c.workloadQueue = make(chan struct{}, 1) })
	select {
	case c.workloadQueue <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		<-c.workloadQueue
		return nil, err
	}
	if err := c.begin(Operation{Action: "workload", requestedAt: requestedAt, BundleID: b.ID, ComponentID: target, State: "running", Phase: "작업 서비스 준비", StartedAt: time.Now()}); err != nil {
		<-c.workloadQueue
		return nil, err
	}
	var once sync.Once
	var cleanupErr error
	var operationErr error
	admitted := false
	release := func() error {
		once.Do(func() {
			if operationErr == nil {
				operationErr = ctx.Err()
			}
			cleanupCtx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			defer cancel()
			memory := readSystemMemory()
			if c.memoryProbe != nil {
				memory = c.memoryProbe()
			}
			// Reuse completed services while headroom remains. Canceled or failed
			// work may still be executing remotely, so it must be stopped.
			if admitted && (ctx.Err() != nil || operationErr != nil || memory.AvailableGiB < normalizedMemoryReserve(reserve)) {
				c.updateOperation(target, progressInfo{Key: "workload:cleanup", Phase: "작업 서비스 메모리 반환"})
				// Failure cleanup is confined to the requested group. Never stop
				// the LLM or retained image core because another service failed.
				for _, id := range b.Components {
					if workloadGroup(id) != workloadGroup(target) {
						continue
					}
					x, _ := c.Catalog().ResolveSupport(b.ID, id)
					if x.KeepResident {
						continue
					}
					if x.IsSupport() && (x.Controller != "compose" || !c.local(x)) {
						continue
					}
					if !c.componentRunning(cleanupCtx, x) {
						continue
					}
					var err error
					if id == "flux2" {
						if ctx.Err() != nil || operationErr != nil {
							err = c.fluxRuntimeAction(cleanupCtx, x, "cancel")
						}
						if err == nil {
							err = c.fluxRuntimeAction(cleanupCtx, x, "reclaim")
						}
					} else if x.IsSupport() {
						idle, probeErr := c.supportIdle(cleanupCtx, x)
						if probeErr != nil {
							err = probeErr
						} else if idle {
							err = c.stopComponent(cleanupCtx, x)
						}
					} else {
						err = c.stopComponent(cleanupCtx, x)
					}
					if err != nil && cleanupErr == nil {
						cleanupErr = err
					}
				}
			} else {
				c.updateOperation(target, progressInfo{Key: "workload:reuse", Phase: "부가 서비스 재사용 대기", Detail: "메모리가 부족한 다음 요청에서 유휴 서비스를 회수합니다."})
			}
			if cleanupErr != nil {
				c.finishOperation("failed", cleanupErr.Error())
			} else if operationErr != nil {
				c.finishOperation("failed", operationErr.Error())
			} else {
				c.finishOperation("complete", "")
			}
			<-c.workloadQueue
		})
		return cleanupErr
	}
	fail := func(err error) (func() error, error) {
		operationErr = err
		cleanup := release()
		if cleanup != nil {
			err = fmt.Errorf("%w; 정리 실패: %v", err, cleanup)
		}
		return nil, err
	}
	imagePrepared := false
	if target == "flux2" && c.componentRunning(ctx, targetComponent) {
		state, err := c.fluxMemoryState(ctx, targetComponent)
		if err != nil || state.Busy == nil || state.CoreReady == nil {
			return fail(fmt.Errorf("Klein 상주 상태를 확인하지 못했습니다"))
		}
		if *state.Busy {
			return fail(fmt.Errorf("Klein 작업이 진행 중입니다"))
		}
		imagePrepared = *state.CoreReady
	}
	if target != "flux2" {
		if image, exists := c.Catalog().ResolveComponent(b.ID, "flux2"); exists && c.componentRunning(ctx, image) {
			state, err := c.fluxMemoryState(ctx, image)
			if err != nil || state.Busy == nil {
				return fail(fmt.Errorf("Klein 사용 상태를 확인하지 못했습니다; 본체 유지"))
			}
			if *state.Busy {
				return fail(fmt.Errorf("Klein 작업이 진행 중입니다; 완료 후 다시 요청하세요"))
			}
		}
	}
	// Reserve the entire requested group's peak, not ASR's smaller startup budget.
	var start []Component
	for _, id := range b.Components {
		if workloadGroup(id) != workloadGroup(target) {
			continue
		}
		// Media-only requests do not need to load the ASR model.
		if target == "extra-media" && id == "nemotron-asr" {
			continue
		}
		// The ASR lease begins only after Media has decoded the input.
		// Starting Media again here would consume the very cold-start pages
		// reclaimed for ASR; inference no longer uses its extraction process.
		if target == "nemotron-asr" && id == "extra-media" {
			continue
		}
		x, _ := c.Catalog().ResolveSupport(b.ID, id)
		if x.IsSupport() && (x.Controller != "compose" || !c.local(x)) {
			continue
		}
		if id == target && id == "nemotron-asr" && len(requestedPeak) > 0 && requestedPeak[0] > 0 {
			x.MemoryGiB = requestedPeak[0]
		}
		if id == target && x.IsSupport() {
			limit, err := supportRequestMemoryGiB(x)
			if err != nil {
				return fail(err)
			}
			x.MemoryGiB = max(x.MemoryGiB, limit)
		}
		start = append(start, x)
	}
	cacheReclaimed := false
	fluxReclaimed := false
	radixASR := target == "nemotron-asr"
	if radixASR {
		core, ok := c.Catalog().ResolveComponent(b.ID, "flash-next-radixark")
		radixASR = ok && core.qwenQADVariant() == "radixark"
	}
	if radixASR && !c.componentRunning(ctx, targetComponent) {
		c.updateOperation(target, progressInfo{Key: "workload:qwen-scratch", Phase: "ASR 기동 전 Qwen 임시 메모리 반환"})
		if err := c.reclaimRadixArkScratch(ctx, b); err != nil {
			return fail(err)
		}
	}
	coldFree := 0.0
	if radixASR {
		// A new GGML CUDA context failed at 4.7 GiB immediate free in the
		// live 1M profile. Preserve a 6 GiB bootstrap allowance after cleanup.
		coldFree = 6
	}
	if image, ok := c.Catalog().ResolveComponent(b.ID, "flux2"); ok && isQwenImage21(image) {
		coldFree = managedAuxiliaryCUDAFreeGiB
	}
	deadline := time.Now().Add(30 * time.Second)
	for {
		// Already resident allocations are reflected in MemAvailable. Reserve
		// only the remaining peak of the requested services, never the full
		// cold-start budget a second time.
		gpu := gpuMemoryByPID(ctx)
		additional := 0.0
		coldCUDA := false
		for _, x := range start {
			resident := 0.0
			running := c.componentRunning(ctx, x)
			coldCUDA = coldCUDA || (!running && isCUDAComponent(x))
			if running {
				if x.ID == target && target == "flux2" {
					x.WorkspaceMemoryGiB = liveWorkspaceMemory(ctx, x, requestedPeak...)
				} else {
					x.WorkspaceMemoryGiB = liveWorkspaceMemory(ctx, x)
				}
				for _, pid := range containerPIDs(ctx, x.Container) {
					resident += gpu[pid]
				}
				resident += containerHostResidentMemoryGiB(ctx, x.Container)
			}
			// ASR's override is a whole-service peak; image jobs supply their
			// minimum ADDITIONAL workspace (resident weights are already counted).
			if x.ID == target && target == "flux2" && len(requestedPeak) > 0 && requestedPeak[0] > 0 {
				x.WorkspaceMemoryGiB = max(x.WorkspaceMemoryGiB, requestedPeak[0])
			}
			if x.ID == "flux2" {
				if !imagePrepared {
					additional += max(0, x.startupMemoryGiB()-resident)
				} else {
					// The core is proven resident: the backend returns an
					// additional workspace, not a second whole-service peak.
					extra := x.WorkspaceMemoryGiB
					if x.MemoryGiB > 10 {
						extra = max(extra, x.MemoryGiB-resident)
					}
					additional += extra
				}
			} else {
				additional += workloadAdditionalMemory(x, resident)
			}
		}
		memory := readSystemMemory()
		if c.memoryProbe != nil {
			memory = c.memoryProbe()
		}
		minimum := normalizedMemoryReserve(reserve)
		err := validateMemoryHeadroom(memory, memoryPlan{NeededGiB: additional, RequiresCUDAStart: coldCUDA, MinimumCUDAFreeGiB: coldFree}, minimum)
		if err == nil && memory.AvailableGiB < minimum {
			err = fmt.Errorf("작업 중 최소 여유 부족: 가용 %.1f GiB, 최소 %.1f GiB", memory.AvailableGiB, minimum)
		}
		if err == nil {
			if target == "flux2" && !imagePrepared {
				// Preloading the protected core consumes real headroom. Check
				// the subsequent text/sampling workspace against fresh memory,
				// rather than admitting both phases using the cold snapshot.
				if e := c.startAndWaitContext(ctx, targetComponent); e != nil {
					return fail(e)
				}
				if e := c.fluxRuntimeAction(ctx, targetComponent, "prepare"); e != nil {
					return fail(e)
				}
				imagePrepared = true
				continue
			}
			break
		}
		reclaimed, reclaimErr := c.reclaimIdleAuxiliary(ctx, b, target, false)
		if reclaimErr != nil {
			return fail(reclaimErr)
		}
		if !reclaimed && !fluxReclaimed {
			if image, ok := c.Catalog().ResolveComponent(b.ID, "flux2"); ok && !image.KeepResident && c.componentRunning(ctx, image) {
				c.updateOperation(image.ID, progressInfo{Key: "workload:modules", Phase: image.Name + " 유휴 메모리 회수"})
				if e := c.fluxRuntimeAction(ctx, image, "reclaim"); e != nil {
					return fail(e)
				}
				reclaimed = true
			}
			fluxReclaimed = true
		}
		if !reclaimed {
			reclaimed, reclaimErr = c.reclaimIdleAuxiliary(ctx, b, target, true)
			if reclaimErr != nil {
				return fail(reclaimErr)
			}
		}
		if time.Now().After(deadline) {
			return fail(fmt.Errorf("작업 전 메모리 확인(유휴 경쟁 서비스 회수 후): %w", err))
		}
		if reclaimed {
			continue
		}
		// Retained core memory is not advertised as reclaimable. Reject rather
		// than kill the core or wait for memory that this policy cannot free.
		if !coldCUDA || cacheReclaimed || memory.AvailableGiB-additional < minimum {
			return fail(fmt.Errorf("핵심 모델 유지 후 작업 메모리 부족: %w", err))
		}
		if coldCUDA && memory.FreeGiB < max(immediateFreeReserve(minimum), coldFree) && !cacheReclaimed {
			c.updateOperation(target, progressInfo{Key: "workload:file-cache", Phase: "종료한 부가 모델 파일 캐시 반환"})
			if reclaimErr := c.reclaimAuxiliaryFileCache(ctx, b); reclaimErr != nil {
				return fail(reclaimErr)
			}
			cacheReclaimed = true
			continue
		}
		return fail(err)
	}
	for _, x := range start {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		if err := c.startAndWaitContext(ctx, x); err != nil {
			return fail(err)
		}
		if x.ID == "flux2" && !imagePrepared {
			if err := c.fluxRuntimeAction(ctx, x, "prepare"); err != nil {
				return fail(err)
			}
		}
	}
	admitted = true
	c.updateOperation(target, progressInfo{Key: "workload:execute", Phase: "작업 실행 중", Detail: "다른 이미지·음성·미디어 요청은 순서대로 대기합니다."})
	return release, nil
}

// A resident weight cache does not reserve the next request's scratch space.
// Reclaim one idle auxiliary, then let the caller measure actual headroom.
// Slow image cores and SSH are never candidates. Shared services are considered
// before ASR, and every failure is propagated without crediting freed memory.
func (c *Controller) reclaimIdleAuxiliary(ctx context.Context, b Bundle, target string, includeASR bool) (bool, error) {
	b = c.workloadBundle(b)
	candidates := append([]string(nil), b.Components...)
	sort.SliceStable(candidates, func(i, j int) bool {
		a, _ := c.Catalog().ResolveSupport(b.ID, candidates[i])
		d, _ := c.Catalog().ResolveSupport(b.ID, candidates[j])
		return a.IsSupport() && !d.IsSupport()
	})
	for _, id := range candidates {
		if workloadGroup(id) == "" || workloadGroup(id) == workloadGroup(target) || id == "flux2" || (!includeASR && id == "nemotron-asr") {
			continue
		}
		x, _ := c.Catalog().ResolveSupport(b.ID, id)
		if x.KeepResident || !c.local(x) || x.Controller != "compose" || !c.componentRunning(ctx, x) {
			continue
		}
		if isManagedTTS(x) {
			if err := c.stopIdleWorkload(ctx, x); err != nil {
				continue
			}
			return true, nil
		}
		if x.IsSupport() {
			idle, err := c.supportIdle(ctx, x)
			if err != nil {
				return false, err
			}
			if !idle {
				continue
			}
		}
		c.updateOperation(id, progressInfo{Key: "workload:reclaim:" + id, Phase: "유휴 부가 서비스 메모리 회수", Detail: x.Name})
		if err := c.stopComponent(ctx, x); err != nil {
			return false, err
		}
		state, err := c.inspectComponent(ctx, x)
		if err != nil || state != "exited" {
			return false, fmt.Errorf("%s 회수 확인 실패: %s %v", x.Name, state, err)
		}
		return true, nil
	}
	return false, nil
}

// Do not let retained weights hide the additional workspace requirement.
func workloadAdditionalMemory(component Component, resident float64) float64 {
	return max(component.WorkspaceMemoryGiB, component.MemoryGiB-resident)
}

func liveWorkspaceMemory(ctx context.Context, component Component, requested ...float64) float64 {
	budget := component.WorkspaceMemoryGiB
	if component.ID != "flux2" || component.ComposeAsset != "compose.flux2.yaml" || budget <= 0 {
		return budget
	}
	// Missing telemetry must cover a cold text encoder, not the smaller warm
	// workspace. Keep configured non-default reservations as explicit floors.
	configured := budget
	budget = max(budget, 6.5)
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(component.Endpoint, "/")+"/v1/runtime/memory", nil)
	if err != nil {
		return budget
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return budget
	}
	defer resp.Body.Close()
	var state struct {
		WorkspaceGiB float64  `json:"workspace_gib"`
		Schema       int      `json:"memory_schema"`
		Kind         string   `json:"workspace_kind"`
		TextReload   *float64 `json:"text_reload_gib"`
		Generation   *float64 `json:"generation_workspace_gib"`
	}
	if resp.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&state) != nil {
		return budget
	}
	valid := state.Schema == 1 && state.Kind == "additional" && state.WorkspaceGiB > 0 && !math.IsNaN(state.WorkspaceGiB) && !math.IsInf(state.WorkspaceGiB, 0)
	legacy := state.Schema == 0 && (state.WorkspaceGiB == 2.5 || state.WorkspaceGiB == 4.5)
	if valid || legacy {
		if legacy && state.WorkspaceGiB == 4.5 {
			state.WorkspaceGiB = 6.5
		}
		if len(requested) > 0 && requested[0] > 0 {
			if valid && state.TextReload != nil && state.Generation != nil && *state.TextReload >= 0 && *state.Generation > 0 && *state.TextReload+*state.Generation == state.WorkspaceGiB {
				state.WorkspaceGiB = *state.TextReload + max(*state.Generation, requested[0])
			} else {
				state.WorkspaceGiB = max(state.WorkspaceGiB, requested[0])
			}
		}
		// Never reduce an explicitly larger user's workspace reservation.
		if configured > 4.5 && configured != 6.5 {
			return max(configured, state.WorkspaceGiB)
		}
		return state.WorkspaceGiB
	}
	return budget
}

// OnDemandIdle is distinct from a failed running service or a disabled feature.
func (c *Controller) OnDemandIdle(ctx context.Context, bundleID, id string) bool {
	b, ok := c.Catalog().Bundle(bundleID)
	if !ok || !b.WorkloadSwap || workloadGroup(id) == "" {
		return false
	}
	for _, member := range c.workloadBundle(b).Components {
		if member == id {
			x, _ := c.Catalog().ResolveSupport(bundleID, id)
			if x.KeepResident {
				return false
			}
			if x.IsSupport() && (x.Controller != "compose" || !c.local(x)) {
				return false
			}
			state, err := c.inspectComponent(ctx, x)
			return state == "exited" || (err != nil && isMissingContainer(err))
		}
	}
	return false
}
