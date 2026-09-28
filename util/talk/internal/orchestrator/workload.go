package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// QAD keeps the LLM resident. Image and speech/media jobs own the auxiliary
// processes exclusively; stopping containers also releases unified CPU memory.
func workloadGroup(id string) string {
	switch id {
	case "flux2":
		return "image"
	case "nemotron-asr", "extra-media":
		return "speech"
	}
	return ""
}

func (c *Controller) workloadBudget(b Bundle) float64 {
	groups := map[string]float64{}
	for _, id := range b.Components {
		if g := workloadGroup(id); g != "" {
			x, _ := c.Catalog().ResolveComponent(b.ID, id)
			groups[g] += x.MemoryGiB
		}
	}
	return max(groups["image"], groups["speech"])
}

func (c *Controller) stopWorkloads(ctx context.Context, b Bundle) error {
	// Startup filters shared support out of b; ownership still includes Media.
	if full, ok := c.Catalog().Bundle(b.ID); ok {
		b = full
	}
	for _, id := range b.Components {
		if workloadGroup(id) == "" {
			continue
		}
		x, _ := c.Catalog().ResolveComponent(b.ID, id)
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
		if err := c.stopComponent(ctx, x); err != nil {
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
	if !ok || !b.WorkloadSwap || workloadGroup(target) == "" {
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
			if ctx.Err() != nil || operationErr != nil || memory.AvailableGiB < normalizedMemoryReserve(reserve) {
				c.updateOperation(target, progressInfo{Key: "workload:cleanup", Phase: "작업 서비스 메모리 반환"})
				cleanupErr = c.stopWorkloads(cleanupCtx, b)
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
		x, _ := c.Catalog().ResolveComponent(b.ID, id)
		if id == target && id == "nemotron-asr" && len(requestedPeak) > 0 && requestedPeak[0] > 0 {
			x.MemoryGiB = requestedPeak[0]
		}
		start = append(start, x)
	}
	cacheReclaimed := false
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
				x.WorkspaceMemoryGiB = liveWorkspaceMemory(ctx, x)
				for _, pid := range containerPIDs(ctx, x.Container) {
					resident += gpu[pid]
				}
				resident += containerHostResidentMemoryGiB(ctx, x.Container)
			}
			additional += workloadAdditionalMemory(x, resident)
		}
		memory := readSystemMemory()
		if c.memoryProbe != nil {
			memory = c.memoryProbe()
		}
		minimum := normalizedMemoryReserve(reserve)
		err := validateMemoryHeadroom(memory, memoryPlan{NeededGiB: additional, RequiresCUDAStart: coldCUDA}, minimum)
		if err == nil && memory.AvailableGiB < minimum {
			err = fmt.Errorf("작업 중 최소 여유 부족: 가용 %.1f GiB, 최소 %.1f GiB", memory.AvailableGiB, minimum)
		}
		if err == nil {
			break
		}
		reclaimed := false
		for _, id := range b.Components {
			if workloadGroup(id) == "" || workloadGroup(id) == workloadGroup(target) {
				continue
			}
			x, _ := c.Catalog().ResolveComponent(b.ID, id)
			if !c.componentRunning(ctx, x) {
				continue
			}
			c.updateOperation(id, progressInfo{Key: "workload:reclaim:" + id, Phase: "유휴 부가 서비스 메모리 회수", Detail: x.Name})
			if stopErr := c.stopComponent(ctx, x); stopErr != nil {
				return fail(stopErr)
			}
			state, probeErr := c.inspectComponent(ctx, x)
			if probeErr != nil || state != "exited" {
				return fail(fmt.Errorf("%s 회수 확인 실패: %s %v", x.Name, state, probeErr))
			}
			reclaimed = true
		}
		if time.Now().After(deadline) {
			return fail(fmt.Errorf("작업 전 메모리 확인(유휴 경쟁 서비스 회수 후): %w", err))
		}
		if reclaimed {
			continue
		}
		if coldCUDA && memory.FreeGiB < immediateFreeReserve(minimum) && !cacheReclaimed {
			c.updateOperation(target, progressInfo{Key: "workload:file-cache", Phase: "종료한 부가 모델 파일 캐시 반환"})
			if reclaimErr := c.reclaimAuxiliaryFileCache(ctx, b); reclaimErr != nil {
				return fail(reclaimErr)
			}
			cacheReclaimed = true
			continue
		}
		select {
		case <-ctx.Done():
			return fail(ctx.Err())
		case <-time.After(time.Second):
		}
	}
	for _, x := range start {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		if err := c.startAndWaitContext(ctx, x); err != nil {
			return fail(err)
		}
	}
	c.updateOperation(target, progressInfo{Key: "workload:execute", Phase: "작업 실행 중", Detail: "다른 이미지·음성·미디어 요청은 순서대로 대기합니다."})
	return release, nil
}

// A resident weight cache does not reserve the next request's scratch space.
// Do not let retained weights hide the additional workspace requirement.
func workloadAdditionalMemory(component Component, resident float64) float64 {
	return max(component.WorkspaceMemoryGiB, component.MemoryGiB-resident)
}

func liveWorkspaceMemory(ctx context.Context, component Component) float64 {
	budget := component.WorkspaceMemoryGiB
	if component.ID != "flux2" || component.ComposeAsset != "compose.flux2.yaml" || budget <= 0 {
		return budget
	}
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
		WorkspaceGiB float64 `json:"workspace_gib"`
	}
	if resp.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&state) != nil {
		return budget
	}
	if state.WorkspaceGiB == 2.5 || state.WorkspaceGiB == 4.5 {
		// Never reduce an explicitly larger user's workspace reservation.
		if budget > 4.5 {
			return budget
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
	for _, member := range b.Components {
		if member == id {
			x, _ := c.Catalog().ResolveComponent(bundleID, id)
			state, err := c.inspectComponent(ctx, x)
			return state == "exited" || (err != nil && isMissingContainer(err))
		}
	}
	return false
}
