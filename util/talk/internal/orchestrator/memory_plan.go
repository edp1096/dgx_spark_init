package orchestrator

import (
	"context"
	"fmt"
	"sort"
)

func normalizedMemoryReserve(reserveGiB float64) float64 {
	if reserveGiB <= 0 {
		return 4
	}
	return reserveGiB
}

// The catalog is a cold-start estimate. A running allocation must never be
// displayed as smaller than it actually is. For alternating auxiliaries,
// account for both the largest workload and all currently retained services.
func observedBundleBudget(catalog Catalog, bundle Bundle, statuses []ComponentStatus) float64 {
	budget := bundle.MemoryGiB
	groups := map[string]float64{}
	observedGroups := map[string]float64{}
	residentAux := 0.0
	retainedImage := 0.0
	residentBudget, residentObserved := 0.0, 0.0
	for _, id := range bundle.Components {
		component, ok := catalog.ResolveComponent(bundle.ID, id)
		if !ok {
			continue
		}
		s := ComponentStatus{Component: component}
		for _, status := range statuses {
			if status.ID == id {
				s = status
				break
			}
		}
		if s.Host != "local" && s.Host != "" {
			continue
		}
		group := workloadGroup(s.ID)
		if bundle.WorkloadSwap && group != "" {
			if s.KeepResident {
				fixed := residentWorkloadMemory(s.Component)
				observed := max(fixed, s.ResidentMemoryGiB)
				residentBudget += fixed
				residentObserved += observed
				groups[group] += workloadAdditionalMemory(s.Component, fixed)
				observedGroups[group] += workloadAdditionalMemory(s.Component, observed)
			} else {
				groups[group] += s.MemoryGiB
				observedGroups[group] += max(s.MemoryGiB, s.ResidentMemoryGiB+s.WorkspaceMemoryGiB)
				residentAux += s.ResidentMemoryGiB
				if s.ID == "flux2" {
					retainedImage += s.ResidentMemoryGiB
				}
			}
		} else {
			if s.MemoryMeasured && s.Health == "online" && s.EngineMemory != nil && s.EngineMemory.valid(s.EngineMemory.Context, s.EngineMemory.Capacity) {
				// A loaded fixed KV pool is already resident. Replace the cold
				// profile with the observed allocation and only its unretained
				// high-water allowance; never add individual allocator subsets.
				extra := max(s.WorkspaceMemoryGiB, s.EngineMemory.PeakReserved-s.EngineMemory.Reserved)
				budget += s.ResidentMemoryGiB + extra - s.MemoryGiB
			} else {
				budget += max(0, s.ResidentMemoryGiB-s.MemoryGiB)
			}
		}
	}
	if bundle.WorkloadSwap {
		peak := observedGroups["image"]
		for group, value := range observedGroups {
			if group != "image" {
				peak = max(peak, retainedImage+value)
			}
		}
		budget += residentObserved - residentBudget + max(residentAux, peak) - maxWorkloadBudget(groups)
	}
	return budget
}

func immediateFreeReserve(reserveGiB float64) float64 {
	if reserveGiB < minimumCUDAImmediateFreeGiB {
		return reserveGiB
	}
	return minimumCUDAImmediateFreeGiB
}

func isCUDAComponent(component Component) bool {
	return component.Role == "llm" || component.Role == "image" || component.Role == "asr" || component.Role == "tts"
}

func (c *Controller) bundleMemoryPlan(ctx context.Context, bundle Bundle) memoryPlan {
	bundle = c.Catalog().StartBundleMembers(bundle)
	desired := make(map[string]struct{}, len(bundle.Components))
	for _, id := range bundle.Components {
		component, _ := c.Catalog().ResolveComponent(bundle.ID, id)
		desired[component.DeploymentKey()] = struct{}{}
	}
	llmNeedsStart := false
	for _, id := range bundle.Components {
		component, _ := c.Catalog().ResolveComponent(bundle.ID, id)
		if component.Role == "llm" && c.local(component) && c.componentNeedsStart(ctx, component) {
			llmNeedsStart = true
		}
	}
	needsStart := false
	gpuByPID := gpuMemoryByPID(ctx)
	plan := memoryPlan{}
	// runBundleStart waits for each service before starting the next one.
	// Retained allocations accumulate; temporary loading peaks do not overlap.
	startup := startupMemoryPhases{ordered: c.Catalog().qad512DiTResident(bundle)}
	if bundle.WorkloadSwap && llmNeedsStart {
		full, _ := c.Catalog().Bundle(bundle.ID)
		full = c.workloadBundle(full)
		// These services are not loaded at bundle startup. Their actual request
		// budget is checked when invoked (ASR also depends on decoded length).
		for _, id := range full.Components {
			if workloadGroup(id) == "" {
				continue
			}
			x, _ := c.Catalog().ResolveSupport(bundle.ID, id)
			if x.KeepResident || !c.local(x) || x.Controller == "external" {
				continue
			}
			if c.componentRunning(ctx, x) {
				for _, pid := range containerPIDs(ctx, x.Container) {
					plan.FreedGiB += gpuByPID[pid]
				}
				plan.FreedGiB += containerHostResidentMemoryGiB(ctx, x.Container)
			}
		}
	}
	deployments := c.Catalog().Deployments(bundle.ID)
	if startup.ordered {
		ordered := c.Catalog().startupOrder(bundle)
		rank := map[string]int{}
		for i, id := range ordered {
			rank[id] = i + 1
		}
		sort.SliceStable(deployments, func(i, j int) bool {
			return rank[deployments[i].ID] < rank[deployments[j].ID]
		})
	}
	for _, component := range deployments {
		if bundle.WorkloadSwap && workloadGroup(component.ID) != "" && !component.KeepResident {
			continue
		}
		if !c.local(component) || component.Controller == "external" || (component.IsSupport() && !bundle.StartSupport) {
			continue
		}
		running := c.componentRunning(ctx, component)
		gpuMemory := 0.0
		if running {
			for _, pid := range containerPIDs(ctx, component.Container) {
				gpuMemory += gpuByPID[pid]
			}
		}
		if _, wanted := desired[component.DeploymentKey()]; wanted {
			if running && component.StartAfterLLM && llmNeedsStart {
				// runBundleStart stops deferred services before loading the LLM.
				needsStart = true
				startup.add(component, 0)
				plan.FreedGiB += gpuMemory + containerHostResidentMemoryGiB(ctx, component.Container)
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			if !running {
				needsStart = true
				startup.add(component, 0)
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			healthy := !c.componentNeedsStart(ctx, component)
			if !healthy {
				needsStart = true
				// Restarting releases the current allocation before rebuilding it.
				startup.add(component, gpuMemory+containerHostResidentMemoryGiB(ctx, component.Container))
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			resident := gpuMemory
			if component.Role == "image" {
				resident += containerHostResidentMemoryGiB(ctx, component.Container)
			}
			plan.NeededGiB += healthyComponentRemainingMemory(component, resident)
			continue
		}
		if component.Role == "llm" && running {
			if gpuMemory <= 0 {
				gpuMemory = component.MemoryGiB
			}
			plan.FreedGiB += gpuMemory
		}
	}
	plan.NeededGiB += startup.needed()
	if !needsStart {
		plan.NeededGiB = 0
	}
	return plan
}

// Serving budgets below the loading peak identify transient staging. When a
// startup budget is smaller (lazy workspace), retain the startup budget rather
// than reserving optional work that has not been requested. Counting all retained
// allocations plus the largest transient remains conservative for any start order.
type startupMemoryPhases struct {
	retained, transient float64
	ordered             bool
	peak                float64
}

func (p *startupMemoryPhases) add(component Component, current float64) {
	startup := component.startupMemoryGiB()
	retained := min(startup, component.MemoryGiB)
	additional := max(0, retained-current)
	p.peak = max(p.peak, p.retained+max(0, startup-current))
	p.retained += additional
	p.transient = max(p.transient, max(0, startup-current)-additional)
}

func (p startupMemoryPhases) needed() float64 {
	if p.ordered {
		return max(p.retained, p.peak)
	}
	return p.retained + p.transient
}

// The tested preset guarantees LLM loading finishes before resident speech and
// the single image DiT load. Do not relax admission for unrelated profiles.
func (c Catalog) qad512DiTResident(bundle Bundle) bool {
	if bundle.ID != "flash-next" || bundle.ContextTokens != 524288 {
		return false
	}
	llm, ok := c.ResolveComponent(bundle.ID, "flash-next")
	image, imageOK := c.ResolveComponent(bundle.ID, "flux2")
	return ok && imageOK && llm.Model == QwenQADHuihuiLIL && llm.RuntimeOptions["MTP_TOKENS"] == "3" && llm.RuntimeOptions["MAX_MODEL_LEN"] == "524288" && image.ComposeAsset == "compose.qwen-image21.yaml" && image.RuntimeOptions["IMAGE_RESIDENCY"] == "dit" && image.KeepResident && image.StartAfterLLM
}

// This is startup admission, not admission for a future image generation job.
// Healthy speech/LLM allocations are already reflected in MemAvailable. A lazy
// image API may still need its retained startup allocation, but generation
// scratch is checked separately by AcquireWorkload. Loading transients are also
// unnecessary for an already healthy service.
func healthyComponentRemainingMemory(component Component, residentMemory float64) float64 {
	if component.Role != "image" {
		return 0
	}
	retainedStartup := min(component.startupMemoryGiB(), component.MemoryGiB)
	return max(0, retainedStartup-residentMemory)
}

func validateMemoryHeadroom(memory SystemMemory, plan memoryPlan, reserveGiB float64) error {
	if plan.NeededGiB <= 0 {
		return nil
	}
	projected := memory.AvailableGiB + plan.FreedGiB - plan.NeededGiB
	if projected < reserveGiB {
		return fmt.Errorf(
			"통합메모리 부족 예상: 시스템 가용 %.1f GiB, 반환 예정 %.1f GiB, 추가 예상 %.1f GiB, 기동 후 약 %.1f GiB (최소 여유 %.1f GiB)",
			memory.AvailableGiB, plan.FreedGiB, plan.NeededGiB, projected, reserveGiB,
		)
	}
	if plan.RequiresCUDAStart {
		immediate := memory.FreeGiB + plan.FreedGiB
		minimum := max(immediateFreeReserve(reserveGiB), plan.MinimumCUDAFreeGiB)
		if immediate < minimum {
			return &cudaStartMemoryError{fmt.Sprintf(
				"CUDA 기동용 즉시 여유 메모리 부족: 현재 %.1f GiB, 반환 예정 포함 %.1f GiB (최소 %.1f GiB, 시스템 가용 %.1f GiB)",
				memory.FreeGiB, immediate, minimum, memory.AvailableGiB,
			)}
		}
	}
	return nil
}

// Loading and serving are separate phases. Startup may be smaller (deferred
// ASR workspace) or larger (weight staging) than the serving budget.
func (c Component) startupMemoryGiB() float64 {
	if c.StartupMemoryGiB > 0 {
		return c.StartupMemoryGiB
	}
	return c.MemoryGiB
}
