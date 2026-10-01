package orchestrator

import (
	"context"
	"fmt"
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
			groups[group] += s.MemoryGiB
			observedGroups[group] += max(s.MemoryGiB, s.ResidentMemoryGiB+s.WorkspaceMemoryGiB)
			residentAux += s.ResidentMemoryGiB
			if s.ID == "flux2" {
				retainedImage += s.ResidentMemoryGiB
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
		budget += max(residentAux, peak) - maxWorkloadBudget(groups)
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
	startup := startupMemoryPhases{}
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
			if !c.local(x) || x.Controller == "external" {
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
	for _, component := range c.Catalog().Deployments(bundle.ID) {
		if bundle.WorkloadSwap && workloadGroup(component.ID) != "" {
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
			if component.ComposeAsset == "compose.flux2.yaml" {
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
}

func (p *startupMemoryPhases) add(component Component, current float64) {
	startup := component.startupMemoryGiB()
	retained := min(startup, component.MemoryGiB)
	additional := max(0, retained-current)
	p.retained += additional
	p.transient = max(p.transient, max(0, startup-current)-additional)
}

func (p startupMemoryPhases) needed() float64 { return p.retained + p.transient }

// Healthy LLM, ASR and TTS services have already loaded their steady-state
// weights. Their host-side allocations are included in MemAvailable but are
// not reported by nvidia-smi, so subtracting GPU usage from the catalog peak
// would count that memory twice. FLUX is different: its API becomes healthy
// before the generation model is loaded and still needs its remaining peak.
func healthyComponentRemainingMemory(component Component, residentMemory float64) float64 {
	if component.Role != "image" {
		return 0
	}
	if residentMemory <= 0 {
		return component.MemoryGiB
	}
	return workloadAdditionalMemory(component, residentMemory)
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
		minimum := immediateFreeReserve(reserveGiB)
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
