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
		} else {
			budget += max(0, s.ResidentMemoryGiB-s.MemoryGiB)
		}
	}
	if bundle.WorkloadSwap {
		budget += max(residentAux, max(observedGroups["image"], observedGroups["speech"])) - max(groups["image"], groups["speech"])
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
	if bundle.WorkloadSwap {
		full, _ := c.Catalog().Bundle(bundle.ID)
		// These services are not loaded at bundle startup. Their actual request
		// budget is checked when invoked (ASR also depends on decoded length).
		for _, id := range full.Components {
			if workloadGroup(id) == "" {
				continue
			}
			x, _ := c.Catalog().ResolveComponent(bundle.ID, id)
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
				plan.NeededGiB += component.startupMemoryGiB()
				plan.FreedGiB += gpuMemory + containerHostResidentMemoryGiB(ctx, component.Container)
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			if !running {
				needsStart = true
				plan.NeededGiB += component.startupMemoryGiB()
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			healthy := !c.componentNeedsStart(ctx, component)
			if !healthy {
				needsStart = true
				// Restarting releases the current allocation before rebuilding it.
				plan.NeededGiB += max(0, component.startupMemoryGiB()-gpuMemory-containerHostResidentMemoryGiB(ctx, component.Container))
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
	if !needsStart {
		plan.NeededGiB = 0
	}
	return plan
}

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

// Startup admission does not reserve an optional transcription workspace as
// if it were already allocated during model loading. MemoryGiB retains peak.
func (c Component) startupMemoryGiB() float64 {
	if c.StartupMemoryGiB > 0 {
		return min(c.StartupMemoryGiB, c.MemoryGiB)
	}
	return c.MemoryGiB
}
