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
	for _, component := range c.Catalog().Deployments(bundle.ID) {
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
				plan.NeededGiB += component.MemoryGiB
				plan.FreedGiB += gpuMemory + containerAnonymousMemoryGiB(ctx, component.Container)
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			if !running {
				needsStart = true
				plan.NeededGiB += component.MemoryGiB
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			healthy := c.isHealthy(ctx, component)
			if !healthy {
				needsStart = true
				// Restarting releases the current allocation before rebuilding it.
				plan.NeededGiB += max(0, component.MemoryGiB-gpuMemory)
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			resident := gpuMemory
			if component.ComposeAsset == "compose.flux2.yaml" {
				resident += containerAnonymousMemoryGiB(ctx, component.Container)
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
	return max(0, component.MemoryGiB-residentMemory)
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
