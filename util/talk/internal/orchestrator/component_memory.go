package orchestrator

import (
	"context"
	"fmt"
)

// A single-service action never stops another LLM, so it must not borrow the
// release credit used by a whole-set switch. Keep TP2/remote startup unchanged.
func (c *Controller) checkLocalComponentStart(ctx context.Context, target Component, restart bool, reserveGiB float64, bundleID string) error {
	if !c.local(target) || target.isCluster() || !isCUDAComponent(target) {
		return nil
	}
	running := c.componentRunning(ctx, target)
	if running && !restart && c.isHealthy(ctx, target) {
		return nil
	}
	gpu := gpuMemoryByPID(ctx)
	allocated := func(component Component) float64 {
		total := 0.0
		for _, pid := range containerPIDs(ctx, component.Container) {
			total += gpu[pid]
		}
		return total
	}
	current := 0.0
	if running {
		current = allocated(target) + containerAnonymousMemoryGiB(ctx, target.Container)
	}
	plan := componentStartMemoryPlan(target, current)
	for _, component := range c.Catalog().Deployments(bundleID) {
		if component.DeploymentKey() == target.DeploymentKey() || !c.local(component) || component.Controller == "external" || component.Role != "image" {
			continue
		}
		if c.componentRunning(ctx, component) {
			resident := allocated(component)
			if component.ComposeAsset == "compose.flux2.yaml" {
				resident += containerAnonymousMemoryGiB(ctx, component.Container)
			}
			plan.NeededGiB += healthyComponentRemainingMemory(component, resident)
		}
	}
	reserve := c.host(target.Host).MemoryReserveGiB
	if reserve <= 0 {
		reserve = normalizedMemoryReserve(reserveGiB)
	}
	if err := validateMemoryHeadroom(readSystemMemory(), plan, reserve); err != nil {
		return fmt.Errorf("%s 개별 기동: %w", target.Name, err)
	}
	return nil
}

func componentStartMemoryPlan(target Component, currentGPU float64) memoryPlan {
	return memoryPlan{NeededGiB: target.MemoryGiB, FreedGiB: currentGPU, RequiresCUDAStart: isCUDAComponent(target)}
}
