package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// This exception is confined to the qualified TP1/1M recipe. Its measured
// loading increase was 113.46 GiB (115.33 GiB in the original qualification),
// versus the conservative 116 GiB catalog estimate. Admit at most a 2 GiB
// estimate shortfall, with the real configured reserve enforced during loading.
const qwenFP4StartupEstimateMarginGiB = 2.0

func (c *Controller) startupMemory() SystemMemory {
	if c.memoryProbe != nil {
		return c.memoryProbe()
	}
	return readSystemMemory()
}

func (c *Controller) guardedQwenFP4Local(x Component) bool {
	return c.local(x) && x.Controller == "compose" && x.Role == "llm" &&
		x.ComposeAsset == "compose.qwen-fp4-local.yaml" && x.Model == QwenFP4Local &&
		x.StartupMemoryGiB == 116 && x.MemoryGiB == 103 && x.WorkspaceMemoryGiB == 3
}

func qwenFP4GuardedHeadroom(memory SystemMemory, plan memoryPlan, reserve float64) bool {
	if plan.NeededGiB != 116 || !plan.RequiresCUDAStart || memory.TotalGiB < plan.NeededGiB+reserve {
		return false
	}
	// Keep the immediate CUDA-free check and the user's reserve unchanged.
	plan.NeededGiB -= qwenFP4StartupEstimateMarginGiB
	return validateMemoryHeadroom(memory, plan, reserve) == nil
}

func (c *Controller) validateBundleStartupMemory(bundle Bundle, memory SystemMemory, plan memoryPlan, reserve float64) error {
	err := validateMemoryHeadroom(memory, plan, reserve)
	if err == nil {
		return nil
	}
	if _, ok := c.guardedQwenFP4Bundle(bundle); ok && qwenFP4GuardedHeadroom(memory, plan, reserve) {
		return nil
	}
	return err
}

func (c *Controller) guardedQwenFP4Bundle(bundle Bundle) (Component, bool) {
	var target Component
	for _, x := range c.Catalog().ModelComponents(bundle.ID) {
		if bundle.WorkloadSwap && workloadGroup(x.ID) != "" && !x.KeepResident {
			continue
		}
		if !c.guardedQwenFP4Local(x) || target.ID != "" {
			return Component{}, false
		}
		target = x
	}
	return target, target.ID != ""
}

// Called only during an actual start, after idle auxiliaries have been stopped.
// Probes and status requests never drop caches or terminate applications.
func (c *Controller) reclaimQwenFP4StartupCache(ctx context.Context, x Component) error {
	c.updateOperation(x.ID, progressInfo{Key: "fp4-cache", Phase: "FP4-Local 기동 메모리 회수", Detail: "파일 캐시를 반환한 뒤 실제 가용 메모리를 다시 검사합니다."})
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	_, err := executeHost(ctx, c.host(x.Host), nil, "docker", "run", "--rm", "--network", "none", "--privileged", "--entrypoint", "sh", "local/qwen38-fp4-local:csf-yarn-20261016", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches")
	if err != nil {
		return fmt.Errorf("FP4-Local 파일 캐시 반환 실패: %w", err)
	}
	return nil
}

func (c *Controller) prepareQwenFP4Startup(ctx context.Context, bundle Bundle, reserve float64) error {
	plan := c.bundleMemoryPlan(ctx, bundle)
	if validateMemoryHeadroom(c.startupMemory(), plan, reserve) == nil {
		return nil
	}
	if x, ok := c.guardedQwenFP4Bundle(bundle); ok && qwenFP4GuardedHeadroom(c.startupMemory(), plan, reserve) && c.componentNeedsStart(ctx, x) {
		return c.reclaimQwenFP4StartupCache(ctx, x)
	}
	return nil // The ordinary headroom check reports an actual capacity shortage.
}

type qwenFP4StartupPressure struct{ available, reserve float64 }

func (e *qwenFP4StartupPressure) Error() string {
	return fmt.Sprintf("FP4-Local 기동 중 가용 메모리 %.2f GiB가 최소 여유 %.2f GiB 아래로 내려가 모델을 자동 정리했습니다", e.available, e.reserve)
}

func (c *Controller) stopQwenFP4Startup(x Component) error {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	_, err := executeHost(ctx, c.host(x.Host), nil, "docker", "stop", "--time", "1", x.Container)
	if isMissingContainer(err) {
		return nil
	}
	return err
}

func (c *Controller) watchQwenFP4Startup(parent context.Context, x Component, reserve float64) error {
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	result := make(chan error, 1)
	go func() { result <- c.startAndWaitContext(ctx, x) }()
	// A 250 ms sample interval catches the short repacking peak on UMA hardware.
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	cleanup := func(cause error, wait bool) error {
		cancel()
		stopErr := c.stopQwenFP4Startup(x)
		if wait {
			<-result
		}
		if stopErr != nil {
			return fmt.Errorf("%v; 모델 자동 정리 실패: %w", cause, stopErr)
		}
		return cause
	}
	for {
		select {
		case err := <-result:
			if err != nil {
				return cleanup(err, false)
			}
			if m := c.startupMemory(); m.AvailableGiB < reserve {
				return cleanup(&qwenFP4StartupPressure{m.AvailableGiB, reserve}, false)
			}
			return nil
		case <-ticker.C:
			if m := c.startupMemory(); m.AvailableGiB < reserve {
				return cleanup(&qwenFP4StartupPressure{m.AvailableGiB, reserve}, true)
			}
		case <-parent.Done():
			return cleanup(parent.Err(), true)
		}
	}
}

func (c *Controller) startWithMemoryWatch(ctx context.Context, x Component, reserve float64) error {
	if !c.guardedQwenFP4Local(x) || c.isHealthy(ctx, x) {
		return c.startAndWaitContext(ctx, x)
	}
	reserve = normalizedMemoryReserve(reserve)
	if configured := c.host(x.Host).MemoryReserveGiB; configured > 0 {
		reserve = configured
	}
	if c.componentRunning(ctx, x) {
		// Replacing an unhealthy instance releases its actual allocation first.
		if err := c.stopQwenFP4Startup(x); err != nil {
			return err
		}
	}
	for attempt := 0; attempt < 2; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		plan := componentStartMemoryPlan(x, 0)
		memory := c.startupMemory()
		if err := validateMemoryHeadroom(memory, plan, reserve); err != nil && !qwenFP4GuardedHeadroom(memory, plan, reserve) {
			return err
		}
		err := c.watchQwenFP4Startup(ctx, x, reserve)
		var pressure *qwenFP4StartupPressure
		if !errors.As(err, &pressure) || attempt == 1 {
			return err
		}
		c.updateOperation(x.ID, progressInfo{Key: "fp4-retry", Phase: "FP4-Local 메모리 회수 후 자동 재시도", Detail: "기동 중 메모리를 정리했습니다. 최소 여유를 유지하며 한 번 더 시작합니다."})
		if err := c.reclaimQwenFP4StartupCache(ctx, x); err != nil {
			return err
		}
	}
	return nil
}
