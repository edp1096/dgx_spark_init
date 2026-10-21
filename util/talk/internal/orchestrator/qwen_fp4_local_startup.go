package orchestrator

import (
	"context"
	"fmt"
	"time"
)

func (c *Controller) startupMemory() SystemMemory {
	if c.memoryProbe != nil {
		return c.memoryProbe()
	}
	return readSystemMemory()
}

func (c *Controller) managedQwenTensorFold(x Component) bool {
	return c.local(x) && x.Controller == "compose" && x.Role == "llm" && x.ProgressKind == "tensorfold" &&
		(x.ComposeAsset == "compose.qwen-fp4-local.yaml" || x.ComposeAsset == "compose.qwen38fn_exl3.yaml")
}

func (c *Controller) validateBundleStartupMemory(bundle Bundle, memory SystemMemory, plan memoryPlan, reserve float64) error {
	return validateMemoryHeadroom(memory, plan, c.Catalog().minimumBundleReserve(bundle, reserve))
}

type qwenFP4StartupPressure struct{ available, reserve float64 }

func (e *qwenFP4StartupPressure) Error() string {
	return fmt.Sprintf("TensorFold 기동 중 가용 메모리 %.2f GiB가 최소 여유 %.2f GiB 아래로 내려가 모델을 자동 정리했습니다", e.available, e.reserve)
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
	if !c.managedQwenTensorFold(x) || c.isHealthy(ctx, x) {
		return c.startAndWaitContext(ctx, x)
	}
	reserve = max(4, normalizedMemoryReserve(reserve), c.host(x.Host).MemoryReserveGiB)
	if c.componentRunning(ctx, x) {
		if err := c.stopQwenFP4Startup(x); err != nil {
			return err
		}
	}
	if err := validateMemoryHeadroom(c.startupMemory(), componentStartMemoryPlan(x, 0), reserve); err != nil {
		return err
	}
	return c.watchQwenFP4Startup(ctx, x, reserve)
}
