package orchestrator

import (
	"context"
	"fmt"
	"time"
)

// The CPU worker is independent of GPU swapping, but follows explicit set stops.
func (c *Controller) EnsureShortsWorker(ctx context.Context, bundle string, reserve float64) error {
	x, ok := c.Catalog().ResolveSupport(bundle, "extra-shorts")
	if !ok {
		return fmt.Errorf("Shorts 서비스가 등록되지 않았습니다")
	}
	if c.isHealthy(ctx, x) {
		return nil
	}
	b, exists := c.Catalog().Bundle(bundle)
	if !exists || !c.embeddingCoreRunning(ctx, b) {
		return fmt.Errorf("모델 세트가 중지되어 있습니다. 세트를 시작한 뒤 쇼츠 기능을 사용하세요")
	}
	if !c.local(x) || x.Controller != "compose" {
		return fmt.Errorf("Shorts 서비스는 현재 로컬 관리 모드가 필요합니다")
	}
	if err := c.begin(Operation{Action: "start", ComponentID: x.ID, State: "running", Phase: "Shorts 준비", StartedAt: time.Now()}); err != nil {
		return err
	}
	err := c.checkLocalComponentStart(ctx, x, false, reserve, bundle)
	if err == nil {
		err = c.startAndWaitContext(ctx, x)
	}
	if err != nil {
		c.finishOperation("failed", err.Error())
	} else {
		c.finishOperation("complete", "")
	}
	return err
}
