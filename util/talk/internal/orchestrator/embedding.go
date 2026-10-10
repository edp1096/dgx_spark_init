package orchestrator

import (
	"context"
	"fmt"
	"sparktalk/internal/embedding"
	"time"
)

func (c *Controller) embeddingCoreRunning(ctx context.Context, b Bundle) bool {
	for _, id := range b.Components {
		x, ok := c.Catalog().ResolveComponent(b.ID, id)
		if ok && x.Role == "llm" {
			return c.componentRunning(ctx, x)
		}
	}
	return false
}

// Shared embedding weights belong only to sets that explicitly contain them.
func (c *Controller) stopUnselectedEmbedding(ctx context.Context, b Bundle) error {
	for _, x := range c.Catalog().Deployments(b.ID) {
		if x.ServiceRole() != "embedding" || !c.local(x) || x.Controller != "compose" {
			continue
		}
		if _, member := c.Catalog().ResolveComponent(b.ID, x.ID); member {
			continue
		}
		if err := c.stopIdleWorkload(ctx, x); err != nil {
			return fmt.Errorf("의미 검색 종료: %w", err)
		}
	}
	return nil
}

// HTTP health is available before the lazy CUDA model has been loaded. A
// resident set is ready only after an actual finite embedding is returned.
func (c *Controller) prepareResidentEmbedding(parent context.Context, x Component) error {
	ctx, cancel := context.WithTimeout(parent, 90*time.Second)
	defer cancel()
	state, err := c.workloadIdleState(ctx, x)
	if err != nil {
		return err
	}
	if state.Ready != nil && *state.Ready {
		return nil
	}
	if err := c.reclaimEmbeddingFileCache(ctx, x); err != nil {
		return err
	}
	memory := readSystemMemory()
	if c.memoryProbe != nil {
		memory = c.memoryProbe()
	}
	if err := validateMemoryHeadroom(memory, memoryPlan{NeededGiB: x.MemoryGiB, RequiresCUDAStart: true, MinimumCUDAFreeGiB: 4}, 1.5); err != nil {
		return err
	}
	c.updateOperation(x.ID, progressInfo{Key: "embedding:load", Phase: "의미 검색 모델 GPU 적재"})
	_, err = embedding.Encode(ctx, x.Endpoint, "query", []embedding.Input{{Text: "의미 검색 준비 확인"}})
	return err
}
