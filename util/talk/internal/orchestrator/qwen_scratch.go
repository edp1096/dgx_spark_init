package orchestrator

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// RadixArk 1M leaves reusable CUDA scratch after language-model requests.
// Pause in place and release only unused allocator blocks before bootstrapping
// another CUDA context. Keep model weights, the fixed KV pool and prefix state.
func (c *Controller) reclaimRadixArkScratch(ctx context.Context, b Bundle) error {
	var core Component
	for _, id := range b.Components {
		x, _ := c.Catalog().ResolveComponent(b.ID, id)
		if x.ServiceRole() == "llm" && x.ComposeAsset == "compose.flash-next.yaml" && x.qwenQADVariant() == "radixark" && c.local(x) {
			core = x
			break
		}
	}
	if core.ID == "" || !c.componentRunning(ctx, core) {
		return nil
	}
	control := func(ctx context.Context, path, body string) error {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(core.Endpoint, "/")+path, bytes.NewBufferString(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		response, err := c.client.Do(req)
		if err != nil {
			return err
		}
		defer response.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		if response.StatusCode != 200 {
			return fmt.Errorf("%s HTTP %d: %s", path, response.StatusCode, data)
		}
		return nil
	}
	pauseCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := control(pauseCtx, "/pause_generation", `{"mode":"in_place"}`); err != nil {
		// A timed-out reply can arrive after the engine has already paused.
		recoveryCtx, recoveryCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer recoveryCancel()
		if recoveryErr := control(recoveryCtx, "/continue_generation", `{}`); recoveryErr != nil {
			return fmt.Errorf("Qwen 정리 준비 및 재개 실패: %v; %w", err, recoveryErr)
		}
		return fmt.Errorf("ASR 기동 전 Qwen 임시 메모리 정리: %w", err)
	}
	// Resume even if the requesting job is canceled while cleanup is pending.
	resumeCtx, resumeCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer resumeCancel()
	if err := control(resumeCtx, "/continue_generation", `{"torch_empty_cache":true}`); err != nil {
		recoveryCtx, recoveryCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer recoveryCancel()
		recoveryErr := control(recoveryCtx, "/continue_generation", `{}`)
		if recoveryErr != nil {
			return fmt.Errorf("Qwen 임시 메모리 정리 및 재개 실패: %v; %w", err, recoveryErr)
		}
		return fmt.Errorf("Qwen 임시 메모리 정리: %w", err)
	}
	return nil
}
