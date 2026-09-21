package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// SGLang can silently cap the actual KV pool below context_length. A healthy
// HTTP endpoint alone is not proof that the configured context fits.
func (c *Controller) checkQwenCapacity(ctx context.Context, component Component) error {
	if component.ComposeAsset != "compose.flash-next.yaml" {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	endpoint := strings.TrimSuffix(strings.TrimRight(component.HealthURL, "/"), "/health") + "/server_info"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("QAD KV 용량 확인 실패: %w", err)
	}
	defer resp.Body.Close()
	var info struct {
		Context  int `json:"context_length"`
		Capacity int `json:"max_total_num_tokens"`
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("QAD KV 용량 확인 HTTP %d", resp.StatusCode)
	}
	if err = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&info); err != nil {
		return fmt.Errorf("QAD KV 용량 응답 오류: %w", err)
	}
	if info.Context <= 0 || info.Capacity <= 0 {
		return fmt.Errorf("QAD 실제 KV 용량을 확인할 수 없습니다")
	}
	if info.Capacity < info.Context {
		return fmt.Errorf("QAD 문맥 설정 %d토큰에 비해 실제 KV 용량은 %d토큰입니다. 부가 서비스를 중지하고 Qwen을 먼저 재기동하세요", info.Context, info.Capacity)
	}
	return nil
}
