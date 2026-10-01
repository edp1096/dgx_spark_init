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
func (c *Controller) checkSGLangCapacity(ctx context.Context, component Component) error {
	label := "QAD"
	if component.ComposeAsset != "compose.flash-next.yaml" {
		if component.Controller != "glm53-cluster" || component.ProgressKind != "sglang" {
			return nil
		}
		label = "GLM"
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
		return fmt.Errorf("%s KV 용량 확인 실패: %w", label, err)
	}
	defer resp.Body.Close()
	var info struct {
		Context  int `json:"context_length"`
		Capacity int `json:"max_total_num_tokens"`
		States   []struct {
			Memory *EngineMemory `json:"qad_memory"`
		} `json:"internal_states"`
	}
	if resp.StatusCode != 200 {
		return fmt.Errorf("%s KV 용량 확인 HTTP %d", label, resp.StatusCode)
	}
	if err = json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&info); err != nil {
		return fmt.Errorf("%s KV 용량 응답 오류: %w", label, err)
	}
	if info.Context <= 0 || info.Capacity <= 0 {
		return fmt.Errorf("%s 실제 KV 용량을 확인할 수 없습니다", label)
	}
	if info.Capacity < info.Context {
		return &modelCapacityError{Model: label, Context: info.Context, Capacity: info.Capacity}
	}
	if component.ComposeAsset == "compose.flash-next.yaml" {
		var receipt *EngineMemory
		if len(info.States) == 1 {
			receipt = info.States[0].Memory
		}
		c.recordEngineMemory(component, receipt, info.Context, info.Capacity)
	}
	return nil
}

type modelCapacityError struct {
	Model             string
	Context, Capacity int
}

func (e *modelCapacityError) Error() string {
	if e.Model == "GLM" {
		return fmt.Sprintf("GLM 문맥 설정 %d토큰에 비해 실제 KV 용량은 %d토큰입니다. GLM 기동 설정과 메모리 여유를 확인하세요", e.Context, e.Capacity)
	}
	return fmt.Sprintf("QAD 문맥 설정 %d토큰에 비해 실제 KV 용량은 %d토큰입니다. 세트 시작으로 Qwen을 먼저 재기동하세요", e.Context, e.Capacity)
}
