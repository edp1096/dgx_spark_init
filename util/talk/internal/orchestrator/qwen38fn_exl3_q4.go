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

func (c *Controller) prepareQwen38FNEXL3Q4Headroom(parent context.Context, component Component) error {
	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(component.Endpoint, "/")+"/v1/models", nil)
	if err != nil {
		return err
	}
	response, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("EXL3 4bit 실제 모델 용량 확인: %w", err)
	}
	defer response.Body.Close()
	var card struct {
		Data []struct {
			ID      string `json:"id"`
			Context int    `json:"max_model_len"`
		} `json:"data"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&card) != nil || len(card.Data) != 1 || card.Data[0].ID != component.Model || card.Data[0].Context != 1048576 {
		return fmt.Errorf("EXL3 4bit 실제 모델·문맥이 검증된 1M 설정과 다릅니다")
	}
	c.updateOperation(component.ID, progressInfo{Key: "velo-weight-cache", Phase: "EXL3 4bit 기동 메모리 정리", Detail: "실제 GPU·Q8 KV·YaRN·MTP 설정을 확인하고 닫힌 가중치 파일 캐시만 반환합니다.", Progress: .99})
	out, err := executeHost(ctx, c.host(component.Host), nil, "docker", "exec", component.Container, "python", "/opt/velogb10/release_weight_cache.py")
	if err != nil {
		return fmt.Errorf("EXL3 4bit 기동 확인: %w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}
