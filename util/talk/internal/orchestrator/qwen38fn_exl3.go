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

func (c *Controller) prepareQwen38FNEXL3Headroom(parent context.Context, component Component) error {
	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(component.Endpoint, "/")+"/health", nil)
	if err != nil {
		return err
	}
	response, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("TensorFold EXL3 준비 상태 확인: %w", err)
	}
	defer response.Body.Close()
	var card struct {
		OK      bool   `json:"ok"`
		Backend string `json:"backend"`
		Context int    `json:"context_length"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&card) != nil || !card.OK || card.Backend != "tensorfold" || card.Context != 1048576 {
		return fmt.Errorf("TensorFold EXL3 실제 문맥이 검증된 1M 설정과 다릅니다")
	}
	out, err := executeHost(ctx, c.host(component.Host), nil, "docker", "exec", component.Container, "cat", "/runtime/runtime-profile.json")
	if err != nil {
		return fmt.Errorf("TensorFold EXL3 실제 모델 설정 확인: %w", err)
	}
	var profile struct {
		Model           string  `json:"model"`
		Context         int     `json:"context"`
		KV              string  `json:"kv"`
		Depth           int     `json:"depth"`
		Vision          bool    `json:"vision"`
		Confidence      float64 `json:"confidence"`
		DraftVocabulary int     `json:"draft_vocabulary"`
		ExpertTiles     string  `json:"expert_tiles"`
		MOEWindow       int     `json:"moe_window"`
	}
	if json.Unmarshal(out, &profile) != nil || profile.Model != component.Model || profile.Context != 1048576 || profile.KV != "int8" || profile.Depth != 5 || profile.Confidence != 0.4 || profile.DraftVocabulary != 86218 || profile.ExpertTiles != "compact-parallel-v1" || profile.MOEWindow != 2048 || !profile.Vision {
		return fmt.Errorf("TensorFold EXL3 실제 모델·문맥·KV·MTP·비전 설정이 다릅니다")
	}
	c.updateOperation(component.ID, progressInfo{Key: "native-ready", Phase: "TensorFold EXL3 1M 준비 완료", Detail: "실제 문맥·Q8 KV 1M·MTP 1–5·비전을 확인했습니다.", Progress: 1})
	return nil
}
