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
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(component.Endpoint, "/")+"/v1/model", nil)
	if err != nil {
		return err
	}
	response, err := c.client.Do(req)
	if err != nil {
		return fmt.Errorf("EXL3 실제 모델 용량 확인: %w", err)
	}
	defer response.Body.Close()
	var card struct {
		ID         string `json:"id"`
		Parameters struct {
			Context int    `json:"max_seq_len"`
			Cache   int    `json:"cache_size"`
			Mode    string `json:"cache_mode"`
			Vision  bool   `json:"use_vision"`
		} `json:"parameters"`
	}
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&card) != nil {
		return fmt.Errorf("EXL3 실제 모델 용량 응답 확인 실패")
	}
	if card.ID != component.Model || card.Parameters.Context != 1048576 || card.Parameters.Cache != 1048576 || card.Parameters.Mode != "Q8" || !card.Parameters.Vision {
		return fmt.Errorf("EXL3 실제 모델·문맥·KV·비전이 검증된 1M 설정과 다릅니다")
	}
	c.updateOperation(component.ID, progressInfo{Key: "native-weight-cache", Phase: "EXL3 기동 메모리 정리", Detail: "GPU 적재가 끝난 일반 가중치 파일 캐시를 반환합니다. PLE·KV는 유지합니다.", Progress: .99})
	out, err := executeHost(ctx, c.host(component.Host), nil, "docker", "exec", component.Container, "python", "/opt/sparktalk-qwen38fn_exl3/release_weight_cache.py")
	if err != nil {
		return fmt.Errorf("EXL3 기동 메모리 정리: %w: %s", err, strings.TrimSpace(string(out)))
	}
	c.updateOperation(component.ID, progressInfo{Key: "native-ready", Phase: "EXL3 1M 준비 완료", Detail: "실제 문맥·Q8 KV 1M과 비전을 확인했습니다.", Progress: 1})
	return nil
}
