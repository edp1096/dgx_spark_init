package server

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// All active chat loops are cancelled before the inference queue is drained.
func (s *Server) trackGeneration(parent context.Context) (context.Context, func(), error) {
	s.generationMu.Lock()
	defer s.generationMu.Unlock()
	if s.queueClearing {
		return nil, nil, fmt.Errorf("비상 큐 비우기 중입니다. 완료 후 다시 요청하세요")
	}
	ctx, cancel := context.WithCancel(parent)
	if s.generations == nil {
		s.generations = make(map[uint64]context.CancelFunc)
	}
	s.generationID++
	id := s.generationID
	s.generations[id] = cancel
	return ctx, func() { cancel(); s.generationMu.Lock(); delete(s.generations, id); s.generationMu.Unlock() }, nil
}

func clearInferenceQueue(ctx context.Context, endpoint, key string) error {
	base := strings.TrimSuffix(strings.TrimRight(endpoint, "/"), "/v1")
	client := &http.Client{Timeout: 10 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	// Only a missing endpoint permits trying the other engine's API.
	for _, candidate := range []struct{ path, body string }{{"/abort_request", `{"abort_all":true}`}, {"/abort_requests", `{"request_ids":[]}`}} {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, base+candidate.path, bytes.NewBufferString(candidate.body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("모델 서버 큐 취소 실패: %w", err)
		}
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		resp.Body.Close()
		if resp.StatusCode >= 200 && resp.StatusCode < 300 {
			return nil
		}
		if resp.StatusCode == 404 || resp.StatusCode == 405 {
			continue
		}
		return fmt.Errorf("모델 서버 큐 취소 HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return fmt.Errorf("Talk 생성은 중단했지만 모델 서버가 전체 취소 API를 제공하지 않습니다. 서버에 취소 API를 활성화하거나 모델 세트를 재시작해야 합니다")
}

func (s *Server) emergencyQueue(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	s.generationMu.Lock()
	if s.queueClearing {
		s.generationMu.Unlock()
		http.Error(w, "이미 큐를 비우는 중입니다", http.StatusConflict)
		return
	}
	s.queueClearing = true
	for _, cancel := range s.generations {
		cancel()
	}
	s.generationMu.Unlock()
	defer func() { s.generationMu.Lock(); s.queueClearing = false; s.generationMu.Unlock() }()
	cfg, _ := s.snapshot()
	// Browser disconnection must not interrupt emergency cleanup.
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Second)
	defer cancel()
	if err := clearInferenceQueue(ctx, cfg.Model.Endpoint, cfg.Model.APIKey); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	io.WriteString(w, `{"message":"현재 모델 서버의 실행·대기 요청 취소를 요청했습니다."}`)
}
