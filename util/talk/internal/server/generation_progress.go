package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

type generationProgress struct {
	Event    string   `json:"event"`
	Stage    string   `json:"stage"`
	Kind     string   `json:"kind"`
	Step     *int     `json:"step"`
	Total    *int     `json:"total"`
	Elapsed  float64  `json:"elapsed_seconds"`
	ETA      *float64 `json:"eta_seconds"`
	ETAScope string   `json:"eta_scope"`
}

func (p generationProgress) label() string {
	if p.Event == "idle" {
		return "생성 파일 준비 완료"
	}
	if p.Event == "failed" {
		return "생성 실패"
	}
	if p.Event == "conditioning_cache_hit" {
		return "프롬프트 캐시 재사용"
	}
	labels := map[string]string{"queued": "생성 작업 대기", "preparing": "생성 서비스 준비", "encode": "프롬프트 인코딩", "release_encoder": "인코더 메모리 정리", "sample": "샘플링", "release_sample_workspace": "샘플링 메모리 정리", "decode_video": "영상 디코딩", "release_video_vae": "디코더 메모리 정리", "decode_audio": "음성 디코딩", "release_audio_vae": "음성 디코더 메모리 정리", "save": "생성 파일 저장", "attaching": "대화에 저장", "attached": "대화 저장 완료"}
	label := labels[p.Stage]
	if p.Kind == "qwim" && p.Stage == "decode_video" {
		label = "이미지 디코딩"
	}
	if label == "" {
		label = "생성 진행 중"
	}
	if p.Step != nil && p.Total != nil && *p.Total > 0 {
		label += fmt.Sprintf(" %d/%d", *p.Step, *p.Total)
	}
	if strings.HasSuffix(p.Event, "_end") {
		label += " 완료"
	}
	return label
}

func progressDuration(seconds float64) string {
	value := max(0, int(seconds+.5))
	return fmt.Sprintf("%d:%02d", value/60, value%60)
}

// Only this goroutine emits while inference is pending. Stop joins it before
// attachments/results are emitted and drains any final worker events.
func startGenerationProgress(ctx context.Context, endpoint, callID, kind string, emit eventEmitter) (string, func()) {
	random := make([]byte, 16)
	if _, err := rand.Read(random); err != nil || emit == nil {
		return "", func() {}
	}
	id := hex.EncodeToString(random)
	started := time.Now()
	progressCtx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	client := &http.Client{Timeout: 2 * time.Second}
	cursor := 0
	report := func(p generationProgress, log bool) {
		p.Elapsed = time.Since(started).Seconds()
		payload := map[string]any{"id": callID, "stream": "stdout", "progress": p}
		if log {
			text := fmt.Sprintf("[%s] %s", progressDuration(p.Elapsed), p.label())
			if p.ETA != nil && p.ETAScope != "" && p.Event != "idle" {
				scope := "전체"
				if p.ETAScope == "stage" {
					scope = "현재 단계"
				}
				text += fmt.Sprintf(" · %s 예상 %s 남음", scope, progressDuration(*p.ETA))
			} else if p.Event != "idle" && p.Event != "failed" {
				text += " · 예상 시간 계산 중"
			}
			payload["delta"] = text + "\n"
		}
		_ = emit("tool_output", payload)
	}
	current := generationProgress{Stage: "preparing", Kind: kind}
	report(current, true)
	poll := func(pollCtx context.Context) {
		request, err := http.NewRequestWithContext(pollCtx, "GET", fmt.Sprintf("%s/v1/runtime/progress/%s?after=%d", endpoint, id, cursor), nil)
		if err != nil {
			return
		}
		response, err := client.Do(request)
		if err != nil {
			return
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			return
		}
		var batch struct {
			Events []generationProgress `json:"events"`
			Next   int                  `json:"next"`
		}
		if json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&batch) != nil || batch.Next < cursor {
			return
		}
		for _, p := range batch.Events {
			current = p
			report(p, true)
		}
		cursor = batch.Next
	}
	go func() {
		defer close(done)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		lastEvent := time.Now()
		for {
			select {
			case <-progressCtx.Done():
				return
			case <-ticker.C:
				previous := cursor
				poll(progressCtx)
				if cursor != previous {
					lastEvent = time.Now()
				} else if time.Since(lastEvent) >= 5*time.Second {
					p := current
					// Count down an estimate between real engine updates, without treating
					// a stale estimate as completion or logging repeated heartbeat lines.
					if p.ETA != nil {
						remaining := max(0, *p.ETA-time.Since(lastEvent).Seconds())
						if remaining < 1 {
							p.ETA = nil
							p.ETAScope = ""
						} else {
							p.ETA = &remaining
						}
					}
					report(p, false)
				}
			}
		}
	}()
	var once sync.Once
	return id, func() {
		once.Do(func() {
			cancel()
			<-done
			if ctx.Err() == nil {
				finalCtx, finish := context.WithTimeout(ctx, time.Second)
				defer finish()
				poll(finalCtx)
			}
		})
	}
}
