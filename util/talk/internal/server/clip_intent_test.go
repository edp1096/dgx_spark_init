package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func streamTestContent(w http.ResponseWriter, content string) {
	chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": content}}}})
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", chunk)
}

func TestClipIntentCannotUseInventedSourceOrAuthorization(t *testing.T) {
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/clips" {
			w.Write([]byte(`[]`))
			return
		}
		w.Write([]byte(`{"status":"ok"}`))
	})
	history, _ := s.db.Messages("session")
	source := history[0].Attachments[0]
	var response atomic.Value
	model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { streamTestContent(w, response.Load().(string)) }))
	defer model.Close()
	client := llm.New(model.URL, "test-model", "", "qwen3.8")
	request := "이 사진을 만화로 바꾸고 영상으로 만들어줘"
	base := map[string]any{"action": "produce", "source_image_id": source.ID, "job_id": "", "brief": request, "resolution": "480p", "evidence": "영상으로 만들어줘"}
	encode := func(a map[string]any) { raw, _ := json.Marshal(a); response.Store(string(raw)) }
	encode(base)
	call, err := s.routeClipIntent(context.Background(), "session", []llm.Message{{Role: "user", Content: request}}, client, "test-model")
	if err != nil || call == nil || call.Function.Name != "clip_produce" {
		t.Fatal(call, err)
	}
	base["source_image_id"] = "unknown"
	encode(base)
	if _, err = s.routeClipIntent(context.Background(), "session", []llm.Message{{Role: "user", Content: request}}, client, "test-model"); err == nil {
		t.Fatal("invented source accepted")
	}
	base["source_image_id"] = source.ID
	base["evidence"] = "the user never said this"
	encode(base)
	if _, err = s.routeClipIntent(context.Background(), "session", []llm.Message{{Role: "user", Content: request}}, client, "test-model"); err == nil {
		t.Fatal("invented authorization accepted")
	}
	base["action"] = "none"
	base["evidence"] = ""
	encode(base)
	if call, err = s.routeClipIntent(context.Background(), "session", []llm.Message{{Role: "user", Content: "사진만 바꿔줘. 영상은 만들지 마."}}, client, "test-model"); err != nil || call != nil {
		t.Fatal("image-only request forced video", call, err)
	}
}

func TestNewClipVersionExecutesInsteadOfReplayingAnOldFailure(t *testing.T) {
	id, oldID := strings.Repeat("d", 32), strings.Repeat("e", 32)
	var creates, routerCalls, chatCalls atomic.Int32
	var sourceID string
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/clips" && r.Method == "GET" {
			json.NewEncoder(w).Encode([]any{map[string]any{"id": oldID, "status": "failed", "project": map[string]string{"source_image_id": sourceID}}})
			return
		}
		if r.URL.Path == "/v1/clips" && r.Method == "POST" {
			creates.Add(1)
			json.NewEncoder(w).Encode(clipJob{ID: id, SessionID: "session", Status: "queued"})
			return
		}
		if r.URL.Path == "/v1/clips/"+id {
			json.NewEncoder(w).Encode(clipJob{ID: id, SessionID: "session", Status: "completed", OutputReady: true, Report: json.RawMessage(`{"passed":true,"plan":{"title":"새 영상"},"metadata":{"width":640,"height":640,"duration":5.167}}`)})
			return
		}
		if strings.HasSuffix(r.URL.Path, "/files/final.mp4") {
			w.Header().Set("Content-Type", "video/mp4")
			w.Write(importTestVideo)
			return
		}
		w.Write([]byte(`{"status":"ok","busy":false}`))
	})
	history, _ := s.db.Messages("session")
	sourceID = history[0].Attachments[0].ID
	request := "같은 원본 사진으로 새 버전을 다시 만들어줘. 차가 조금 굴러가는 영상으로."
	model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content any `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if text, ok := body.Messages[0].Content.(string); ok && strings.Contains(text, "photo-to-video intent router") {
			routerCalls.Add(1)
			data, _ := json.Marshal(map[string]string{"action": "produce", "source_image_id": sourceID, "job_id": "", "brief": request, "resolution": "480p", "evidence": "새 버전을 다시 만들어줘"})
			streamTestContent(w, string(data))
			return
		}
		chatCalls.Add(1)
		streamTestContent(w, "새 버전 시도했는데 실패했습니다. 실제 도구는 호출하지 않았습니다.")
	}))
	defer model.Close()
	client := llm.New(model.URL, "test-model", "", "qwen3.8")
	messages := []llm.Message{{Role: "assistant", Content: "이전 영상은 검수에 실패했습니다."}, {Role: "user", Content: request}}
	result, err := runCompletionLoopForSessionWithMedia(s, "session", context.Background(), client, messages, "test-model", "none", "Be helpful.", config.ToolsConfig{MaxRounds: 4}, false, func(string, any) error { return nil }, func(db.Attachment) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if creates.Load() != 1 || routerCalls.Load() != 1 || chatCalls.Load() != 0 {
		t.Fatal("new request did not execute exactly once", creates.Load(), routerCalls.Load(), chatCalls.Load())
	}
	if !strings.Contains(result.Content, "검수한 영상을 첨부했습니다") || strings.Contains(result.Content, "실패") {
		t.Fatal("unverified narrative escaped", result.Content)
	}
	if len(result.ToolTrace) != 1 || result.ToolTrace[0].Name != "clip_produce" {
		t.Fatal("execution trace missing", result.ToolTrace)
	}
}
