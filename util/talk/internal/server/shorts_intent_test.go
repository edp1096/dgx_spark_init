package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestNarratedShortExecutesDespitePriorFalseCompletion(t *testing.T) {
	id := strings.Repeat("a", 32)
	var creates, chatCalls atomic.Int32
	request := "선택한 기본 음성으로 15초짜리 쇼츠를 만들어줘. 음악 없이 두 장면으로."
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/v1/voices":
			w.Write([]byte(`{"voices":[{"id":"ryan","name":"Ryan"}],"selected_voice":"ryan"}`))
		case r.URL.Path == "/v1/projects" && r.Method == "GET":
			w.Write([]byte(`[]`))
		case r.URL.Path == "/v1/projects" && r.Method == "POST":
			creates.Add(1)
			var a shortsRequest
			json.NewDecoder(r.Body).Decode(&a)
			if a.Brief != request || a.DurationSeconds != 15 || a.VoicePreset != "ryan" {
				t.Error("lost actual request", a)
			}
			json.NewEncoder(w).Encode(shortsJob{ID: id, SessionID: "session", Status: "queued"})
		case r.URL.Path == "/v1/projects/"+id:
			json.NewEncoder(w).Encode(shortsJob{ID: id, SessionID: "session", Status: "completed", OutputReady: true, Applied: json.RawMessage(`{"voice_preset":"ryan","duration_seconds":15}`)})
		case strings.HasSuffix(r.URL.Path, "/files/final.mp4"):
			w.Header().Set("Content-Type", "video/mp4")
			w.Write(importTestVideo)
		default:
			w.Write([]byte(`{"status":"ok","busy":false}`))
		}
	})
	model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []llm.Message `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if len(body.Messages) > 0 && strings.Contains(body.Messages[0].Content.(string), "narrated-video execution intent router") {
			streamTestContent(w, shortsIntentJSON(map[string]any{"action": "create", "title": "검증", "brief": "model paraphrase must not replace user text", "duration_seconds": 15, "bgm": false, "voice_id": "ryan", "job_id": "", "evidence": "쇼츠를 만들어줘"}))
			return
		}
		chatCalls.Add(1)
		streamTestContent(w, "완성했어! 실제 작업 없이 첨부했다고 주장")
	}))
	defer model.Close()
	if !shortsIntentCandidate(request) {
		t.Fatal("not eligible", request)
	}
	checkReg := newCompletionToolRegistry(s, "session", config.ToolsConfig{MaxRounds: 4}, false, func(db.Attachment) error { return nil })
	if checkReg.handlers["shorts_generate"] == nil {
		t.Fatal("missing shorts registry", s.cfg.Runtime.Mode)
	}
	client := llm.New(model.URL, "test-model", "", "qwen3.8")
	result, err := runCompletionLoopForSessionWithMedia(s, "session", context.Background(), client, []llm.Message{{Role: "assistant", Content: "영상 완성! 첨부했어."}, {Role: "user", Content: request}}, "test-model", "none", "Be helpful.", config.ToolsConfig{MaxRounds: 4}, false, func(string, any) error { return nil }, func(db.Attachment) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if creates.Load() != 1 || chatCalls.Load() != 0 || !strings.Contains(result.Content, "검수한 영상을 첨부") || !strings.Contains(result.Content, "ryan") {
		t.Fatal("unexecuted completion", creates.Load(), chatCalls.Load(), result.Content, result.ToolTrace)
	}
}

func TestShortsIntentRoutingLive(t *testing.T) {
	endpoint := os.Getenv("TALK_SHORTS_INTENT_ENDPOINT")
	if endpoint == "" {
		t.Skip("opt-in read-only local model check")
	}
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/voices" {
			w.Write([]byte(`{"voices":[{"id":"ryan","name":"Ryan"},{"id":"aiden","name":"Aiden"}],"selected_voice":"ryan"}`))
			return
		}
		if r.URL.Path == "/v1/projects" {
			w.Write([]byte(`[{"id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","session_id":"session","status":"completed"}]`))
			return
		}
		w.Write([]byte(`{"status":"ok"}`))
	})
	client := llm.New(endpoint, "qwen38fn_exl3", "", "qwen3.8-tensorfold")
	for _, tc := range []struct{ request, tool string }{
		{"선택한 기본 음성으로 15초짜리 쇼츠를 만들어줘. 배경음악 없이 내레이션만 넣어. 남색 단색 배경에 흰 글씨 두 장면으로 해줘.", "shorts_generate"},
		{"기존 영상에 Aiden 음성을 적용해. 나머지는 그대로 둬.", "shorts_voices"},
		{"영상 생성에 쓸 Fish 음성 목록을 보여줘.", ""},
		{"쇼츠를 만들지 말고 지원하는 음성 목록만 알려줘.", ""},
		{"쇼츠를 만들 수 있는 기능이 있는지 설명만 해줘.", ""},
	} {
		t.Run(tc.request, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
			defer cancel()
			call, err := s.routeShortsIntent(ctx, "session", []llm.Message{{Role: "assistant", Content: "완성했다고 말했지만 실제로 새 작업은 실행하지 않았습니다."}, {Role: "user", Content: tc.request}}, client, "qwen38fn_exl3")
			if err != nil {
				t.Fatal(err)
			}
			if tc.tool == "" {
				if call != nil {
					t.Fatal("unexpected production", call)
				}
			} else if call == nil || call.Function.Name != tc.tool {
				t.Fatal(call)
			}
		})
	}
	t.Run("preview_then_apply_reference", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()
		call, err := s.routeShortsIntent(ctx, "session", []llm.Message{{Role: "user", Content: "Aiden 목소리를 들려줘"}, {Role: "assistant", Content: "aiden의 저장된 Fish 미리 듣기입니다."}, {Role: "user", Content: "그 음성을 기존 영상에 적용해"}}, client, "qwen38fn_exl3")
		if err != nil || call == nil {
			t.Fatal(call, err)
		}
		var args map[string]string
		if json.Unmarshal([]byte(call.Function.Arguments), &args) != nil || call.Function.Name != "shorts_voices" || args["voice_id"] != "aiden" {
			t.Fatal("preview reference confused with saved Ryan default", call)
		}
	})

}
