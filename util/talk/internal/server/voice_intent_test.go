package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/llm"
	"strings"
	"testing"
)

func TestVoiceHelpRoutesWithoutModelOrWork(t *testing.T) {
	s := &Server{}
	for _, request := range []string{"음성 생성 어떻게 하냐", "음성 도움말", "목소리 등록 방법 알려줘", "how to generate voice"} {
		call, err := s.routeVoiceIntent(context.Background(), "session", []llm.Message{{Role: "user", Content: request}}, nil, "")
		if err != nil || call == nil || call.Function.Name != "voice_tools" || call.Function.Arguments != `{"action":"help"}` {
			t.Fatalf("%q: %+v %v", request, call, err)
		}
	}
	for _, request := range []string{"음성으로 안녕하세요를 읽어줘", "목소리를 등록해", "어떻게든 영상을 만들어줘"} {
		if voiceHelpRequest(request) {
			t.Fatal("action mistaken for help", request)
		}
	}
}

func TestVoiceReferenceIntentKeepsNormalTranscription(t *testing.T) {
	for _, request := range []string{"이 음성을 내 목소리로 등록해", "이 파일 목소리로 음성 만들어줘"} {
		if !voiceReferenceIntent(request) {
			t.Fatal(request)
		}
	}
	for _, request := range []string{"이 음성을 전사해", "영상 내용을 요약해", "음성에서 무슨 말을 했어?"} {
		if voiceReferenceIntent(request) {
			t.Fatal("blocked normal media", request)
		}
	}
	help := voiceHelpText(map[string]any{"input_formats": []any{"flac", "ogg"}})
	if !strings.Contains(help, "FLAC, OGG") || !strings.Contains(help, "생성 작업을 시작하지 않았습니다") {
		t.Fatal(help)
	}
}

func TestVoiceResetEvidenceAndRetry(t *testing.T) {
	for _, tc := range []struct {
		name, url          string
		paraphrase, reject bool
		calls              int
	}{
		{name: "whitespace", url: "https://www.youtube.com/watch?v=mme_JyKGd5I", calls: 1},
		{name: "repair_quote", url: "https://www.youtube.com/watch?v=mme_JyKGd5I", paraphrase: true, calls: 2},
		{name: "reject_invented_authorization", url: "https://www.youtube.com/watch?v=mme_JyKGd5I", paraphrase: true, reject: true, calls: 2},
		{name: "shorts_url_is_not_video_creation", url: "https://www.youtube.com/shorts/new-source", calls: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			id := "voice_" + strings.Repeat("a", 32)
			s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/v1/voice-tools":
					json.NewEncoder(w).Encode(map[string]any{"voices": []map[string]string{{"id": id, "name": "장태완 장포스", "scope": "persistent"}}})
				case "/v1/voice-jobs":
					w.Write([]byte(`[]`))
				default:
					w.Write([]byte(`{"status":"ok"}`))
				}
			})
			calls := 0
			model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				var input struct {
					Messages []llm.Message `json:"messages"`
				}
				json.NewDecoder(r.Body).Decode(&input)
				if calls == 2 && len(input.Messages) != 4 {
					t.Error("repair did not retain initial decision and request")
				}
				evidence := tc.url + " 장포스 이걸로 다시 세팅해봐라."
				if tc.paraphrase && (calls == 1 || tc.reject) {
					evidence = "장포스 음성을 재학습하고 기존 음성을 삭제해"
				}
				streamTestContent(w, shortsIntentJSON(map[string]any{"action": "prepare", "evidence": evidence, "arguments": map[string]any{"target_voice_id": id, "source_url": tc.url, "name": "장태완 장포스", "scope": "persistent"}}))
			}))
			defer model.Close()
			call, err := s.routeVoiceIntent(context.Background(), "session", []llm.Message{{Role: "user", Content: tc.url + "   장포스 이걸로 다시 세팅해봐라."}}, llm.New(model.URL, "test-model", "", "qwen3.8"), "test-model")
			if calls != tc.calls {
				t.Fatalf("model calls %d", calls)
			}
			if tc.reject {
				if err == nil || call != nil {
					t.Fatal("accepted invented authorization", call, err)
				}
				return
			}
			if err != nil || call == nil {
				t.Fatal(call, err)
			}
			var args voiceRequest
			if err = json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
				t.Fatal(err)
			}
			if args.Action != "prepare" || args.TargetVoiceID != id || args.SourceURL != tc.url || args.SourceJobID != "" || args.VoiceID != "" {
				t.Fatalf("wrong replacement %+v", args)
			}
		})
	}
	for _, evidence := range []string{"", "장포스 음성을 삭제해", "장포스 이걸로 다시 세팅해봐라."} {
		if voiceIntentEvidenceMatches("장포스 이걸로 다시 세팅하지 마라.", evidence) {
			t.Fatal("accepted changed wording", evidence)
		}
	}
}

func TestVoiceReferenceReplacementRejectsAmbiguousSourcesBeforeWork(t *testing.T) {
	s := &Server{}
	for _, input := range []voiceRequest{
		{Action: "prepare", VoiceID: "voice_" + strings.Repeat("a", 32), SourceURL: "https://example.com/new.wav"},
		{Action: "prepare", SourceJobID: strings.Repeat("a", 32), SourceURL: "https://example.com/new.wav"},
	} {
		if _, err := s.createVoiceJob(context.Background(), input); err == nil {
			t.Fatal("accepted ambiguous replacement", input)
		}
	}
}

func TestVoiceReferenceResetPreservesExistingNameWhenOmitted(t *testing.T) {
	for _, given := range []string{"", "새 이름"} {
		t.Run(given, func(t *testing.T) {
			id := "voice_" + strings.Repeat("a", 32)
			expected := given
			if expected == "" {
				expected = "장태완 장포스"
			}
			creates := 0
			s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/v1/voice-tools":
					w.Write([]byte(`{"name":"장태완 장포스"}`))
				case "/v1/voice-jobs":
					creates++
					var payload map[string]any
					json.NewDecoder(r.Body).Decode(&payload)
					if payload["name"] != expected || payload["target_voice_id"] != id {
						t.Error("lost name or target", payload)
					}
					json.NewEncoder(w).Encode(voiceJob{ID: strings.Repeat("b", 32), SessionID: "session", Status: "queued"})
				default:
					w.Write([]byte(`{"status":"ok"}`))
				}
			})
			_, err := s.createVoiceJob(context.Background(), voiceRequest{SessionID: "session", Action: "prepare", TargetVoiceID: id, Name: given, SourceJobID: strings.Repeat("c", 32)})
			if err != nil || creates != 1 {
				t.Fatal(creates, err)
			}
		})
	}
}
