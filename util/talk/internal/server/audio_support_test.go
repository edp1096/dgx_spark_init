package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestExcludedAudioRejectsRequestsBeforeAnyWorkerCall(t *testing.T) {
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("excluded worker called: %s", r.URL.Path)
		http.Error(w, "unexpected", 500)
	})
	for _, bundle := range []string{"flash-next-fp4-local", "flash-next-tp2", "glm53-worker-extra", "ds41"} {
		s.cfg.Runtime.ActiveBundle = bundle
		for _, route := range []string{"/api/shorts/music", "/api/shorts/speech"} {
			w := httptest.NewRecorder()
			s.shortsAPI(w, httptest.NewRequest("POST", route, strings.NewReader(`{}`)))
			if w.Code != http.StatusConflict || !strings.Contains(w.Body.String(), "제외") {
				t.Fatal(bundle, route, w.Code, w.Body.String())
			}
		}

		reg := completionToolRegistry{sessionID: "session", handlers: map[string]registeredToolHandler{}}
		s.registerShortsTool(&reg)
		for _, tool := range []string{"music_generate", "voice_tools", "shorts_voices", "shorts_generate"} {
			if reg.handlers[tool] != nil {
				t.Fatal(bundle, tool, "still offered")
			}
		}
		if !strings.Contains(strings.Join(reg.prompts, "\n"), "excluded") {
			t.Fatal("missing explanation", bundle)
		}
		if _, err := s.generateMusic(context.Background(), musicRequest{Model: "ace-xl", Prompt: "calm piano music", Duration: 75}); err == nil || !strings.Contains(err.Error(), "제외") {
			t.Fatal(bundle, err)
		}
		if _, err := s.shortsSpeech(context.Background(), shortsRequest{Texts: []string{"안녕하세요"}}); err == nil || !strings.Contains(err.Error(), "제외") {
			t.Fatal(bundle, err)
		}
		if _, err := s.createVoiceJob(context.Background(), voiceRequest{SessionID: "session", Action: "generate", Text: "안녕하세요"}); err == nil || !strings.Contains(err.Error(), "제외") {
			t.Fatal(bundle, err)
		}
	}
	s.cfg.Runtime.ActiveBundle = "qwen38fn_exl3"
	reg := completionToolRegistry{sessionID: "session", handlers: map[string]registeredToolHandler{}}
	s.registerShortsTool(&reg)
	if reg.handlers["music_generate"] == nil || reg.handlers["voice_tools"] == nil {
		t.Fatal("other set definitions removed")
	}
}
