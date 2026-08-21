package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/tts"
)

func TestSynthesizeSpeechProxiesAudio(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["seed"] != float64(9876) {
			t.Fatalf("unexpected seed: %+v", body)
		}
		w.Header().Set("Content-Type", "audio/pcm")
		fmt.Fprint(w, "pcm-audio")
	}))
	defer upstream.Close()
	cfg := config.TTSConfig{Enabled: true, Endpoint: upstream.URL, Model: "tts", Language: "Korean", Voice: "Sohee", Seed: -1, Timeout: "5s"}
	s := &Server{tts: tts.New(cfg)}

	request := httptest.NewRequest(http.MethodPost, "/api/tts/speech", strings.NewReader(`{"text":"읽어 주세요","seed":9876}`))
	response := httptest.NewRecorder()
	s.synthesizeSpeech(response, request)
	if response.Code != http.StatusOK || response.Body.String() != "pcm-audio" || response.Header().Get("Content-Type") != "audio/pcm" || response.Header().Get("X-Audio-Sample-Rate") != "24000" {
		t.Fatalf("unexpected response: code=%d type=%q body=%q", response.Code, response.Header().Get("Content-Type"), response.Body.String())
	}
}
