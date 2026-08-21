package tts

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"sparktalk/internal/config"
)

func TestSpeechUsesConfiguredCustomVoice(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/speech" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request["input"] != "안녕하세요" || request["voice"] != "sohee" || request["seed"] != float64(7) {
			t.Fatalf("unexpected request: %+v", request)
		}
		if request["response_format"] != "pcm" || request["stream"] != true || request["stream_format"] != "audio" {
			t.Fatalf("streaming not requested: %+v", request)
		}
		w.Header().Set("Content-Type", "audio/pcm")
		_, _ = w.Write([]byte("pcm-test"))
	}))
	defer server.Close()

	client := New(config.TTSConfig{Enabled: true, Endpoint: server.URL, Model: "qwen-tts", Language: "Korean", Voice: "Sohee", Seed: 7, Timeout: "5s"})
	audio, contentType, err := client.Speech(context.Background(), "안녕하세요")
	if err != nil {
		t.Fatal(err)
	}
	if string(audio) != "pcm-test" || contentType != "audio/pcm" {
		t.Fatalf("unexpected response: %q %q", audio, contentType)
	}
}
