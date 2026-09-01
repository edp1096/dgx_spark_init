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
		if _, ok := body["seed"]; ok {
			t.Fatalf("Magpie payload must ignore seed: %+v", body)
		}
		w.Header().Set("Content-Type", "audio/pcm")
		fmt.Fprint(w, "pcm-audio")
	}))
	defer upstream.Close()
	cfg := config.TTSConfig{Enabled: true, Endpoint: upstream.URL, Model: "tts", Language: "ko-KR", HanjaReading: "korean", Voice: "Sofia", SampleRate: 22050, Timeout: "5s"}
	s := &Server{tts: tts.New(cfg)}

	request := httptest.NewRequest(http.MethodPost, "/api/tts/speech", strings.NewReader(`{"text":"읽어 주세요"}`))
	response := httptest.NewRecorder()
	s.synthesizeSpeech(response, request)
	if response.Code != http.StatusOK || response.Body.String() != "pcm-audio" || response.Header().Get("Content-Type") != "audio/pcm" || response.Header().Get("X-Audio-Sample-Rate") != "22050" {
		t.Fatalf("unexpected response: code=%d type=%q body=%q", response.Code, response.Header().Get("Content-Type"), response.Body.String())
	}
}

func TestSynthesizeSpeechStreamsMixedKoreanAndEnglishParts(t *testing.T) {
	languages := make([]string, 0, 3)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		languages = append(languages, fmt.Sprint(body["language"]))
		fmt.Fprintf(w, "[%s:%s]", body["language"], body["input"])
	}))
	defer upstream.Close()
	s := &Server{tts: tts.New(config.TTSConfig{
		Enabled: true, Endpoint: upstream.URL, Model: "tts",
		Language: "auto", HanjaReading: "korean", Voice: "Sofia", SampleRate: 22050, Timeout: "5s",
	})}

	request := httptest.NewRequest(http.MethodPost, "/api/tts/speech", strings.NewReader(`{"text":"한국어 API 테스트"}`))
	response := httptest.NewRecorder()
	s.synthesizeSpeech(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("unexpected response status: %d", response.Code)
	}
	if got, want := strings.Join(languages, ","), "ko-KR,en-US,ko-KR"; got != want {
		t.Fatalf("languages = %q, want %q", got, want)
	}
	if got, want := response.Body.String(), "[ko-KR:한국어][en-US:API][ko-KR:테스트]"; got != want {
		t.Fatalf("audio concatenation = %q, want %q", got, want)
	}
}
