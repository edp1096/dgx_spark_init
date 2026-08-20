package asr

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
)

func TestTranscribeStreamsFFmpegWAVIntoASR(t *testing.T) {
	ffmpeg := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/extract" || r.URL.Query().Get("sample_rate") != "16000" {
			t.Fatalf("unexpected FFmpeg request: %s", r.URL.String())
		}
		data, _ := io.ReadAll(r.Body)
		if string(data) != "video-bytes" || r.Header.Get("Content-Type") != "video/mp4" {
			t.Fatalf("unexpected FFmpeg input: type=%s data=%q", r.Header.Get("Content-Type"), data)
		}
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("wav-bytes"))
	}))
	defer ffmpeg.Close()

	asrServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		file, header, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		data, _ := io.ReadAll(file)
		if string(data) != "wav-bytes" || !strings.HasSuffix(header.Filename, ".wav") {
			t.Fatalf("unexpected ASR file: name=%s data=%q", header.Filename, data)
		}
		if r.FormValue("model") != "qwen3-asr" || r.FormValue("language") != "auto" {
			t.Fatalf("unexpected ASR fields: %+v", r.MultipartForm.Value)
		}
		_ = json.NewEncoder(w).Encode(Result{Text: "안녕하세요", Language: "Korean"})
	}))
	defer asrServer.Close()

	client := New(config.ASRConfig{
		Enabled: true, FFmpegEndpoint: ffmpeg.URL, Endpoint: asrServer.URL,
		Model: "qwen3-asr", Language: "auto", Timeout: "5s",
	})
	result, err := client.Transcribe(context.Background(), strings.NewReader("video-bytes"), "clip.mp4", "video/mp4")
	if err != nil {
		t.Fatal(err)
	}
	if result.Text != "안녕하세요" || result.Language != "Korean" {
		t.Fatalf("unexpected transcript: %+v", result)
	}
}

func TestTranscribeRecognizesMissingAudio(t *testing.T) {
	ffmpeg := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		_, _ = w.Write([]byte(`{"error":"Stream map 0:a:0 matches no streams"}`))
	}))
	defer ffmpeg.Close()
	client := New(config.ASRConfig{Enabled: true, FFmpegEndpoint: ffmpeg.URL, Endpoint: "http://unused", Timeout: "5s"})
	_, err := client.Transcribe(context.Background(), strings.NewReader("video"), "silent.mp4", "video/mp4")
	if err != ErrNoAudio {
		t.Fatalf("got %v, want ErrNoAudio", err)
	}
}
