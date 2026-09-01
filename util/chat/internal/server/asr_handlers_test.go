package server

import (
	"bytes"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"strings"
	"testing"

	"sparktalk/internal/asr"
	"sparktalk/internal/config"
)

func TestTranscribeVoiceStreamsRecordingToASR(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/audio/extract":
			data, _ := io.ReadAll(r.Body)
			if string(data) != "webm-audio" || r.Header.Get("Content-Type") != "audio/webm" {
				t.Fatalf("unexpected recording: type=%q data=%q", r.Header.Get("Content-Type"), data)
			}
			w.Header().Set("Content-Type", "audio/wav")
			_, _ = w.Write([]byte("wav-audio"))
		case "/v1/audio/transcriptions":
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Fatal(err)
			}
			file, _, err := r.FormFile("file")
			if err != nil {
				t.Fatal(err)
			}
			data, _ := io.ReadAll(file)
			_ = file.Close()
			if string(data) != "wav-audio" {
				t.Fatalf("unexpected ASR audio: %q", data)
			}
			if r.FormValue("language") != "ko-KR" || r.FormValue("model") != "nemotron" {
				t.Fatalf("unexpected voice ASR fields: %+v", r.MultipartForm.Value)
			}
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"text":"안녕하세요","language":"Korean"}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	cfg := config.Config{ASR: config.ASRConfig{Enabled: true, FFmpegEndpoint: upstream.URL, Endpoint: upstream.URL, Model: "nemotron", VoiceLanguage: "ko-KR", Timeout: "5s"}}
	s := &Server{cfg: cfg, asr: asr.New(cfg.ASR)}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	header := make(textproto.MIMEHeader)
	header["Content-Disposition"] = []string{`form-data; name="audio"; filename="voice.webm"`}
	header["Content-Type"] = []string{"audio/webm"}
	part, err := writer.CreatePart(header)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write([]byte("webm-audio"))
	_ = writer.Close()

	request := httptest.NewRequest(http.MethodPost, "/api/asr/transcribe", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()
	s.transcribeVoice(response, request)
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), "안녕하세요") {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}
