package server

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/media"
)

func TestLLMMessagesTranscribesAudioAndKeepsVideoVisuals(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	audio, err := store.SaveAttachment(testFileHeader(t, "voice.mp3", "audio/mpeg", []byte("ID3sample audio")))
	if err != nil {
		t.Fatal(err)
	}
	video, err := store.SaveAttachment(testFileHeader(t, "clip.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)))
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{ASR: config.ASRConfig{Enabled: true, Endpoint: "http://asr", FFmpegEndpoint: "http://ffmpeg", Model: "qwen3-asr", Language: "auto"}}
	fingerprint := transcriptFingerprint(cfg.ASR)
	for _, item := range []db.Attachment{audio, video} {
		if err := store.SaveTranscript(item.ID, media.TranscriptCache{Fingerprint: fingerprint, Text: item.Name + " 전사", Language: "Korean"}); err != nil {
			t.Fatal(err)
		}
	}
	s := &Server{media: store}
	messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: "분석해줘", Attachments: []db.Attachment{audio, video}}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	payload, _ := json.Marshal(messages[0].Content)
	text := string(payload)
	for _, expected := range []string{"video_url", "data:video/mp4;base64,", "voice.mp3 전사", "clip.mp4 전사"} {
		if !strings.Contains(text, expected) {
			t.Fatalf("multimodal payload does not contain %q: %s", expected, text)
		}
	}
	if strings.Contains(text, "audio_url") || strings.Contains(text, "data:audio/mpeg;base64,") {
		t.Fatalf("raw audio must not be sent to the visual language model: %s", text)
	}
}

func TestMediaFileHandlerSupportsRangeRequests(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	item, err := store.SaveAttachment(testFileHeader(t, "clip.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)))
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store}
	req := httptest.NewRequest(http.MethodGet, item.URL, nil)
	req.Header.Set("Range", "bytes=0-3")
	recorder := httptest.NewRecorder()
	s.file(recorder, req)
	if recorder.Code != http.StatusPartialContent || recorder.Header().Get("Content-Type") != "video/mp4" || recorder.Body.Len() != 4 {
		t.Fatalf("unexpected range response: status=%d type=%q bytes=%d", recorder.Code, recorder.Header().Get("Content-Type"), recorder.Body.Len())
	}
}

func TestUploadSourceStoresMediaResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/source/download" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "video/mp4")
		w.Header().Set("Content-Disposition", `attachment; filename="source.mp4"`)
		w.Header().Set("X-Media-Title", "Remote+clip")
		_, _ = w.Write(append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...))
	}))
	defer upstream.Close()
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store, cfg: config.Config{ASR: config.ASRConfig{FFmpegEndpoint: upstream.URL, Timeout: "5s"}}}
	req := httptest.NewRequest(http.MethodPost, "/api/media/source", strings.NewReader(`{"url":"https://example.com/video"}`))
	recorder := httptest.NewRecorder()
	s.uploadSource(recorder, req)
	if recorder.Code != http.StatusCreated {
		t.Fatalf("unexpected status %d: %s", recorder.Code, recorder.Body.String())
	}
	var item db.Attachment
	if err := json.Unmarshal(recorder.Body.Bytes(), &item); err != nil {
		t.Fatal(err)
	}
	if item.Name != "Remote clip.mp4" || item.MIME != "video/mp4" {
		t.Fatalf("unexpected attachment: %+v", item)
	}
}

func testFileHeader(t *testing.T, name, mimeType string, data []byte) *multipart.FileHeader {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", `form-data; name="file"; filename="`+name+`"`)
	header.Set("Content-Type", mimeType)
	part, err := writer.CreatePart(header)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/files", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	if err := req.ParseMultipartForm(1 << 20); err != nil {
		t.Fatal(err)
	}
	file, fileHeader, err := req.FormFile("file")
	if err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	return fileHeader
}
