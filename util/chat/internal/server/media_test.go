package server

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"strings"
	"testing"

	"sparktalk/internal/db"
	"sparktalk/internal/media"
)

func TestLLMMessagesUsesAudioAndVideoContentParts(t *testing.T) {
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
	s := &Server{media: store}
	messages, err := s.llmMessages([]db.Message{{Role: "user", Content: "분석해줘", Attachments: []db.Attachment{audio, video}}})
	if err != nil {
		t.Fatal(err)
	}
	payload, _ := json.Marshal(messages[0].Content)
	text := string(payload)
	for _, expected := range []string{"audio_url", "data:audio/mpeg;base64,", "video_url", "data:video/mp4;base64,"} {
		if !strings.Contains(text, expected) {
			t.Fatalf("multimodal payload does not contain %q: %s", expected, text)
		}
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
