package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

func TestHTTPRetryFailedVideoPreservesFramesAndClearsStaleError(t *testing.T) {
	s, _ := testImageServer(t)
	video, err := s.media.SaveReader(bytes.NewReader(append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)), "clip.mp4", "video/mp4", media.MaxAttachmentBytes)
	if err != nil {
		t.Fatal(err)
	}
	parent, err := s.db.AddPendingMessage("session", "Analyze this video", []db.Attachment{video})
	if err != nil {
		t.Fatal(err)
	}
	if err = s.db.FailPendingTurn(parent.ID, db.MessageFailed, "At most 0 video(s) may be provided", "", "old interrupted reasoning", nil); err != nil {
		t.Fatal(err)
	}
	history, _ := s.db.Messages("session")
	target := history[len(history)-1]
	framesCalled, modelCalled := false, false
	frames := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		framesCalled = true
		jpeg.Encode(w, image.NewRGBA(image.Rect(0, 0, 16, 8)), nil)
	}))
	defer frames.Close()
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelCalled = true
		payload, _ := io.ReadAll(r.Body)
		if strings.Contains(string(payload), "At most 0 video") || strings.Contains(string(payload), "video_url") || !strings.Contains(string(payload), "data:image/jpeg;base64,") || !strings.Contains(string(payload), video.ID) {
			t.Errorf("incorrect retry payload: %s", payload)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Visible frame analysis\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer backend.Close()
	s.cfg = config.Config{Model: config.ModelConfig{Endpoint: backend.URL, DefaultModel: "ornith", ModelType: "qwen3.5", VideoInputs: map[string]string{backend.URL + "\nornith": "frames"}}, ASR: config.ASRConfig{FFmpegEndpoint: frames.URL}, Context: config.ContextConfig{WindowTokens: 32768}, Tools: config.ToolsConfig{MaxRounds: 1}}
	s.llm = llm.New(backend.URL, "ornith", "")
	w := httptest.NewRecorder()
	s.messageAction(w, httptest.NewRequest("POST", fmt.Sprintf("/api/messages/%d/retry", target.ID), strings.NewReader(`{"model":"ornith","tools_enabled":false}`)))
	if !framesCalled || !modelCalled || strings.Contains(w.Body.String(), "event: error") || !strings.Contains(w.Body.String(), "event: done") {
		t.Fatalf("retry failed: frames=%v model=%v %s", framesCalled, modelCalled, w.Body.String())
	}
	history, err = s.db.Messages("session")
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range history {
		if m.ID == parent.ID && (m.Status != db.MessageCompleted || len(m.Attachments) != 1) {
			b, _ := json.Marshal(m)
			t.Fatalf("retry did not restore stored request: %s", b)
		}
	}
}
