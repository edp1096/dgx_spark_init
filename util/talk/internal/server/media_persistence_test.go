package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

var importTestVideo = append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)

func TestImportedVideoPersistsBeforeProcessingAndAfterCompletion(t *testing.T) {
	for _, framesFail := range []bool{false, true} {
		t.Run(fmt.Sprintf("frame_failure_%v", framesFail), func(t *testing.T) {
			s, _ := testImageServer(t)
			png, _ := base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
			var persistedBeforeFrames atomic.Bool
			mediaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/video/frames" {
					history, err := s.db.Messages("session")
					if err == nil && len(history) > 0 && len(history[len(history)-1].Attachments) == 1 && history[len(history)-1].Attachments[0].MIME == "video/mp4" {
						persistedBeforeFrames.Store(true)
					}
					if framesFail {
						http.Error(w, "frame decoder unavailable", 500)
						return
					}
					w.Header().Set("Content-Type", "image/jpeg")
					_ = jpeg.Encode(w, image.NewRGBA(image.Rect(0, 0, 16, 8)), nil)
					return
				}
				var source struct {
					URL string `json:"url"`
				}
				_ = json.NewDecoder(r.Body).Decode(&source)
				if strings.HasSuffix(source.URL, "/poster") {
					w.Header().Set("Content-Type", "image/png")
					w.Header().Set("Content-Disposition", `attachment; filename="cover.png"`)
					_, _ = w.Write(png)
				} else {
					w.Header().Set("Content-Type", "video/mp4")
					w.Header().Set("Content-Disposition", `attachment; filename="clip.mp4"`)
					_, _ = w.Write(importTestVideo)
				}
			}))
			defer mediaServer.Close()
			var calls atomic.Int32
			var analysisEvidence atomic.Bool
			model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var payload struct {
					Stream   bool            `json:"stream"`
					Messages json.RawMessage `json:"messages"`
				}
				_ = json.NewDecoder(r.Body).Decode(&payload)
				if !payload.Stream {
					fmt.Fprint(w, `{"choices":[{"message":{"content":"test"}}]}`)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				round := calls.Add(1)
				if round == 2 {
					expected := "video_frames"
					if framesFail {
						expected = "analysis_error"
					}
					analysisEvidence.Store(bytes.Contains(payload.Messages, []byte(expected)))
				}
				if round <= 2 {
					suffix := "video"
					if round == 2 {
						suffix = "poster"
					}
					args, _ := json.Marshal(map[string]string{"url": "https://example.org/" + suffix})
					delta, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": fmt.Sprint(round), "type": "function", "function": map[string]string{"name": "media_import", "arguments": string(args)}}}}}}})
					fmt.Fprintf(w, "data: %s\n\n", delta)
				} else {
					fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"완료\"}}]}\n\n")
				}
				fmt.Fprint(w, "data: [DONE]\n\n")
			}))
			defer model.Close()
			s.cfg = config.Config{Model: config.ModelConfig{Endpoint: model.URL, DefaultModel: "test-model", ModelType: "qwen3.5", VideoInputs: map[string]string{model.URL + "\ntest-model": "frames"}}, ASR: config.ASRConfig{FFmpegEndpoint: mediaServer.URL, Timeout: "5s"}, Context: config.ContextConfig{WindowTokens: 32768}, Tools: config.ToolsConfig{MediaImportEnabled: true, MaxRounds: 4, Timeout: "5s"}}
			s.llm = llm.New(model.URL, "test-model", "")
			// Avoid an unrelated asynchronous title request in this regression.
			_, _ = s.db.AddMessage("session", "user", "earlier question", "", nil, nil)
			_, _ = s.db.AddMessage("session", "assistant", "earlier reply", "", nil, nil)
			w := httptest.NewRecorder()
			s.chat(w, httptest.NewRequest("POST", "/api/chat", strings.NewReader(`{"session_id":"session","content":"Analyze https://example.org/video and https://example.org/poster","tools_enabled":true}`)))
			if !strings.Contains(w.Body.String(), "event: done") || strings.Contains(w.Body.String(), "event: error") {
				t.Fatalf("completion failed: %s", w.Body.String())
			}
			if strings.Count(w.Body.String(), "event: media_attached") != 2 {
				t.Fatalf("attachment events missing or duplicated: %s", w.Body.String())
			}
			history, err := s.db.Messages("session")
			if err != nil {
				t.Fatal(err)
			}
			parent := history[len(history)-2]
			if len(parent.Attachments) != 2 || parent.Attachments[0].MIME != "video/mp4" || parent.Attachments[1].MIME != "image/png" {
				t.Fatalf("video replaced by cover after completion: %+v", parent)
			}
			if !analysisEvidence.Load() {
				t.Fatal("model did not receive frame evidence or explicit preparation failure")
			}
			if !persistedBeforeFrames.Load() {
				t.Fatal("original video was not persisted before frame extraction")
			}
			for _, a := range parent.Attachments {
				f, err := s.media.Open(a)
				if err != nil {
					t.Fatal(err)
				}
				f.Close()
			}
		})
	}
}

func TestRetryPosterCannotReplaceStoredVideo(t *testing.T) {
	s, poster := testImageServer(t)
	video, err := s.media.SaveReader(bytes.NewReader(importTestVideo), "clip.mp4", "video/mp4", media.MaxAttachmentBytes)
	if err != nil {
		t.Fatal(err)
	}
	video.SourceURL = "https://example.org/watch"
	parent, err := s.db.AddPendingMessage("session", "video", []db.Attachment{video})
	if err != nil {
		t.Fatal(err)
	}
	poster.SourceURL = video.SourceURL
	sink := s.persistMediaAttachments(parent.ID, 0, parent.Attachments, map[string]string{video.SourceURL: video.ID})
	if err := sink(poster); err != nil {
		t.Fatal(err)
	}
	history, _ := s.db.Messages("session")
	stored := history[len(history)-1]
	if len(stored.Attachments) != 2 || stored.Attachments[0].ID != video.ID {
		t.Fatalf("poster discarded video: %+v", stored.Attachments)
	}
}
