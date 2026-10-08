package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
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

func TestMediaImportToolAttachesAndInjectsVisualInput(t *testing.T) {
	pngData, err := base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
	if err != nil {
		t.Fatal(err)
	}
	mediaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		w.Header().Set("Content-Disposition", `attachment; filename="sample.png"`)
		_, _ = w.Write(pngData)
	}))
	defer mediaServer.Close()

	const sourceURL = "https://example.com/sample-image"
	var requests atomic.Int32
	var sawVisual atomic.Bool
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(payload["messages"])
		w.Header().Set("Content-Type", "text/event-stream")
		if requests.Add(1) == 1 {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"media-1\",\"type\":\"function\",\"function\":{\"name\":\"media_import\",\"arguments\":\"{\\\"url\\\":\\\"%s\\\"}\"}}]}}]}\n", sourceURL)
		} else {
			sawVisual.Store(strings.Contains(string(encoded), `"image_url"`) && strings.Contains(string(encoded), "data:image/png;base64,"))
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"가져온 이미지를 확인했습니다."}}]}`)
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	mediaStore, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{cfg: config.Config{ASR: config.ASRConfig{FFmpegEndpoint: mediaServer.URL, Timeout: "5s"}}, media: mediaStore}
	var attached db.Attachment
	var events []string
	result, err := runCompletionLoopForSessionWithMedia(
		s, "session", context.Background(), llm.New(modelServer.URL, "test-model", ""),
		[]llm.Message{{Role: "user", Content: "이 URL 이미지를 설명해줘 " + sourceURL}}, "test-model", "low", "",
		config.ToolsConfig{MediaImportEnabled: true, MaxRounds: 3, Timeout: "1s"}, false,
		func(event string, _ any) error { events = append(events, event); return nil },
		func(item db.Attachment) error { attached = item; return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "가져온 이미지를 확인했습니다." || attached.MIME != "image/png" || attached.SourceURL != sourceURL || !sawVisual.Load() {
		t.Fatalf("media tool did not complete multimodal round: result=%+v attachment=%+v visual=%v", result, attached, sawVisual.Load())
	}
	if !strings.Contains(strings.Join(events, ","), "media_attached") {
		t.Fatalf("media attachment event missing: %v", events)
	}
}

func TestImportedMediaReplacementsReadsPreviousToolResult(t *testing.T) {
	trace := []db.ToolEvent{{Name: "media_import", Result: `{"source_url":"https://example.com/video","attachment":{"id":"old-video"}}`}}
	replacements := importedMediaReplacements(trace)
	if replacements["https://example.com/video"] != "old-video" {
		t.Fatalf("unexpected replacements: %+v", replacements)
	}
}

func TestMediaImportRejectsURLNotSuppliedByUser(t *testing.T) {
	conversation := []llm.Message{{Role: "user", Content: "https://example.com/allowed"}}
	if userSuppliedMediaURL(conversation, "https://example.com/not-allowed") {
		t.Fatal("unexpected URL was accepted")
	}
}

func TestMediaImportAcceptsURLInMultimodalUserText(t *testing.T) {
	const target = "https://example.com/video?id=42"
	conversation := []llm.Message{{Role: "user", Content: []map[string]any{
		{"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,abc"}},
		{"type": "text", "text": "이 영상도 함께 확인해줘 " + target},
	}}}
	if !userSuppliedMediaURL(conversation, target) {
		t.Fatal("URL in a multimodal user's text part was rejected")
	}
}
