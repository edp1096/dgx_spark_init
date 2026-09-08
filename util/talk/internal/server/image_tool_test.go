package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
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

var onePixelPNG, _ = base64.StdEncoding.DecodeString("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")

func TestImageGenerateToolDefinitionSeparatesBasicAndExtendedFeatures(t *testing.T) {
	decodeProperties := func(tool llm.Tool) map[string]json.RawMessage {
		t.Helper()
		var schema struct {
			Properties map[string]json.RawMessage `json:"properties"`
		}
		if err := json.Unmarshal(tool.Function.Parameters, &schema); err != nil {
			t.Fatal(err)
		}
		return schema.Properties
	}

	basic := decodeProperties(imageGenerateToolDefinition("basic"))
	if len(basic) != 3 || basic["prompt"] == nil || basic["size"] == nil || basic["seed"] == nil || basic["operation"] != nil {
		t.Fatalf("unexpected basic image schema: %v", basic)
	}
	extended := decodeProperties(imageGenerateToolDefinition("extended"))
	if extended["operation"] == nil || extended["source_image_id"] == nil || extended["user_loras"] == nil {
		t.Fatalf("extended image schema is missing editing features: %v", extended)
	}

	payload := commonImagePayload(imageGenerationArgs{Prompt: "a fox", Size: "512x512"}, false)
	if len(payload) != 2 || payload["filter_mode"] != nil || payload["sampler_name"] != nil {
		t.Fatalf("basic payload leaked extended API fields: %v", payload)
	}
}

func TestImageIdentityEditUsesConversationAttachmentAndReturnsAssistantMedia(t *testing.T) {
	var request map[string]any
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"seed": 42, "data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()

	s, source := testImageServer(t)
	call := llm.ToolCall{ID: "image-one", Function: llm.FunctionCall{Name: "image_generate", Arguments: `{"operation":"identity_edit","prompt":"Change the jacket to red while preserving identity.","source_image_id":"` + source.ID + `","size":"512x512","styles":[{"name":"retroanime","strength":0.8}]}`}}
	result, err := s.executeImageGenerateTool(context.Background(), "session", config.ImageConfig{Endpoint: worker.URL, Model: "test-image", Mode: "extended", DefaultSize: "512x512", Timeout: "2s"}, call, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Attachments) != 1 || len(result.Followups) != 1 || result.Attachments[0].MIME != "image/png" {
		t.Fatalf("unexpected result: %+v", result)
	}
	if request["model"] != "test-image" || request["source_image"] == nil || request["steps"] != float64(10) {
		t.Fatalf("unexpected request: %+v", request)
	}
	if !strings.HasPrefix(request["source_image"].(string), "data:image/png;base64,") {
		t.Fatalf("source image not encoded: %+v", request["source_image"])
	}
}

func TestImageSpriteGeneratesEightDirectionsAndPacksSheet(t *testing.T) {
	var calls atomic.Int32
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_ = json.NewEncoder(w).Encode(map[string]any{"seed": calls.Load(), "data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	s, _ := testImageServer(t)
	call := llm.ToolCall{ID: "sprite", Function: llm.FunctionCall{Name: "image_generate", Arguments: `{"operation":"sprite_8way","prompt":"A small armored fox hero","size":"512x512","sprite_cell_size":512,"pixel_art":true}`}}
	result, err := s.executeImageGenerateTool(context.Background(), "session", config.ImageConfig{Endpoint: worker.URL, Model: "test-image", Mode: "extended", DefaultSize: "512x512", Timeout: "5s"}, call, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 8 || len(result.Attachments) != 1 {
		t.Fatalf("calls=%d result=%+v", calls.Load(), result)
	}
	file, err := s.media.Open(result.Attachments[0])
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	decoded, _, err := image.Decode(file)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.Bounds().Dx() != 2048 || decoded.Bounds().Dy() != 1024 {
		t.Fatalf("sheet=%v", decoded.Bounds())
	}
}

func testImageServer(t *testing.T) (*Server, db.Attachment) {
	t.Helper()
	root := t.TempDir()
	store, err := media.New(root + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	database, err := db.Open(root + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	if _, err := database.CreateSession("session", "test", "model", "low"); err != nil {
		t.Fatal(err)
	}
	source, err := store.SaveReader(bytes.NewReader(onePixelPNG), "source.png", "image/png", media.MaxImageBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := database.AddMessage("session", "user", "edit this", "", nil, []db.Attachment{source}); err != nil {
		t.Fatal(err)
	}
	return &Server{db: database, media: store}, source
}
