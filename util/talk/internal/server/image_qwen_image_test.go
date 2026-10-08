package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

func TestQwenImageToolsExposeNativeOperationsAndSendNoLegacyOptions(t *testing.T) {
	schema := string(imageGenerateToolDefinition("qwen-image21").Function.Parameters)
	prompt := imageToolSystemPrompt("qwen-image21")
	for _, field := range []string{"head_swap_strength", "background_method", "user_loras"} {
		if strings.Contains(schema, `"`+field+`"`) {
			t.Fatal("legacy tool option", field)
		}
	}
	for _, old := range []string{"BFS Klein", "specialized LoRA", "via rembg"} {
		if strings.Contains(prompt, old) {
			t.Fatal("legacy prompt", old)
		}
	}
	s, _ := testImageServer(t)
	source, err := s.media.SaveReader(bytes.NewReader(onePixelPNG), "source.png", "image/png", media.MaxImageBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.db.AddMessage("session", "user", "remove background", "", nil, []db.Attachment{source}); err != nil {
		t.Fatal(err)
	}
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		if payload["operation"] != "background_remove" || payload["source_image"] == nil || payload["background_method"] != nil || payload["head_swap_strength"] != nil {
			t.Errorf("invalid native payload %+v", payload)
		}
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	cfg := config.ImageConfig{Endpoint: worker.URL, Model: "qwen-image-2.1-uc-nvfp4", Mode: "qwen-image21", DefaultSize: "1024x1024", Timeout: "2s"}
	call := llm.ToolCall{Function: llm.FunctionCall{Name: "image_generate", Arguments: `{"operation":"background_remove","source_image_id":"` + source.ID + `"}`}}
	result, err := s.executeImageGenerateTool(context.Background(), "session", cfg, call, func(string, any) error { return nil })
	if err != nil || len(result.Attachments) != 1 {
		t.Fatalf("native tool failed: %v %+v", err, result)
	}
	call.Function.Arguments = `{"operation":"background_remove","source_image_id":"` + source.ID + `","background_method":"rembg"}`
	if _, err = s.executeImageGenerateTool(context.Background(), "session", cfg, call, func(string, any) error { return nil }); err == nil {
		t.Fatal("legacy option accepted")
	}
}
