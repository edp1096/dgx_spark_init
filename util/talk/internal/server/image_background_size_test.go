package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"strings"
	"testing"
)

func TestBackgroundRemovalUsesSourceDimensions(t *testing.T) {
	s, _ := testImageServer(t)
	var data bytes.Buffer
	if err := png.Encode(&data, image.NewNRGBA(image.Rect(0, 0, 1600, 864))); err != nil {
		t.Fatal(err)
	}
	source, err := s.media.SaveReader(bytes.NewReader(data.Bytes()), "wide.png", "image/png", media.MaxImageBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.db.AddMessage("session", "user", "배경 지워라", "", nil, []db.Attachment{source}); err != nil {
		t.Fatal(err)
	}
	calls := 0
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		if payload["operation"] == "background_remove" && payload["background_method"] != "rembg" {
			t.Errorf("wrong operation: %+v", payload)
		}
		encoded := payload["source_image"].(string)
		b, e := base64.StdEncoding.DecodeString(strings.SplitN(encoded, ",", 2)[1])
		if e != nil {
			t.Error(e)
		}
		c, _, e := image.DecodeConfig(bytes.NewReader(b))
		if e != nil || c.Width != 1600 || c.Height != 864 {
			t.Error("source resized")
		}
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(data.Bytes())}}})
	}))
	defer worker.Close()
	cfg := config.ImageConfig{Endpoint: worker.URL, Model: "test", Mode: "paint", DefaultSize: "1024x1024", Timeout: "2s"}
	invoke := func(operation, size string) error {
		b, _ := json.Marshal(map[string]any{"operation": operation, "prompt": "remove background", "source_image_id": source.ID, "size": size})
		_, e := s.executeImageGenerateTool(context.Background(), "session", cfg, llm.ToolCall{Function: llm.FunctionCall{Name: "image_generate", Arguments: string(b)}}, func(string, any) error { return nil })
		return e
	}
	for _, size := range []string{"768x416", "1600x864"} {
		if e := invoke("background_remove", size); e != nil {
			t.Fatal(e)
		}
	}
	for _, size := range []string{"1024x560", "768x416"} {
		if e := invoke("background_cleanup", size); e != nil {
			t.Fatal(e)
		}
	}

	if calls != 4 {
		t.Fatalf("not all source-sized operations reached worker: calls=%d", calls)
	}
}
