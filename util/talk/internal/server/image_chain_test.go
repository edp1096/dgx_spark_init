package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"strings"
	"testing"
)

func TestGeneratedImageAvailableInSameTurnOnly(t *testing.T) {
	s, source := testImageServer(t)
	calls := 0
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	cfg := config.ImageConfig{Endpoint: worker.URL, Model: "test", Mode: "paint", DefaultSize: "1024x1024", Timeout: "2s"}
	ctx := context.WithValue(context.Background(), turnImageKey{}, &turnImages{sessionID: "session", items: make(map[string]db.Attachment)})
	invoke := func(ctx context.Context, id string) (registeredToolResult, error) {
		b, _ := json.Marshal(map[string]any{"operation": "background_remove", "source_image_id": id})
		return s.executeImageGenerateTool(ctx, "session", cfg, llm.ToolCall{Function: llm.FunctionCall{Name: "image_generate", Arguments: string(b)}}, func(string, any) error { return nil })
	}
	first, e := invoke(ctx, source.ID)
	if e != nil {
		t.Fatal(e)
	}
	id := first.Attachments[0].ID
	stored, e := s.sessionImageAttachments("session")
	if e != nil {
		t.Fatal(e)
	}
	if _, ok := stored[id]; ok {
		t.Fatal("test must cover image not yet persisted in conversation")
	}
	if _, e = invoke(ctx, id); e != nil {
		t.Fatalf("chained output lookup failed: %v", e)
	}
	if _, e = invoke(context.Background(), id); e == nil || !strings.Contains(e.Error(), "not available") {
		t.Fatalf("turn-local image escaped scope: %v", e)
	}
	other := context.WithValue(context.Background(), turnImageKey{}, &turnImages{sessionID: "other", items: map[string]db.Attachment{id: first.Attachments[0]}})
	if _, e = invoke(other, id); e == nil {
		t.Fatal("cross-session image accepted")
	}
	if calls != 2 {
		t.Fatalf("unexpected worker calls %d", calls)
	}
}
