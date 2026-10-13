package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestGemma31ContextEstimateIncludesTransmittedCompletionPolicy(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "context.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	model := "Huihui-gemma-4-31B-it-abliterated-v2-NVFP4"
	store.CreateSession("s", "test", model, "on")
	user, err := store.AddMessage("s", "user", "소설을 써 줘.", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var wire struct {
		Messages []llm.Message `json:"messages"`
		Tools    []llm.Tool    `json:"tools"`
	}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&wire); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer backend.Close()
	cfg := config.Config{Model: config.ModelConfig{SystemPrompt: "생각은 짧게 한다."}, Context: config.ContextConfig{Enabled: true, WindowTokens: 16000, OutputReserve: 1024, SafetyMargin: 512, CompactAtPercent: 80, RecentTokens: 128, ImageTokens: 100}}
	s := &Server{cfg: cfg, db: store}
	client := llm.New(backend.URL, model, "", "gemma4")
	ctx := context.Background()
	messages, state, err := s.buildContext(ctx, "s", []db.Message{user}, model, cfg, client, false, false)
	if err != nil {
		t.Fatal(err)
	}
	registry := newCompletionToolRegistryForContext(ctx, s, "s", cfg.Tools, false, func(db.Attachment) error { return nil })
	assembled := assembleModelConversation(cfg.Model.SystemPrompt, messages, registry.prompts, 0, true)
	rawEstimate := estimateModelInput(assembled, registry.definitions, cfg.Context.ImageTokens)
	if _, err = client.Stream(ctx, assembled, model, "on", registry.definitions, func(string, string) error { return nil }); err != nil {
		t.Fatal(err)
	}
	expected := estimateModelInput(wire.Messages, wire.Tools, cfg.Context.ImageTokens)
	if state.EstimatedTokens != expected {
		t.Fatalf("compaction estimate %d does not match transmitted request %d", state.EstimatedTokens, expected)
	}
	if state.EstimatedTokens <= rawEstimate {
		t.Fatal("model-specific completion policy was not reserved before compaction")
	}
}
