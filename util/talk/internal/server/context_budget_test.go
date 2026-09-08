package server

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestAssembledContextCountsToolsExtractedTextAndIgnoresImageTransport(t *testing.T) {
	messages := []llm.Message{{Role: "user", Content: []map[string]any{{"type": "text", "text": strings.Repeat("transcript ", 2000)}, {"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64," + strings.Repeat("a", 500000)}}}}}
	tool := llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "test", Description: strings.Repeat("description ", 200), Parameters: json.RawMessage(`{"type":"object"}`)}}
	n := estimateModelInput(messages, []llm.Tool{tool}, 2048)
	if n < 7500 || n > 12000 {
		t.Fatalf("text/schema/image estimate incorrect: %d", n)
	}
	if n <= estimateModelInput(messages, nil, 2048) {
		t.Fatal("schema not counted")
	}
}
func TestTrimKeepsLatestToolBatchAndOriginalMessages(t *testing.T) {
	raw := strings.Repeat("original data ", 2000)
	messages := []llm.Message{{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: "old"}}}, {Role: "tool", ToolCallID: "old", Content: raw}, {Role: "assistant", ToolCalls: []llm.ToolCall{{ID: "new"}}}, {Role: "tool", ToolCallID: "new", Content: "latest evidence"}}
	out, n := trimContextTools(messages, map[string]int64{"old": 7, "new": 8}, 500, nil, 100)
	if n != 1 || !strings.Contains(out[1].Content.(string), "archive_id=7") {
		t.Fatalf("missing reference: %+v", out)
	}
	if messages[1].Content != raw || out[3].Content != "latest evidence" || out[1].ToolCallID != "old" {
		t.Fatal("original or latest batch corrupted")
	}
	out, n = trimContextTools(messages, nil, 500, nil, 100)
	if n != 0 || out[1].Content != raw {
		t.Fatal("unarchived evidence removed")
	}
}
func TestContextOffPreviewMatchesSentHistoryDespiteSavedSummary(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "context.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	store.CreateSession("s", "test", "model", "low")
	m1, _ := store.AddMessage("s", "user", strings.Repeat("old context ", 200), "", nil, nil)
	m2, _ := store.AddMessage("s", "assistant", "answer", "", nil, nil)
	store.AddContextSegment("s", m1.ID, m2.ID, "summary", "summary", "model", 1000)
	items, _ := store.Messages("s")
	cfg := config.Config{Context: config.ContextConfig{WindowTokens: 8192, OutputReserve: 1024, SafetyMargin: 256, ImageTokens: 100, CompactAtPercent: 80}}
	server := &Server{db: store, cfg: cfg}
	client := llm.New("http://unused", "model", "")
	preview, err := server.inspectContext(context.Background(), "s", "model", items, cfg, client)
	if err != nil {
		t.Fatal(err)
	}
	sent, state, err := server.prepareContext(context.Background(), "s", items, "model", cfg, client, false)
	if err != nil {
		t.Fatal(err)
	}
	if preview.SummaryThrough != 0 || preview.SummaryTokens != 0 || preview.AppliedSegmentID != 0 || preview.ActiveStart != m1.ID {
		t.Fatalf("OFF applied summary: %+v", preview)
	}
	if preview.EstimatedTokens != state.EstimatedTokens || len(sent) != 2 {
		t.Fatalf("preview/send mismatch: %+v %+v", preview, state)
	}
}
func TestBudgetStopsOversizedLatestResultAndEmitsActualUsage(t *testing.T) {
	run := &contextRun{state: contextState{Managed: true, InputBudget: 100}, cfg: config.ContextConfig{ImageTokens: 10}}
	ctx := context.WithValue(context.Background(), contextRunKey{}, run)
	events := []contextState{}
	emit := func(event string, value any) error {
		if event == "context" {
			events = append(events, value.(contextState))
		}
		return nil
	}
	_, err := updateRequestContext(ctx, []llm.Message{{Role: "user", Content: strings.Repeat("a", 1000)}}, nil, nil, emit)
	if err == nil {
		t.Fatal("oversize request allowed")
	}
	emitContextUsage(ctx, llm.Usage{PromptTokens: 123}, emit)
	if events[len(events)-1].ActualTokens != 123 {
		t.Fatal("real usage lost")
	}
}
