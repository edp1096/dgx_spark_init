package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestContinuousToolsBudgetArchivesAndReplaysWithoutLosingLatest(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "tools.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	store.CreateSession("s", "test", "model", "low")
	user, _ := store.AddMessage("s", "user", "read the archived evidence", "", nil, nil)
	source, _ := store.ArchiveContextTool("s", "fixture", strings.Repeat("Z", 6000), user.ID)
	calls := 0
	trimSeen := false
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var body struct {
			Messages []llm.Message `json:"messages"`
			Limit    int           `json:"max_completion_tokens"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if body.Limit != 512 {
			t.Errorf("output reserve not forwarded: %d", body.Limit)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		var delta map[string]any
		finish := "tool_calls"
		if calls <= 2 {
			delta = map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": fmt.Sprintf("c%d", calls), "type": "function", "function": map[string]any{"name": "context_read", "arguments": fmt.Sprintf(`{"archive_id":%d,"offset":0}`, source)}}}}
		} else {
			finish = "stop"
			delta = map[string]any{"content": "done"}
			for _, m := range body.Messages {
				if m.Role == "tool" && m.ToolCallID == "c1" {
					trimSeen = strings.Contains(fmt.Sprint(m.Content), "Archived tool result")
				}
				if m.ToolCallID == "c2" && !strings.Contains(fmt.Sprint(m.Content), strings.Repeat("Z", 100)) {
					t.Error("latest tool result lost")
				}
			}
		}
		chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": delta, "finish_reason": finish}}, "usage": map[string]int{"prompt_tokens": 777, "completion_tokens": 5, "total_tokens": 782}})
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", chunk)
	}))
	defer backend.Close()
	cfg := config.Config{Context: config.ContextConfig{Enabled: true, WindowTokens: 2700, OutputReserve: 512, SafetyMargin: 256, CompactAtPercent: 80, RecentTokens: 128, ImageTokens: 100}, Tools: config.ToolsConfig{MaxRounds: 3}}
	server := &Server{db: store, cfg: cfg}
	states := []contextState{}
	emit := func(event string, value any) error {
		if event == "context" {
			states = append(states, value.(contextState))
		}
		return nil
	}
	result, err := server.runContextCompletion(context.Background(), "s", []db.Message{user}, "model", "none", cfg, llm.New(backend.URL, "model", ""), false, emit, nil)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 3 || result.Content != "done" || !trimSeen || len(result.ToolTrace) != 2 {
		t.Fatalf("calls=%d trimmed=%v result=%+v", calls, trimSeen, result)
	}
	for _, event := range result.ToolTrace {
		raw, err := store.ReadContextTool("s", event.ArchiveID)
		if err != nil || !strings.Contains(raw, strings.Repeat("Z", 100)) {
			t.Fatal("full result not preserved")
		}
	}
	if states[len(states)-1].ActualTokens != 777 {
		t.Fatal("backend usage not emitted")
	}
}

func TestRejectedSummaryKeepsCheckpointAndOriginalTranscript(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "summary.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	store.CreateSession("s", "test", "model", "low")
	for i := 0; i < 6; i++ {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		store.AddMessage("s", role, strings.Repeat("evidence ", 300), "", nil, nil)
	}
	items, _ := store.Messages("s")
	previous, _ := store.AddContextSegment("s", items[0].ID, items[1].ID, "previous", "previous", "model", 1000)
	calls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"## Objective\ntruncated"},"finish_reason":"length"}]}`)
	}))
	defer backend.Close()
	cfg := config.Config{Context: config.ContextConfig{Enabled: true, WindowTokens: 16000, OutputReserve: 1024, SafetyMargin: 512, CompactAtPercent: 80, RecentTokens: 256, ImageTokens: 100}}
	server := &Server{db: store, cfg: cfg}
	_, state, err := server.prepareContext(context.Background(), "s", items, "model", cfg, llm.New(backend.URL, "model", ""), true)
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || state.Compacted || state.AppliedSegmentID != previous.ID || state.Notice == "" {
		t.Fatalf("failed summary replaced checkpoint: %+v", state)
	}
	rows, _ := store.ContextSegments("s")
	after, _ := store.Messages("s")
	if len(rows) != 1 || len(after) != len(items) || after[0].Content != items[0].Content {
		t.Fatal("source or checkpoint changed")
	}
}
