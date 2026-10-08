package server

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"strings"
	"testing"
)

func TestContextReadMissingAndOtherConversation(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "context.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	for _, id := range []string{"current", "other"} {
		if _, err := store.CreateSession(id, id, "model", "none"); err != nil {
			t.Fatal(err)
		}
	}
	own, err := store.AddMessage("current", "user", strings.Repeat("가", 4001), "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	foreign, err := store.AddMessage("other", "user", "private-other-session", "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	archive, err := store.ArchiveContextTool("other", "fixture", "private-other-session")
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{db: store, cfg: config.Config{Context: config.ContextConfig{Enabled: true}}}
	registry := newCompletionToolRegistry(server, "current", config.ToolsConfig{}, false, nil)
	invoke := func(args string) (registeredToolResult, error) {
		call := llm.ToolCall{}
		call.Function.Name = "context_read"
		call.Function.Arguments = args
		return registry.handlers["context_read"](context.Background(), call, nil, nil)
	}
	for _, args := range []string{fmt.Sprintf(`{"message_id":%d}`, foreign.ID), fmt.Sprintf(`{"archive_id":%d}`, archive), `{"message_id":999999}`, `{"archive_id":999999}`} {
		result, err := invoke(args)
		if err != nil {
			t.Fatal(err)
		}
		var value map[string]any
		if err := json.Unmarshal([]byte(result.Result), &value); err != nil {
			t.Fatal(err)
		}
		if value["status"] != "not_found" || value["found"] != false || strings.Contains(result.Result, "private-other-session") || strings.Contains(result.Result, "sql:") {
			t.Fatalf("bad missing result: %s", result.Result)
		}
	}
	result, err := invoke(fmt.Sprintf(`{"message_id":%d,"offset":4000}`, own.ID))
	if err != nil {
		t.Fatal(err)
	}
	var value map[string]any
	if err := json.Unmarshal([]byte(result.Result), &value); err != nil {
		t.Fatal(err)
	}
	if value["content"] != "가" || value["complete"] != true || value["next_offset"] != float64(4001) || value["found"] != true {
		t.Fatalf("pagination broken: %s", result.Result)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := invoke(fmt.Sprintf(`{"message_id":%d}`, own.ID)); err == nil {
		t.Fatal("database failure must not be reported as not_found")
	}
}
