package server

import (
	"context"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
)

func TestCompletionToolRegistryRegistersEnabledWebTools(t *testing.T) {
	registry := newCompletionToolRegistry(nil, "", config.ToolsConfig{Enabled: true, SearchResults: 3, Timeout: "1s"}, true, nil)
	if len(registry.definitions) != 2 || len(registry.prompts) != 1 {
		t.Fatalf("unexpected registry: definitions=%d prompts=%d", len(registry.definitions), len(registry.prompts))
	}
	if _, ok := registry.handlers["web_search"]; !ok {
		t.Fatal("web_search handler was not registered")
	}
}

func TestCompletionToolRegistryRejectsUnknownTool(t *testing.T) {
	registry := completionToolRegistry{handlers: make(map[string]registeredToolHandler)}
	_, err := registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "missing"}}, nil, nil)
	if err == nil {
		t.Fatal("expected unknown tool error")
	}
}
