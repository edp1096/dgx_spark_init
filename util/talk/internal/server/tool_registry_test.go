package server

import (
	"context"
	"strings"
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

func TestCompletionToolRegistryLoadsOnlySkillsForActiveToolsets(t *testing.T) {
	registry := newCompletionToolRegistry(nil, "", config.ToolsConfig{Enabled: true, SkillsEnabled: true, SearchResults: 3, Timeout: "1s"}, true, nil)
	if len(registry.definitions) != 3 || len(registry.prompts) != 2 {
		t.Fatalf("unexpected registry: definitions=%d prompts=%d", len(registry.definitions), len(registry.prompts))
	}
	if _, ok := registry.handlers["skill_view"]; !ok {
		t.Fatal("skill_view handler was not registered")
	}
	result, err := registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "skill_view", Arguments: `{"name":"web-research"}`}}, nil, nil)
	if err != nil || !strings.Contains(result.Result, "web_search") {
		t.Fatalf("result=%s err=%v", result.Result, err)
	}
	_, err = registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "skill_view", Arguments: `{"name":"ssh-inspection"}`}}, nil, nil)
	if err == nil {
		t.Fatal("inactive SSH skill was available")
	}
}

func TestCompletionToolRegistryRejectsUnknownTool(t *testing.T) {
	registry := completionToolRegistry{handlers: make(map[string]registeredToolHandler)}
	_, err := registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "missing"}}, nil, nil)
	if err == nil {
		t.Fatal("expected unknown tool error")
	}
}
