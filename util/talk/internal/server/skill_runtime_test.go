package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/skills"
)

func TestStoredSkillsMigrationAndLifecycle(t *testing.T) {
	path := t.TempDir() + "/skills.db"
	store, err := db.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{db: store, cfg: config.Config{Tools: config.ToolsConfig{SkillsEnabled: true}}}
	put := func(name, body string) int {
		t.Helper()
		w := httptest.NewRecorder()
		s.skillItem(w, httptest.NewRequest("PUT", "/api/skills/"+name, strings.NewReader(body)))
		return w.Code
	}
	if code := put("review", `{"description":"Review code","instructions":"Check evidence before concluding.","toolsets":[],"enabled":true}`); code != 204 {
		t.Fatal(code)
	}
	if code := put("web-research", `{"instructions":"overwrite","enabled":true}`); code != 400 {
		t.Fatal("builtin changed", code)
	}
	if code := put("web-research", `{"enabled":false}`); code != 204 {
		t.Fatal(code)
	}
	if code := put("bad", `{"description":"x","instructions":"x","enabled":true,"toolsets":["unknown"]}`); code != 400 {
		t.Fatal(code)
	}
	store.Close()
	store, err = db.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	s.db = store
	items, err := s.allSkills()
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, item := range items {
		if item.Name == "web-research" && item.Enabled {
			t.Fatal("builtin setting lost")
		}
		if item.Name == "review" {
			found = item.Instructions == "Check evidence before concluding." && item.Enabled
		}
	}
	if !found {
		t.Fatal("custom skill did not survive reopening")
	}
	w := httptest.NewRecorder()
	s.skillItem(w, httptest.NewRequest("DELETE", "/api/skills/review", nil))
	if w.Code != 204 {
		t.Fatal(w.Code)
	}
	items, err = store.Skills()
	if err != nil || len(items) != 0 {
		t.Fatal(items, err)
	}
}
func TestSkillRequestSnapshotAndDeduplication(t *testing.T) {
	store, err := db.Open(t.TempDir() + "/test.db")
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	item := skills.Skill{Name: "review", Description: "Review code", Instructions: "original procedure", Enabled: true, Toolsets: []string{}}
	if err := store.SaveSkill(item); err != nil {
		t.Fatal(err)
	}
	s := &Server{db: store, cfg: config.Config{}}
	cfg := config.ToolsConfig{SkillsEnabled: true}
	registry := newCompletionToolRegistry(s, "", cfg, false, nil)
	if strings.Contains(strings.Join(registry.prompts, ""), item.Instructions) {
		t.Fatal("full procedure leaked into index")
	}
	item.Instructions = "edited procedure"
	store.SaveSkill(item)
	ctx := context.WithValue(context.Background(), contextRunKey{}, &contextRun{})
	trace, err := registry.selectSkills(ctx, []llm.Message{{Role: "user", Content: "@skill:review\nCheck this code"}})
	if err != nil || len(trace) != 1 {
		t.Fatal(trace, err)
	}
	if !strings.Contains(strings.Join(registry.prompts, ""), "original procedure") {
		t.Fatal("snapshot changed during request")
	}
	result, err := registry.execute(ctx, llm.ToolCall{Function: llm.FunctionCall{Name: "skill_view", Arguments: `{"name":"review"}`}}, nil, nil)
	if err != nil || strings.Contains(result.Result, "original procedure") || !strings.Contains(result.Result, "already loaded") {
		t.Fatal(result, err)
	}
	item.Enabled = false
	store.SaveSkill(item)
	registry = newCompletionToolRegistry(s, "", cfg, false, nil)
	if _, err := registry.selectSkills(ctx, []llm.Message{{Role: "user", Content: "@skill:review\nCheck"}}); err == nil {
		t.Fatal("disabled skill selected")
	}
	item.Enabled = true
	item.Toolsets = []string{"web"}
	store.SaveSkill(item)
	registry = newCompletionToolRegistry(s, "", cfg, false, nil)
	if _, ok := registry.skills["review"]; ok {
		t.Fatal("unavailable dependency exposed")
	}
}
func TestSkillSelectionOnlyFromCurrentExplicitPrefix(t *testing.T) {
	tests := []struct {
		messages []llm.Message
		want     int
	}{
		{[]llm.Message{{Role: "user", Content: "@skill:a @skill:a @skill:b\nTask"}}, 2},
		{[]llm.Message{{Role: "user", Content: "explain @skill:a"}}, 0},
		{[]llm.Message{{Role: "user", Content: "@skill:a task"}, {Role: "assistant", Content: "done"}}, 0},
		{[]llm.Message{{Role: "user", Content: "@skill:a task"}, {Role: "assistant", Content: "done"}, {Role: "user", Content: "next"}}, 0},
		{[]llm.Message{{Role: "user", Content: []map[string]any{{"type": "text", "text": "read attachment"}, {"type": "text", "text": "@skill:a"}}}}, 0},
	}
	for _, test := range tests {
		if got := requestedSkills(test.messages); len(got) != test.want {
			t.Fatalf("got %v want count %d", got, test.want)
		}
	}
}
func TestSkillContextAccounting(t *testing.T) {
	run := &contextRun{state: contextState{}, cfg: config.ContextConfig{}}
	ctx := context.WithValue(context.Background(), contextRunKey{}, run)
	selected := skills.Skill{Name: "selected", Instructions: strings.Repeat("selected ", 20)}
	auto := skills.Skill{Name: "auto", Instructions: strings.Repeat("automatic ", 20)}
	recordSkillUsage(ctx, selected, "selected")
	recordSkillUsage(ctx, auto, "auto")
	recordSkillUsage(ctx, auto, "auto")
	data, _ := json.Marshal(auto)
	messages := []llm.Message{{Role: "system", Content: selected.Instructions}, {Role: "user", Content: "question"}, {Role: "tool", Content: string(data)}}
	_, err := updateRequestContext(ctx, messages, nil, nil, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	state := run.state
	if len(state.Skills) != 2 || state.SkillTokens == 0 {
		t.Fatal(state)
	}
	sum := state.ActiveTokens + state.SystemToolTokens + state.ToolResultTokens + state.SkillTokens + state.SummaryTokens + state.RecallTokens
	if sum != state.EstimatedTokens {
		t.Fatalf("double counted: %d vs %d", sum, state.EstimatedTokens)
	}
}

func TestSkillCompletionLoadsProcedureOnlyOnce(t *testing.T) {
	for _, selected := range []bool{false, true} {
		t.Run(fmt.Sprint(selected), func(t *testing.T) {
			store, err := db.Open(t.TempDir() + "/test.db")
			if err != nil {
				t.Fatal(err)
			}
			defer store.Close()
			instruction := "Inspect the actual implementation and report the evidence."
			store.SaveSkill(skills.Skill{Name: "review", Description: "Code review workflow", Instructions: instruction, Enabled: true, Toolsets: []string{}})
			s := &Server{db: store}
			var requests atomic.Int32
			modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Messages []llm.Message `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
				}
				n := requests.Add(1)
				payload, _ := json.Marshal(body.Messages)
				count := strings.Count(string(payload), instruction)
				expected := 1
				if !selected && n == 1 {
					expected = 0
				}
				if count != expected {
					t.Errorf("request %d: instruction count %d, want %d", n, count, expected)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if n < 3 {
					fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"skill-call","type":"function","function":{"name":"skill_view","arguments":"{\"name\":\"review\"}"}}]}}]}`)
				} else {
					fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"Verified."}}]}`)
				}
				fmt.Fprintln(w, "data: [DONE]")
			}))
			defer modelServer.Close()
			input := "Review this code"
			if selected {
				input = "@skill:review\n" + input
			}
			result, err := runCompletionLoopForServer(s, context.Background(), llm.New(modelServer.URL, "test", ""), []llm.Message{{Role: "user", Content: input}}, "test", "", "base rules", config.ToolsConfig{SkillsEnabled: true, MaxRounds: 4}, false, func(string, any) error { return nil })
			if err != nil || result.Content != "Verified." || requests.Load() != 3 {
				t.Fatal(result, err, requests.Load())
			}
			if len(result.ToolTrace) < 2 {
				t.Fatal("usage trace missing")
			}
		})
	}
}
