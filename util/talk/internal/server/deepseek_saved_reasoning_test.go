package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestSavedDeepSeekToolHistoryReachesNextTurnAndBudget(t *testing.T) {
	thought := strings.Repeat("Completed earlier reasoning. ", 64)
	for _, tc := range []struct {
		name, modelType string
		configuredTools bool
		override        *bool
		retain          bool
	}{
		{name: "v41 tools", modelType: "deepseek-v4", configuredTools: true, retain: true},
		{name: "no tools", modelType: "deepseek-v4"},
		{name: "disabled for request", modelType: "deepseek-v4", configuredTools: true, override: savedReasoningBool(false)},
		{name: "enabled for request", modelType: "deepseek-v4", override: savedReasoningBool(true), retain: true},
		{name: "other engine", modelType: "other", configuredTools: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := context.Background()
			if tc.override != nil {
				ctx = context.WithValue(ctx, contextToolsKey{}, *tc.override)
			}
			cfg := config.Config{}
			cfg.Model.ModelType = tc.modelType
			cfg.Tools.Enabled = tc.configuredTools
			items := modelHistory([]db.Message{
				{ID: 1, Role: "user", Status: db.MessageCompleted, Content: "previous question"},
				{ID: 2, Role: "assistant", Status: db.MessageCompleted, Content: "previous answer", Reasoning: thought},
				{ID: 3, Role: "assistant", Status: db.MessageFailed, Reasoning: "abandoned loop"},
				{ID: 4, Role: "user", Status: db.MessagePending, Content: "next question"},
			}, 4)
			s := &Server{}
			messages, err := s.llmMessages(ctx, items, cfg)
			if err != nil {
				t.Fatal(err)
			}
			want := ""
			if tc.retain {
				want = thought
			}
			if len(messages) != 3 || messages[1].ReasoningContent != want {
				t.Fatalf("saved reasoning missing or replayed in wrong mode: %+v", messages)
			}
			preview, incomplete := s.previewContextMessages(ctx, items, cfg)
			if incomplete || preview[1].ReasoningContent != want || estimateModelInput(preview, nil, 0) != estimateModelInput(messages, nil, 0) {
				t.Fatal("context preview must count the same saved reasoning as the actual request")
			}
			api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var body struct {
					Messages []llm.Message  `json:"messages"`
					Options  map[string]any `json:"chat_template_kwargs"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				if len(body.Messages) != 3 || body.Messages[1].ReasoningContent != want {
					t.Errorf("wire history lost reasoning: %+v", body.Messages)
				}
				if tc.retain && body.Options["drop_thinking"] != false {
					t.Error("encoder would discard saved reasoning")
				}
				raw, _ := json.Marshal(body.Messages)
				if strings.Contains(string(raw), "abandoned loop") {
					t.Error("failed reasoning entered the next request")
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			}))
			defer api.Close()
			var tools []llm.Tool
			if contextToolsEnabled(ctx, cfg.Tools.Enabled) {
				tools = []llm.Tool{{Type: "function", Function: llm.ToolFunction{Name: "lookup", Parameters: json.RawMessage(`{"type":"object"}`)}}}
			}
			_, err = llm.New(api.URL, "model", "", tc.modelType).Stream(ctx, messages, "model", "high", tools, func(string, string) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func savedReasoningBool(v bool) *bool { return &v }
