package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestStreamSendsGemmaThinkingBudgetOnlyWhenThinking(t *testing.T) {
	var payloads []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		payloads = append(payloads, payload)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer server.Close()

	client := New(server.URL, "model", "", "gemma4").WithThinkingBudget(768)
	for _, effort := range []string{"on", "none"} {
		if _, err := client.Stream(context.Background(), []Message{{Role: "user", Content: "test"}}, "model", effort, nil, func(string, string) error { return nil }); err != nil {
			t.Fatal(err)
		}
	}
	custom, ok := payloads[0]["custom_params"].(map[string]any)
	if !ok || custom["thinking_budget"] != float64(768) {
		t.Fatalf("thinking request missing budget: %#v", payloads[0])
	}
	if _, exists := payloads[1]["custom_params"]; exists {
		t.Fatalf("non-thinking request must omit budget: %#v", payloads[1])
	}
}

func TestReasoningValueSupportsNamesAndNumericValues(t *testing.T) {
	if got := reasoningValue("xhigh"); got != "xhigh" {
		t.Fatalf("named effort changed: %#v", got)
	}
	if got := reasoningValue("0.75"); got != float64(0.75) {
		t.Fatalf("numeric effort was not parsed: %#v", got)
	}
	if got := reasoningValue(""); got != nil {
		t.Fatalf("empty effort should be omitted: %#v", got)
	}
}

func TestApplyReasoningOptionsUsesGemmaThinkingToggle(t *testing.T) {
	for _, test := range []struct {
		effort  string
		enabled bool
	}{{"none", false}, {"0", false}, {"low", true}, {"on", true}, {"xhigh", true}} {
		payload := map[string]any{}
		applyReasoningOptions(payload, "gemma4", test.effort)
		kwargs, ok := payload["chat_template_kwargs"].(map[string]any)
		if !ok || kwargs["enable_thinking"] != test.enabled {
			t.Fatalf("effort %q produced %#v", test.effort, payload)
		}
		if _, exists := payload["reasoning_effort"]; exists {
			t.Fatalf("Gemma 4 must not receive reasoning_effort: %#v", payload)
		}
	}
}

func TestApplyReasoningOptionsUsesEXL3ThinkingToggle(t *testing.T) {
	for _, test := range []struct {
		effort  string
		enabled bool
	}{{"none", false}, {"off", false}, {"on", true}, {"high", true}} {
		payload := map[string]any{}
		applyReasoningOptions(payload, "qwen3.8-exl3", test.effort)
		kwargs, ok := payload["chat_template_kwargs"].(map[string]any)
		if !ok || kwargs["enable_thinking"] != test.enabled {
			t.Fatalf("effort %q produced %#v", test.effort, payload)
		}
		if _, exists := payload["reasoning_effort"]; exists {
			t.Fatalf("EXL3 must not receive reasoning_effort: %#v", payload)
		}
	}
}

func TestApplyReasoningOptionsPreservesGenericEffort(t *testing.T) {
	payload := map[string]any{}
	applyReasoningOptions(payload, "generic", "0.75")
	if payload["reasoning_effort"] != float64(0.75) {
		t.Fatalf("unexpected generic options: %#v", payload)
	}
}

func TestApplyReasoningOptionsConstrainsQwenEffort(t *testing.T) {
	for _, test := range []struct{ input, want string }{{"none", "none"}, {"low", "low"}, {"medium", "medium"}, {"xhigh", "xhigh"}, {"high", "medium"}, {"on", "medium"}} {
		payload := map[string]any{}
		applyReasoningOptions(payload, "qwen3.8", test.input)
		if payload["reasoning_effort"] != test.want {
			t.Fatalf("Qwen effort %q produced %#v, want %q", test.input, payload, test.want)
		}
	}
}

func TestApplyReasoningOptionsUsesCompatibleGLM53Controls(t *testing.T) {
	for _, test := range []struct {
		input   string
		want    string
		enabled bool
	}{{"none", "off", false}, {"off", "off", false}, {"low", "low", true}, {"high", "high", true}, {"max", "max", true}, {"xhigh", "max", true}} {
		payload := map[string]any{}
		applyReasoningOptions(payload, "glm5.3", test.input)
		kwargs, ok := payload["chat_template_kwargs"].(map[string]any)
		if !ok || kwargs["enable_thinking"] != test.enabled || kwargs["clear_thinking"] != true {
			t.Fatalf("GLM-5.3 effort %q produced %#v", test.input, payload)
		}
		if test.enabled {
			if payload["reasoning_effort"] != test.want {
				t.Fatalf("GLM-5.3 effort %q produced %#v, want %q", test.input, payload, test.want)
			}
		} else if _, exists := payload["reasoning_effort"]; exists {
			t.Fatalf("GLM-5.3 off must omit reasoning_effort: %#v", payload)
		}
	}
}

func TestStreamAssemblesToolCallDeltas(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"web_search","arguments":"{\"query\":"}}]}}]}`)
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"dgx spark\"}"}}]},"finish_reason":"tool_calls"}]}`)
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer server.Close()

	client := New(server.URL, "model", "")
	tools := []Tool{{Type: "function", Function: ToolFunction{Name: "web_search", Parameters: []byte(`{"type":"object"}`)}}}
	result, err := client.Stream(context.Background(), []Message{{Role: "user", Content: "search"}}, "model", "low", tools, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolCalls) != 1 || result.ToolCalls[0].Function.Name != "web_search" || result.ToolCalls[0].Function.Arguments != `{"query":"dgx spark"}` {
		t.Fatalf("unexpected tool calls: %+v", result.ToolCalls)
	}
}

func TestDeepSeekV4ThinkingTemplateContract(t *testing.T) {
	for _, effort := range []string{"off", "low", "high", "max"} {
		p := map[string]any{}
		applyReasoningOptions(p, "deepseek-v4", effort)
		kw := p["chat_template_kwargs"].(map[string]any)
		if kw["thinking"] != (effort != "off") {
			t.Fatalf("%s: %#v", effort, p)
		}
		if _, exists := kw["enable_thinking"]; exists {
			t.Fatal("GLM option leaked into DeepSeek")
		}
		if effort != "off" && kw["reasoning_effort"] != effort {
			t.Fatalf("%s: %#v", effort, p)
		}
		if effort == "off" && kw["reasoning_effort"] != nil {
			t.Fatal("off sends effort")
		}
	}
}

func TestToolReasoningIsOnlySentToDeepSeek(t *testing.T) {
	for _, modelType := range []string{"deepseek-v4", "glm5.3", "generic"} {
		t.Run(modelType, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var payload struct {
					Messages []map[string]any `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					t.Error(err)
				}
				_, present := payload.Messages[0]["reasoning_content"]
				if present != (modelType == "deepseek-v4") {
					t.Errorf("unexpected reasoning field for %s", modelType)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"ok"}}]}`)
				fmt.Fprintln(w, "data: [DONE]")
			}))
			defer server.Close()
			messages := []Message{{Role: "assistant", ReasoningContent: "lookup reasoning"}}
			_, err := New(server.URL, "model", "", modelType).Stream(context.Background(), messages, "model", "low", nil, func(string, string) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			if messages[0].ReasoningContent != "lookup reasoning" {
				t.Fatal("caller transcript mutated")
			}
		})
	}
}
