package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
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

func TestStreamTerminationPreservesPartialOutput(t *testing.T) {
	for _, tc := range []struct{ name, end, want string }{
		{"limit", `{"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":20,"completion_tokens":512,"total_tokens":532}}` + "\ndata: [DONE]", "finish_reason=length"},
		{"lost connection", "", "종료 신호 없이"},
		{"filter", `{"choices":[{"delta":{},"finish_reason":"content_filter"}]}`, "content_filter"},
		{"backend error", `{"error":{"message":"worker disconnected"}}`, "worker disconnected"},
		{"invalid event", `{invalid`, "스트림 형식 오류"},
		{"normal", `{"choices":[{"delta":{},"finish_reason":"stop"}]}`, ""},
		{"compatible done", `[DONE]`, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning_content":"analysis","content":"func main() {"}}]}`)
				if tc.end != "" {
					fmt.Fprintln(w, "data: "+tc.end)
				}
			}))
			defer backend.Close()
			result, err := New(backend.URL, "model", "").Stream(WithOutputLimit(context.Background(), 512), nil, "model", "none", nil, func(string, string) error { return nil })
			if (err != nil) != (tc.want != "") || (err != nil && !strings.Contains(err.Error(), tc.want)) {
				t.Fatalf("unexpected error: %v", err)
			}
			if result.Content != "func main() {" || result.Reasoning != "analysis" {
				t.Fatalf("partial output lost: %+v", result)
			}
			if tc.name == "limit" && (result.FinishReason != "length" || result.Usage.CompletionTokens != 512) {
				t.Fatalf("termination metadata lost: %+v", result)
			}
		})
	}
}

func TestEXL3ReasoningTemplateOptions(t *testing.T) {
	for _, tc := range []struct{ input, want string }{
		{"none", "none"}, {"off", "none"}, {"", "medium"}, {"low", "low"}, {"medium", "medium"}, {"xhigh", "xhigh"}, {"on", "xhigh"}, {"high", "xhigh"}, {"max", "medium"}, {"unknown", "medium"},
	} {
		p := map[string]any{}
		applyReasoningOptions(p, "qwen38fn_exl3", tc.input)
		kw := p["chat_template_kwargs"].(map[string]any)
		if kw["enable_thinking"] != (tc.want != "none") {
			t.Fatalf("%s: %#v", tc.input, p)
		}
		if tc.want == "none" {
			if _, ok := kw["reasoning_effort"]; ok {
				t.Fatalf("off includes effort: %#v", p)
			}
		} else if kw["reasoning_effort"] != tc.want {
			t.Fatalf("%s: %#v", tc.input, p)
		}
		if _, ok := p["reasoning_effort"]; ok {
			t.Fatalf("effort must be passed directly to template: %#v", p)
		}
	}
}

func TestOrnithThinkingToggle(t *testing.T) {
	for _, effort := range []string{"none", "on", "low", "high"} {
		payload := map[string]any{}
		applyReasoningOptions(payload, "qwen3.5", effort)
		options, ok := payload["chat_template_kwargs"].(map[string]any)
		if !ok || options["enable_thinking"] != (effort != "none") {
			t.Fatalf("Incorrect Ornith toggle: %#v", payload)
		}
		if _, ok := payload["reasoning_effort"]; ok {
			t.Fatal("Ornith must use its template toggle")
		}
	}
}

func TestGemmaVLLMThinkingBudgetUsesNativeField(t *testing.T) {
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
	client := New(server.URL, "model", "", "gemma4-vllm").WithThinkingBudget(768)
	for _, effort := range []string{"on", "none"} {
		if _, err := client.Stream(context.Background(), []Message{{Role: "user", Content: "test"}}, "model", effort, nil, func(string, string) error { return nil }); err != nil {
			t.Fatal(err)
		}
	}
	if payloads[0]["thinking_token_budget"] != float64(768) {
		t.Fatalf("Missing native budget: %#v", payloads[0])
	}
	for _, payload := range payloads {
		if _, exists := payload["custom_params"]; exists {
			t.Fatal("SGLang field sent to vLLM")
		}
	}
	if _, exists := payloads[1]["thinking_token_budget"]; exists {
		t.Fatal("Budget should be omitted when thinking is off")
	}
}

func TestReasoningBudgetIsPerRequestAndKeepsChatEffort(t *testing.T) {
	var payloads []map[string]any
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var p map[string]any
		json.NewDecoder(r.Body).Decode(&p)
		payloads = append(payloads, p)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer backend.Close()
	c := New(backend.URL, "qwen38fn_exl3", "", "qwen38fn_exl3")
	for _, ctx := range []context.Context{WithReasoningBudget(context.Background(), 2048), context.Background()} {
		if _, err := c.Stream(ctx, []Message{{Role: "user", Content: "test"}}, "qwen38fn_exl3", "xhigh", nil, func(string, string) error { return nil }); err != nil {
			t.Fatal(err)
		}
	}
	if payloads[0]["reasoning_budget_tokens"] != float64(2048) {
		t.Fatal("native budget not forwarded", payloads[0])
	}
	if _, ok := payloads[1]["reasoning_budget_tokens"]; ok {
		t.Fatal("budget leaked into ordinary chat")
	}
	for _, p := range payloads {
		if p["chat_template_kwargs"].(map[string]any)["reasoning_effort"] != "xhigh" {
			t.Fatal("chat effort changed")
		}
	}
}

func TestNativeReasoningBudgetDoesNotAssumeVeloSupportsTabbyAPI(t *testing.T) {
	if !SupportsReasoningBudget("qwen38fn_exl3", "qwen38fn_exl3") || SupportsReasoningBudget("qwen38fn_exl3_q4", "qwen38fn_exl3") || SupportsReasoningBudget("model", "generic") {
		t.Fatal("runtime capability distinction lost")
	}
}

func TestVeloReasoningBudgetRequiresStructuredOutput(t *testing.T) {
	var bodies []map[string]any
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		bodies = append(bodies, body)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer backend.Close()
	c := New(backend.URL, "qwen38fn_exl3_q4", "", "qwen38fn_exl3")
	budget := WithReasoningBudget(context.Background(), 256)
	for _, ctx := range []context.Context{WithJSONSchema(budget, "test", json.RawMessage(`{"type":"object"}`)), budget, context.Background()} {
		if _, err := c.Stream(ctx, []Message{{Role: "user", Content: "test"}}, "qwen38fn_exl3_q4", "medium", nil, func(string, string) error { return nil }); err != nil {
			t.Fatal(err)
		}
	}
	if bodies[0]["reasoning_budget_tokens"] != float64(256) {
		t.Fatal("structured reasoning budget absent")
	}
	for _, body := range bodies[1:] {
		if _, ok := body["reasoning_budget_tokens"]; ok {
			t.Fatal("schema-only capability leaked into chat")
		}
	}
	if !SupportsStructuredReasoningBudget("qwen38fn_exl3_q4", "qwen38fn_exl3") || SupportsStructuredReasoningBudget("other", "generic") {
		t.Fatal("incorrect schema runtime capability")
	}
}

func TestTensorFoldReasoningBudgetUsesTheServingAPI(t *testing.T) {
	var payload map[string]any
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&payload)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer backend.Close()
	c := New(backend.URL, "qwen38fn_exl3", "", "qwen3.8-tensorfold")
	_, err := c.Stream(WithReasoningBudget(context.Background(), 2048), []Message{{Role: "user", Content: "test"}}, "qwen38fn_exl3", "medium", nil, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if payload["thinking_budget"] != float64(2048) || payload["reasoning_budget_tokens"] != nil {
		t.Fatal("TensorFold received legacy Tabby budget fields", payload)
	}
	options := payload["chat_template_kwargs"].(map[string]any)
	if options["enable_thinking"] != true || options["reasoning_effort"] != "medium" {
		t.Fatal("thinking settings lost", options)
	}
}
