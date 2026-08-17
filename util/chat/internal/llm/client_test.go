package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

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
