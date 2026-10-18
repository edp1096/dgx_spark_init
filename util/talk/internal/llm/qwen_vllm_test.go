package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestQwenVLLMRequestsPassThinkingControlsToTemplate(t *testing.T) {
	for _, effort := range []string{"none", "low", "medium", "xhigh"} {
		t.Run(effort, func(t *testing.T) {
			var payload map[string]any
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					t.Error(err)
				}
				fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"42"},"finish_reason":"stop"}]}`)
				fmt.Fprintln(w, "data: [DONE]")
			}))
			defer backend.Close()
			client := New(backend.URL, "fp4-local", "", "qwen3.8-vllm")
			result, err := client.Stream(WithOutputLimit(context.Background(), 64), []Message{{Role: "user", Content: "17+25"}}, "fp4-local", effort, nil, func(string, string) error { return nil })
			if err != nil || result.Content != "42" {
				t.Fatal(result, err)
			}
			kw, ok := payload["chat_template_kwargs"].(map[string]any)
			if !ok || kw["enable_thinking"] != (effort != "none") {
				t.Fatal("vLLM did not receive explicit thinking control", payload)
			}
			if effort == "none" {
				if _, exists := kw["reasoning_effort"]; exists {
					t.Fatal("disabled thinking must not pass unsupported effort none to the template")
				}
			} else if kw["reasoning_effort"] != effort {
				t.Fatal("reasoning effort was not forwarded to the model template", payload)
			}
			if _, exists := payload["reasoning_effort"]; exists {
				t.Fatal("unexpected top-level effort")
			}
		})
	}
}
