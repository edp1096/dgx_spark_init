package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/modelidentity"
	"testing"
)

func TestQwen38FNEXL3AcceptsLegacyRequestIDWithSameReasoningProtocol(t *testing.T) {
	var payload map[string]any
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer api.Close()
	client := New(api.URL, modelidentity.LegacyQwenModel, "", modelidentity.LegacyQwenType)
	_, err := client.Stream(context.Background(), []Message{{Role: "user", Content: "test"}}, modelidentity.LegacyQwenModel, "none", nil, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if payload["model"] != modelidentity.Qwen38FNEXL3 || client.modelType != modelidentity.Qwen38FNEXL3 {
		t.Fatal("request still uses legacy identity")
	}
	template, _ := payload["chat_template_kwargs"].(map[string]any)
	if template["enable_thinking"] != false {
		t.Fatal("thinking protocol changed", payload)
	}
}
