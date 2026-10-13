package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestStructuredOutputDoesNotChangeOrdinaryChat(t *testing.T) {
	var requests []map[string]any
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var v map[string]any
		json.NewDecoder(r.Body).Decode(&v)
		requests = append(requests, v)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"{}\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer backend.Close()
	c := New(backend.URL, "m", "")
	ctx := WithJSONSchema(context.Background(), "test", json.RawMessage(`{"type":"object"}`))
	for _, v := range []context.Context{ctx, context.Background()} {
		if _, e := c.Stream(v, []Message{{Role: "user", Content: "test"}}, "m", "off", nil, func(string, string) error { return nil }); e != nil {
			t.Fatal(e)
		}
	}
	if requests[0]["response_format"] == nil || requests[1]["response_format"] != nil {
		t.Fatal("schema leaked into ordinary chat")
	}
	if _, e := c.Stream(ctx, nil, "m", "off", []Tool{{Type: "function"}}, func(string, string) error { return nil }); e == nil {
		t.Fatal("mixed tool/schema modes accepted")
	}
}
