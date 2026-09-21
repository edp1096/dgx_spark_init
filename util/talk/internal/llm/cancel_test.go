package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestNativeAbortBeforeDisconnectAndOnlyOwnRequest(t *testing.T) {
	ready, aborted := make(chan struct{}), make(chan struct{})
	var premature atomic.Bool
	var rid string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/server_info":
			fmt.Fprint(w, `{"max_total_num_tokens":100,"mamba_radix_cache_strategy":"auto"}`)
		case "/abort_request":
			var b struct {
				RID string `json:"rid"`
				All bool   `json:"abort_all"`
			}
			json.NewDecoder(r.Body).Decode(&b)
			if b.All || b.RID != rid || !strings.HasPrefix(b.RID, "talk-") {
				t.Errorf("unsafe abort: %+v", b)
			}
			close(aborted)
			w.WriteHeader(200)
		case "/v1/chat/completions":
			var b map[string]any
			json.NewDecoder(r.Body).Decode(&b)
			rid, _ = b["rid"].(string)
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n")
			w.(http.Flusher).Flush()
			close(ready)
			select {
			case <-aborted:
			case <-r.Context().Done():
				premature.Store(true)
			}
			<-r.Context().Done()
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	client := New(server.URL, "test", "").WithBackendCancellation(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		client.Stream(ctx, []Message{{Role: "user", Content: "test"}}, "test", "none", nil, func(string, string) error { return nil })
	}()
	<-ready
	cancel()
	<-done
	if premature.Load() {
		t.Fatal("stream disconnected before abort")
	}
}

func TestGenericBackendDoesNotReceiveNativeFields(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/server_info" {
			http.NotFound(w, r)
			return
		}
		var b map[string]any
		json.NewDecoder(r.Body).Decode(&b)
		if b["rid"] != nil {
			t.Error("native field leaked to generic provider")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer backend.Close()
	c := New(backend.URL, "test", "").WithBackendCancellation(context.Background())
	r, err := c.Stream(context.Background(), []Message{{Role: "user", Content: "test"}}, "test", "none", nil, func(string, string) error { return nil })
	if err != nil || r.Content != "ok" {
		t.Fatal(r, err)
	}
}
