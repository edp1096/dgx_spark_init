package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGemmaSamplingDefaults(t *testing.T) {
	for _, family := range []string{"gemma4", "gemma4-vllm", "generic"} {
		t.Run(family, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var p map[string]any
				if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
					t.Fatal(err)
				}
				if family == "generic" {
					if p["temperature"] != 0.7 {
						t.Error("unrelated model default changed")
					}
				} else if p["temperature"] != 1.0 || p["top_p"] != 0.95 || p["top_k"] != float64(64) {
					t.Errorf("incorrect Gemma sampling: %v", p)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
			}))
			defer srv.Close()
			_, err := New(srv.URL, "model", "", family).Stream(context.Background(), []Message{{Role: "user", Content: "hi"}}, "model", "none", nil, func(string, string) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
