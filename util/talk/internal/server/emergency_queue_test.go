package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestEmergencyQueueEngineAPIs(t *testing.T) {
	for _, engine := range []string{"sglang", "vllm", "unsupported", "unauthorized"} {
		t.Run(engine, func(t *testing.T) {
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if r.Header.Get("Authorization") != "Bearer test" {
					t.Error("missing auth")
				}
				if r.Method != "POST" {
					t.Error("wrong method")
				}
				b, _ := io.ReadAll(r.Body)
				if engine == "unauthorized" {
					w.WriteHeader(401)
					return
				}
				if engine == "sglang" && r.URL.Path == "/abort_request" {
					if !strings.Contains(string(b), `"abort_all":true`) {
						t.Error("not abort all")
					}
					return
				}
				if engine == "vllm" && r.URL.Path == "/abort_requests" {
					if string(b) != `{"request_ids":[]}` {
						t.Error("bad vllm body")
					}
					return
				}
				w.WriteHeader(404)
			}))
			defer srv.Close()
			err := clearInferenceQueue(context.Background(), srv.URL+"/v1", "test")
			if (err != nil) != (engine == "unsupported" || engine == "unauthorized") {
				t.Fatal(err)
			}
			if engine == "unauthorized" && calls != 1 {
				t.Fatal("must not fallback on auth failure")
			}
		})
	}
}
func TestEmergencyQueueGenerationTracking(t *testing.T) {
	s := &Server{}
	ctx, done, err := s.trackGeneration(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	s.generationMu.Lock()
	s.queueClearing = true
	for _, cancel := range s.generations {
		cancel()
	}
	s.generationMu.Unlock()
	if ctx.Err() == nil {
		t.Fatal("not cancelled")
	}
	if _, _, err = s.trackGeneration(context.Background()); err == nil {
		t.Fatal("accepted during cleanup")
	}
	done()
	if len(s.generations) != 0 {
		t.Fatal("leaked registration")
	}
}
