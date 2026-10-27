package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
)

func TestModelAPISwitchAndPayload(t *testing.T) {
	var seen []string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		seen = append(seen, string(body["model"]))
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer upstream-secret" || r.Header.Get("Cookie") != "" {
			t.Errorf("unexpected request: %s %v", r.URL.Path, r.Header)
		}
		if string(body["tools"]) != `[{"type":"function"}]` || string(body["seed"]) != `9007199254740993` {
			t.Errorf("payload changed: %s", body)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[]}`)
	}))
	defer upstream.Close()
	s := &Server{cfg: config.Config{Model: config.ModelConfig{Endpoint: upstream.URL + "/v1", DefaultModel: "first", APIKey: "upstream-secret"}}}
	for _, model := range []string{"first", "second"} {
		s.mu.Lock()
		s.cfg.Model.DefaultModel = model
		s.mu.Unlock()
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"sparktalk","tools":[{"type":"function"}],"seed":9007199254740993}`))
		req.Header.Set("Authorization", "Bearer caller-secret")
		req.Header.Set("Cookie", "private=1")
		w := httptest.NewRecorder()
		s.modelAPI(w, req)
		if w.Code != 200 {
			t.Fatalf("%d %s", w.Code, w.Body.String())
		}
	}
	if strings.Join(seen, ",") != `"first","second"` {
		t.Fatal(seen)
	}
}

func TestModelAPIStreamingCancellation(t *testing.T) {
	cancelled := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		<-r.Context().Done()
		close(cancelled)
	}))
	defer upstream.Close()
	s := &Server{cfg: config.Config{Model: config.ModelConfig{Endpoint: upstream.URL, DefaultModel: "first"}}}
	gateway := httptest.NewServer(http.HandlerFunc(s.modelAPI))
	defer gateway.Close()
	response, err := http.Post(gateway.URL+"/v1/chat/completions", "application/json", strings.NewReader(`{"model":"sparktalk","stream":true}`))
	if err != nil {
		t.Fatal(err)
	}
	b := make([]byte, len("data: first\n\n"))
	if _, err = io.ReadFull(response.Body, b); err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	select {
	case <-cancelled:
	case <-time.After(3 * time.Second):
		t.Fatal("upstream not cancelled")
	}
}

func TestModelAPIErrorsAndRoutes(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(429)
		fmt.Fprint(w, `{"error":{"message":"busy"}}`)
	}))
	defer upstream.Close()
	s := &Server{cfg: config.Config{Model: config.ModelConfig{Endpoint: upstream.URL, DefaultModel: "first"}}}
	for _, tc := range []struct {
		method, path, body string
		code               int
	}{
		{"GET", "/v1/models", "", 200}, {"POST", "/v1/models", "{}", 405},
		{"POST", "/v1/chat/completions", "null", 400}, {"POST", "/v1/chat/completions", `{"model":"stale"}`, 400},
		{"POST", "/v1/responses", `{"model":"sparktalk"}`, 429}, {"GET", "/v1/admin", "", 404},
	} {
		w := httptest.NewRecorder()
		s.modelAPI(w, httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body)))
		if w.Code != tc.code {
			t.Fatalf("%s: %d %s", tc.path, w.Code, w.Body.String())
		}
	}
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{}`))
	req.Header.Set("X-SparkTalk-Proxy", "1")
	w := httptest.NewRecorder()
	s.modelAPI(w, req)
	if w.Code != 508 {
		t.Fatal(w.Code)
	}
}

func TestModelAPIEndpointSwitchAndUnavailable(t *testing.T) {
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "first") }))
	defer first.Close()
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "second") }))
	s := &Server{cfg: config.Config{Model: config.ModelConfig{Endpoint: first.URL, DefaultModel: "first"}}}
	call := func() *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		s.modelAPI(w, httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"sparktalk"}`)))
		return w
	}
	if got := call(); got.Body.String() != "first" {
		t.Fatal(got.Body.String())
	}
	s.mu.Lock()
	s.cfg.Model.Endpoint = second.URL
	s.cfg.Model.DefaultModel = "second"
	s.mu.Unlock()
	if got := call(); got.Body.String() != "second" {
		t.Fatal(got.Body.String())
	}
	second.Close()
	if got := call(); got.Code != 502 || !strings.Contains(got.Body.String(), `"error"`) {
		t.Fatal(got.Code, got.Body.String())
	}
}
