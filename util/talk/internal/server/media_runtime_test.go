package server

import (
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/config"
	"testing"
)

func TestMediaRuntimeUsesConfiguredService(t *testing.T) {
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.RequestURI() != "/v1/runtime/yt-dlp?check=1" {
			t.Errorf("unexpected URL: %s", r.URL)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"current":"test"}`))
	}))
	defer upstream.Close()
	s := &Server{cfg: config.Config{Extra: config.ExtraConfig{MediaEndpoint: upstream.URL}}}
	w := httptest.NewRecorder()
	s.mediaRuntime(w, httptest.NewRequest("GET", "/api/media/yt-dlp?check=1", nil))
	if w.Code != 200 || calls != 1 {
		t.Fatalf("response %d %s, calls %d", w.Code, w.Body.String(), calls)
	}
	r := httptest.NewRequest("POST", "http://local/api/media/yt-dlp/update", nil)
	r.Header.Set("Origin", "http://other")
	w = httptest.NewRecorder()
	s.mediaRuntime(w, r)
	if w.Code != 403 || calls != 1 {
		t.Fatalf("cross-origin mutation forwarded: %d", w.Code)
	}
}
