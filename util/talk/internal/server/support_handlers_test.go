package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/config"
	"sync"
	"testing"
)

func TestSupportHealthSeparatesPermissionAndAvailability(t *testing.T) {
	calls := map[string]bool{}
	var mu sync.Mutex
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls[r.URL.Path] = true
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"ok"}`))
	}))
	defer api.Close()
	cfg := config.Config{Extra: config.ExtraConfig{MediaEndpoint: api.URL + "/media", DocumentsEndpoint: api.URL + "/documents", SSHEndpoint: api.URL + "/ssh", CollectorEndpoint: api.URL + "/collector"}, Tools: config.ToolsConfig{MediaImportEnabled: true}}
	got := supportHealth(context.Background(), cfg)
	for _, key := range []string{"media", "documents", "ssh", "collector"} {
		state := got[key].(map[string]any)
		if state["status"] != "ok" || state["enabled"] != (key == "media") {
			t.Fatalf("%s: %#v", key, state)
		}
		if !calls["/"+key+"/health"] {
			t.Fatal("missing health", key)
		}
	}
}
