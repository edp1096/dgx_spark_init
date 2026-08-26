package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestUpscaleEngineBusy(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","busy":true}`))
	}))
	defer worker.Close()

	server := New(config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"upscale": {Endpoint: worker.URL}},
	}, nil, nil)
	job := jobs.Job{Kind: "video", Params: map[string]any{"mode": "upscale"}}
	if !server.generationEngineBusy(job) {
		t.Fatal("expected busy SeedVR2 engine to be detected")
	}
	job.Params["mode"] = "create"
	if server.generationEngineBusy(job) {
		t.Fatal("ordinary video generation must not query the upscale lock")
	}
}
