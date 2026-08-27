package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestGenerationEngineBusy(t *testing.T) {
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
	server.cfg.Engines["video"] = config.Engine{Endpoint: worker.URL}
	if !server.generationEngineBusy(job) {
		t.Fatal("expected busy LTX engine to be detected")
	}
	server.cfg.Image = config.Image{
		DefaultMode: "create",
		Backends: map[string]config.ImageBackend{
			"create": {Endpoint: worker.URL},
		},
	}
	imageJob := jobs.Job{Kind: "image", Params: map[string]any{"mode": "create"}}
	if !server.generationEngineBusy(imageJob) {
		t.Fatal("expected busy Krea engine to be detected")
	}
	imageJob.Params["mode"] = "upscale"
	if !server.generationEngineBusy(imageJob) {
		t.Fatal("expected busy SeedVR2 image upscale to be detected")
	}
}

func TestGenerationEngineBusyConflict(t *testing.T) {
	busy := &engineHTTPError{StatusCode: http.StatusConflict, Body: `{"detail":"another video generation is running"}`}
	if !isGenerationEngineBusyConflict(busy) {
		t.Fatal("expected LTX busy conflict to be recognized")
	}
	if isGenerationEngineBusyConflict(&engineHTTPError{StatusCode: http.StatusBadRequest, Body: "bad input"}) {
		t.Fatal("ordinary engine errors must remain failures")
	}
	imageBusy := &engineHTTPError{StatusCode: http.StatusConflict, Body: `{"detail":"image engine is busy"}`}
	if !isGenerationEngineBusyConflict(imageBusy) {
		t.Fatal("expected image busy conflict to be recognized")
	}
}
