package server

import (
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"testing"
	"time"
)

func TestJobTransitionsPreserveLifecycleContract(t *testing.T) {
	startedAt := time.Date(2026, time.August, 27, 10, 20, 30, 40, time.Local)
	job := jobs.Job{
		ID:     "transition-contract",
		Status: "queued",
		Error:  "old error",
		Params: map[string]any{"media_eta_seconds": float64(12)},
	}

	transitionJobRunning(&job, "generation", startedAt)
	if job.Status != "running" || job.Error != "" {
		t.Fatalf("running transition = status %q, error %q", job.Status, job.Error)
	}
	if got := job.Params["stage"]; got != "generation" {
		t.Fatalf("running stage = %#v", got)
	}
	if got := job.Params["started_at"]; got != startedAt.Format(time.RFC3339Nano) {
		t.Fatalf("started_at = %#v", got)
	}

	transitionJobFailed(&job, "engine failed")
	if job.Status != "failed" || job.Error != "engine failed" {
		t.Fatalf("failed transition = status %q, error %q", job.Status, job.Error)
	}
	if got := job.Params["stage"]; got != "generation" {
		t.Fatalf("failed transition unexpectedly changed stage to %#v", got)
	}

	transitionJobCancelled(&job)
	if job.Status != "cancelled" || job.Error != "" || job.Params["stage"] != "cancelled" {
		t.Fatalf("cancel transition = %#v", job)
	}
	if _, ok := job.Params["media_eta_seconds"]; ok {
		t.Fatal("cancel transition retained media_eta_seconds")
	}

	transitionJobCompleted(&job, "/api/outputs/result.png")
	if job.Status != "completed" || job.Error != "" || job.OutputURL != "/api/outputs/result.png" {
		t.Fatalf("completed transition = %#v", job)
	}
	if got := job.Params["stage"]; got != "cancelled" {
		t.Fatalf("completed transition unexpectedly changed stage to %#v", got)
	}
}

func TestCompleteGenerationJobDoesNotPublishAfterCancellation(t *testing.T) {
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	server := New(config.Config{DataDir: dataDir}, store, nil)
	job := jobs.Job{ID: "cancel-race", Kind: "image", Status: "cancelled"}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}

	cleaned := false
	published, err := server.completeGenerationJob(&job, "/api/outputs/cancel-race.png", func() {
		cleaned = true
	})
	if err != nil {
		t.Fatal(err)
	}
	if published {
		t.Fatal("cancelled generation was published")
	}
	if !cleaned {
		t.Fatal("cancelled generation output was not cleaned")
	}
	stored, ok := store.Get(job.ID)
	if !ok || stored.Status != "cancelled" || stored.OutputURL != "" {
		t.Fatalf("stored job after cancellation race = %#v, found=%v", stored, ok)
	}
}

func TestCompleteGenerationJobPublishesOutput(t *testing.T) {
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	server := New(config.Config{DataDir: dataDir}, store, nil)
	job := jobs.Job{ID: "publish", Kind: "image", Status: "running", Error: "stale"}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}

	published, err := server.completeGenerationJob(&job, "/api/outputs/publish.png", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !published {
		t.Fatal("running generation was not published")
	}
	stored, ok := store.Get(job.ID)
	if !ok || stored.Status != "completed" || stored.Error != "" || stored.OutputURL != "/api/outputs/publish.png" {
		t.Fatalf("stored completed job = %#v, found=%v", stored, ok)
	}
}
