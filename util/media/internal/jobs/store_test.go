package jobs

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDeleteRemovesFinishedJobAndFiles(t *testing.T) {
	dir := t.TempDir()
	store, err := New(dir)
	if err != nil {
		t.Fatal(err)
	}
	id := "finished-job"
	inputDir := filepath.Join(dir, "inputs", id)
	if err = os.MkdirAll(inputDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(inputDir, "input.wav"), []byte("audio"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(store.OutputPath(id+".txt"), []byte("text"), 0o644); err != nil {
		t.Fatal(err)
	}
	job := Job{ID: id, Kind: "recognition", Status: "completed", OutputURL: "/api/outputs/" + id + ".txt", CreatedAt: time.Now()}
	if err = store.Save(job); err != nil {
		t.Fatal(err)
	}

	if err = store.Delete(id); err != nil {
		t.Fatal(err)
	}
	if _, ok := store.Get(id); ok {
		t.Fatal("deleted job remains in store")
	}
	if _, err = os.Stat(inputDir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("input directory still exists: %v", err)
	}
	if _, err = os.Stat(store.OutputPath(id + ".txt")); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("output still exists: %v", err)
	}
}

func TestDeleteRejectsActiveJob(t *testing.T) {
	store, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	job := Job{ID: "active-job", Kind: "image", Status: "running", CreatedAt: time.Now()}
	if err = store.Save(job); err != nil {
		t.Fatal(err)
	}
	if err = store.Delete(job.ID); !errors.Is(err, ErrActive) {
		t.Fatalf("Delete() error = %v, want %v", err, ErrActive)
	}
	if _, ok := store.Get(job.ID); !ok {
		t.Fatal("active job was removed")
	}
}

func TestDeleteFinishedKeepsActiveJobs(t *testing.T) {
	store, err := New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	for _, job := range []Job{
		{ID: "done", Status: "completed", CreatedAt: time.Now()},
		{ID: "failed", Status: "failed", CreatedAt: time.Now()},
		{ID: "queued", Status: "queued", CreatedAt: time.Now()},
		{ID: "running", Status: "running", CreatedAt: time.Now()},
	} {
		if err = store.Save(job); err != nil {
			t.Fatal(err)
		}
	}
	deleted, err := store.DeleteFinished()
	if err != nil {
		t.Fatal(err)
	}
	if deleted != 2 {
		t.Fatalf("deleted = %d, want 2", deleted)
	}
	if got := len(store.List()); got != 2 {
		t.Fatalf("remaining jobs = %d, want 2", got)
	}
	if _, ok := store.Get("queued"); !ok {
		t.Fatal("queued job was deleted")
	}
	if _, ok := store.Get("running"); !ok {
		t.Fatal("running job was deleted")
	}
}
