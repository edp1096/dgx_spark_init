package orchestrator

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestAlreadyHealthySetStartHasNoNewAllocation(t *testing.T) {
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	defer api.Close()
	dir := t.TempDir()
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\ncase \"$*\" in *inspect*) printf running;; esac\n"), 0700); err != nil {
		t.Fatal(err)
	}
	catalog, _ := LoadCatalog()
	for i := range catalog.Components {
		catalog.Components[i].HealthURL = api.URL + "/health"
	}
	c, err := NewControllerWithCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := c.Catalog().Bundle("flash-next")
	plan := c.bundleMemoryPlan(context.Background(), b)
	if plan.NeededGiB != 0 || plan.RequiresCUDAStart {
		t.Fatalf("healthy set must be a no-op: %+v", plan)
	}
}
