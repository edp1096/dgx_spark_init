package orchestrator

import (
	"context"
	"math"
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

func TestEXL3ColdStartBesideHealthyResidentImage(t *testing.T) {
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	defer api.Close()
	dir := t.TempDir()
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	docker := "#!/bin/sh\ncase \"$*\" in\n *inspect*sparktalk-qwim-mmh3*) printf running;;\n *inspect*) printf exited;;\n *top*sparktalk-qwim-mmh3*) printf 'PID\\n999999999\\n';;\n esac\n"
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(docker), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "nvidia-smi"), []byte("#!/bin/sh\nprintf '999999999, 17920\\n'\n"), 0700); err != nil {
		t.Fatal(err)
	}
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for i := range catalog.Components {
		if catalog.Components[i].ID == "qwim-mmh3" {
			catalog.Components[i].HealthURL = api.URL
		}
	}
	c, err := NewControllerWithCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := c.Catalog().Bundle("qwen38fn_exl3")
	plan := c.bundleMemoryPlan(context.Background(), b)
	if math.Abs(plan.NeededGiB-92) > .001 || plan.FreedGiB != 0 || !plan.RequiresCUDAStart {
		t.Fatalf("expected LLM85 + ASR3.5 + TTS3 + image0.5, got %+v", plan)
	}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 100, FreeGiB: 71}, plan, 1.5); err != nil {
		t.Fatalf("reported incident must now pass startup admission: %v", err)
	}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 93, FreeGiB: 71}, plan, 1.5); err == nil {
		t.Fatal("actual startup shortage must still be rejected")
	}
}
