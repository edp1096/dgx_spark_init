package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func workloadTestController(t *testing.T) *Controller {
	t.Helper()
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"status":"ok","core_ready":true,"busy":false,"active":0}`)
	}))
	t.Cleanup(api.Close)
	dir := t.TempDir()
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	t.Setenv("WORKLOAD_TEST_STATE", filepath.Join(dir, "state"))
	script := `#!/bin/sh
case "$*" in
 *State.Pid*) if test -f "$WORKLOAD_TEST_STATE"; then printf "running|0"; else printf "exited|0"; fi;;
 *State.Status*) if test -f "$WORKLOAD_TEST_STATE"; then printf running; else printf exited; fi;;
 stop*) rm -f "$WORKLOAD_TEST_STATE";;
 *"up -d"*) touch "$WORKLOAD_TEST_STATE";;
esac
`
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}

	cat, _ := LoadCatalog()
	for i := range cat.Components {
		cat.Components[i].HealthURL = api.URL
		cat.Components[i].Endpoint = api.URL
		if workloadGroup(cat.Components[i].ID) != "" {
			cat.Components[i].MemoryGiB = 0.001
		}
	}
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "flash-next" {
			cat.Bundles[i].Components = []string{"flash-next", "flux2", "nemotron-asr", "extra-media", "extra-ssh", "extra-collector"}
		}
	}
	c, err := NewControllerWithCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	c.ConfigurePaths(filepath.Join(dir, "runtime"), filepath.Join(dir, "models"))
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 120, FreeGiB: 110} }
	return c
}

func TestWorkloadQueueCancellationAndRuntimeExclusion(t *testing.T) {
	c := workloadTestController(t)
	release, err := c.AcquireWorkload(context.Background(), "flash-next", "flux2", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.StopBundle("flash-next"); err == nil {
		t.Fatal("manual stop interrupted active work")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if _, err := c.AcquireWorkload(ctx, "flash-next", "nemotron-asr", 1.5); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("wait cancellation: %v", err)
	}
	if c.Operation().ComponentID != "flux2" {
		t.Fatal("canceled waiter changed active job")
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	} // idempotent, does not consume next lease
	next, err := c.AcquireWorkload(context.Background(), "flash-next", "nemotron-asr", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	if err := next(); err != nil {
		t.Fatal(err)
	}
}

func TestWorkloadBudgetUsesPeakNotASRStartup(t *testing.T) {
	cat, _ := LoadCatalog()
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "flash-next" {
			cat.Bundles[i].Components = []string{"flash-next", "flux2", "nemotron-asr", "extra-media", "extra-ssh", "extra-collector"}
		}
	}
	c, err := NewControllerWithCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := c.Catalog().Bundle("flash-next")
	if c.workloadBudget(b) != 10 || b.MemoryGiB != 107 {
		t.Fatalf("budget=%v bundle=%v", c.workloadBudget(b), b.MemoryGiB)
	}
	for i := range cat.Components {
		if cat.Components[i].ID == "nemotron-asr" {
			cat.Components[i].MemoryGiB = 20
			cat.Components[i].StartupMemoryGiB = 3.5
		}
	}
	c, err = NewControllerWithCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	b, _ = c.Catalog().Bundle("flash-next")
	if c.workloadBudget(b) <= 20 {
		t.Fatal("media or ASR peak omitted")
	}
}

func TestWorkloadCanceledBeforeAcquisitionDoesNotMutate(t *testing.T) {
	c := workloadTestController(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.AcquireWorkload(ctx, "flash-next", "flux2", 1.5); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if c.Operation().State != "" {
		t.Fatal("canceled request started operation")
	}
}

func TestWorkloadStopsMediaEvenWithFilteredStartupBundle(t *testing.T) {
	c := workloadTestController(t)
	b, _ := c.Catalog().Bundle("flash-next")
	// Only Media is alive; the real startup list excludes support services.
	dir := t.TempDir()
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	marker := filepath.Join(dir, "media")
	t.Setenv("WORKLOAD_MEDIA_MARKER", marker)
	os.WriteFile(marker, []byte("running"), 0600)
	script := `#!/bin/sh
case "$*" in
 *sparktalk-extra-media*)
 case "$*" in
 stop*) rm -f "$WORKLOAD_MEDIA_MARKER";;
 *State.Pid*) if test -f "$WORKLOAD_MEDIA_MARKER"; then printf 'running|0'; else printf 'exited|0'; fi;;
 *State.Status*) if test -f "$WORKLOAD_MEDIA_MARKER"; then printf running; else printf exited; fi;;
 esac;;
 *State.Status*) printf exited;;
esac
`
	os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700)
	if err := c.stopWorkloads(context.Background(), c.Catalog().StartBundleMembers(b)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("Media survived startup cleanup")
	}
}

func TestWorkloadStopFailurePreventsNewJob(t *testing.T) {
	c := workloadTestController(t)
	cat := c.Catalog()
	for i := range cat.Components {
		if workloadGroup(cat.Components[i].ID) != "" {
			cat.Components[i].MemoryGiB = 1000000
		}
	}
	var setupErr error
	c, setupErr = NewControllerWithCatalog(cat)
	if setupErr != nil {
		t.Fatal(setupErr)
	}
	dir := t.TempDir()
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	script := `#!/bin/sh
case "$*" in
 *State.Pid*) printf 'running|0';;
 *State.Status*) printf running;;
 stop*) echo stop-failed; exit 1;;
 *"up -d"*) echo 'UNEXPECTED START'; exit 1;;
esac
`
	os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700)
	release, err := c.AcquireWorkload(context.Background(), "flash-next", "nemotron-asr", 1.5)
	if err == nil || release != nil || c.Operation().State != "failed" {
		t.Fatalf("unreleased memory admitted: %v", err)
	}
	// Failure must release the queue, but never admit another job while stop fails.
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if _, err = c.AcquireWorkload(ctx, "flash-next", "flux2", 1.5); err == nil || errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("failure stranded queue: %v", err)
	}
}

func TestWorkloadDoesNotEvictWhenHeadroomSuffices(t *testing.T) {
	c := workloadTestController(t)
	// Simulate resident idle auxiliaries with enough free memory. Acquire must
	// keep them across completion and the next request.
	marker := os.Getenv("WORKLOAD_TEST_STATE")
	if err := os.WriteFile(marker, []byte("running"), 0600); err != nil {
		t.Fatal(err)
	}
	release, err := c.AcquireWorkload(context.Background(), "flash-next", "flux2", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatal("idle services evicted despite sufficient headroom")
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatal("completed job was discarded despite sufficient headroom")
	}
}

func TestWorkloadReleasesOnLowHeadroomOrCancellation(t *testing.T) {
	for _, canceled := range []bool{false, true} {
		t.Run(fmt.Sprint(canceled), func(t *testing.T) {
			c := workloadTestController(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			release, err := c.AcquireWorkload(ctx, "flash-next", "flux2", 1.5)
			if err != nil {
				t.Fatal(err)
			}
			if canceled {
				cancel()
			} else {
				c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 1, FreeGiB: 1} }
			}
			if err := release(); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(os.Getenv("WORKLOAD_TEST_STATE")); err != nil {
				t.Fatal("Klein core was stopped instead of reclaiming modules")
			}
		})
	}
}

func TestASRDecodedRequestBudgetReplacesGenericSixGiB(t *testing.T) {
	c := workloadTestController(t)
	for i := range c.catalog.Components {
		if c.catalog.Components[i].ID == "nemotron-asr" {
			c.catalog.Components[i].MemoryGiB = 6
		}
	}
	var err error
	c.catalog, err = ValidateCatalog(c.catalog)
	if err != nil {
		t.Fatal(err)
	}
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 5.9, FreeGiB: 5} }
	release, err := c.AcquireWorkload(context.Background(), "flash-next", "nemotron-asr", 1.5, 4.25)
	if err != nil {
		t.Fatalf("short request incorrectly blocked by generic budget: %v", err)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
	component, _ := c.Catalog().ResolveComponent("flash-next", "nemotron-asr")
	if component.MemoryGiB != 6 {
		t.Fatal("request changed shared catalog")
	}
}

func TestImageRequestWorkspaceCannotBeReplacedBySmallerLiveBudget(t *testing.T) {
	c := workloadTestController(t)
	// Simulate resident image weights with a small default allocation. The
	// requested BFS workspace still has to fit on its own before inference.
	for i := range c.catalog.Components {
		if c.catalog.Components[i].ID == "flux2" {
			c.catalog.Components[i].MemoryGiB = .25
			c.catalog.Components[i].WorkspaceMemoryGiB = .25
		}
	}
	if err := os.WriteFile(os.Getenv("WORKLOAD_TEST_STATE"), []byte("running"), 0600); err != nil {
		t.Fatal(err)
	}
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 9, FreeGiB: 8} }
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if release, err := c.AcquireWorkload(ctx, "flash-next", "flux2", 1.5, 8.0); err == nil {
		_ = release()
		t.Fatal("BFS workspace admitted with less than the required reserve")
	}
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 10, FreeGiB: 9} }
	release, err := c.AcquireWorkload(context.Background(), "flash-next", "flux2", 1.5, 8.0)
	if err != nil {
		t.Fatal(err)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
}

func TestFluxWorkspaceTracksTextWeightResidency(t *testing.T) {
	for _, tc := range []struct {
		body             string
		configured, want float64
	}{{`{"workspace_gib":2.5}`, 4.5, 2.5}, {`{"workspace_gib":4.5}`, 4.5, 6.5}, {`{"workspace_gib":0}`, 4.5, 6.5}, {`{"workspace_gib":2.5}`, 6, 6},
		{`{"memory_schema":1,"workspace_kind":"additional","workspace_gib":3.25}`, 4.5, 3.25},
		{`{"memory_schema":1,"workspace_kind":"additional","workspace_gib":7}`, 6, 7},
		{`{"memory_schema":1,"workspace_kind":"total","workspace_gib":3.25}`, 4.5, 6.5},
		{`{"memory_schema":2,"workspace_kind":"additional","workspace_gib":2.5}`, 4.5, 6.5},
		{`{"workspace_gib":3.25}`, 4.5, 6.5}} {
		api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/runtime/memory" {
				t.Errorf("wrong path %s", r.URL.Path)
			}
			fmt.Fprint(w, tc.body)
		}))
		c := Component{ID: "flux2", ComposeAsset: "compose.flux2.yaml", Endpoint: api.URL, WorkspaceMemoryGiB: tc.configured}
		if got := liveWorkspaceMemory(context.Background(), c); got != tc.want {
			t.Fatalf("workspace=%v want=%v", got, tc.want)
		}
		api.Close()
	}
}

func TestColdLoRAWorkspaceIncludesTextReload(t *testing.T) {
	for _, tc := range []struct{ reload, total, want float64 }{{4, 6.5, 8.5}, {0, 2.5, 4.5}} {
		api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, `{"memory_schema":1,"workspace_kind":"additional","workspace_gib":%v,"text_reload_gib":%v,"generation_workspace_gib":2.5}`, tc.total, tc.reload)
		}))
		c := Component{ID: "flux2", ComposeAsset: "compose.flux2.yaml", Endpoint: api.URL, WorkspaceMemoryGiB: 6.5}
		if got := liveWorkspaceMemory(context.Background(), c, 4.5); got != tc.want {
			t.Fatalf("got %v want %v", got, tc.want)
		}
		api.Close()
	}
}

func TestOnDemandBundleStartupDoesNotReserveInactiveWorkloads(t *testing.T) {
	c := workloadTestController(t)
	b, _ := c.Catalog().Bundle("flash-next")
	core, _ := c.Catalog().ResolveComponent(b.ID, "flash-next")
	plan := c.bundleMemoryPlan(context.Background(), b)
	if plan.NeededGiB != core.startupMemoryGiB() {
		t.Fatalf("inactive auxiliaries blocked core startup: got %.3f core %.3f", plan.NeededGiB, core.startupMemoryGiB())
	}
}

func TestColdWorkloadUsesConfiguredImmediateReserve(t *testing.T) {
	c := workloadTestController(t)
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 8, FreeGiB: 3.8} }
	release, err := c.AcquireWorkload(context.Background(), "flash-next", "flux2", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
}
