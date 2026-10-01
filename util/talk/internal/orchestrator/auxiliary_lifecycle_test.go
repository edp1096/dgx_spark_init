package orchestrator

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

type auxiliaryTransport func(*http.Request) (*http.Response, error)

func (f auxiliaryTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// Docker is simulated by per-container state files. HTTP uses a RoundTripper;
// these lifecycle tests work without a Docker daemon or any listening socket.
func offlineAuxiliaryController(t *testing.T) (*Controller, string, *[]string) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	t.Setenv("AUX_TEST_ROOT", dir)
	script := `#!/bin/sh
last=""
file=""
prev=""
for arg in "$@"; do
 if test "$prev" = "-f"; then file="$arg"; fi
 prev="$arg"; last="$arg"
done
case "$*" in
 *" config") cat;;
 *State.Pid*) if test -f "$AUX_TEST_ROOT/$last"; then printf 'running|0'; else printf 'exited|0'; fi;;
 *State.Status*) if test -f "$AUX_TEST_ROOT/$last"; then printf running; else printf exited; fi;;
 stop*) echo "stop $last" >> "$AUX_TEST_ROOT/actions"; rm -f "$AUX_TEST_ROOT/$last";;
 *"up -d"*) name=$(awk '/container_name:/ {print $2; exit}' "$file"); touch "$AUX_TEST_ROOT/$name"; echo "start $name" >> "$AUX_TEST_ROOT/actions";;
esac
`
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for i := range cat.Components {
		x := &cat.Components[i]
		x.AutoAddress = false
		x.Endpoint = "http://" + x.ID + ".test"
		x.HealthURL = x.Endpoint + "/health"
	}
	c, err := NewControllerWithCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	c.ConfigurePaths(filepath.Join(dir, "runtime"), filepath.Join(dir, "models"))
	actions := []string{}
	transport := auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
		// A stopped fake container must not look healthy before startup.
		if r.URL.Path == "/health" {
			id := strings.TrimSuffix(r.URL.Host, ".test")
			x, _ := c.Catalog().Component(id)
			if _, e := os.Stat(filepath.Join(dir, x.Container)); e != nil {
				return nil, errors.New("not running")
			}
		}
		actions = append(actions, r.Method+" "+r.URL.Path)
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"status":"ok","core_ready":true,"busy":false,"active":0,"workspace_gib":4.5}`))}, nil
	})
	c.client.Transport = transport
	previous := http.DefaultClient
	http.DefaultClient = &http.Client{Transport: transport}
	t.Cleanup(func() { http.DefaultClient = previous })
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 120, FreeGiB: 110} }
	return c, dir, &actions
}

func seedAuxiliary(t *testing.T, dir string, names ...string) {
	t.Helper()
	for _, name := range names {
		if e := os.WriteFile(filepath.Join(dir, name), []byte("running"), 0600); e != nil {
			t.Fatal(e)
		}
	}
}

func TestAuxiliaryReclaimRequiresMeasuredHeadroom(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	seedAuxiliary(t, dir, "sglang-qwen38-fn", "flux2-klein-nvfp4-api", "sparktalk-nemotron-asr", "sparktalk-extra-documents")
	probes := 0
	c.memoryProbe = func() SystemMemory {
		probes++
		// Stop can succeed before memory returns. Do not credit a catalog value.
		return SystemMemory{AvailableGiB: 3, FreeGiB: 2}
	}
	if release, err := c.AcquireWorkload(context.Background(), "flash-next", "flux2", 1.5); err == nil || release != nil {
		t.Fatal("request admitted although measured free memory never recovered")
	}
	if probes < 2 {
		t.Fatal("memory was not remeasured after reclaim")
	}
	for _, name := range []string{"sglang-qwen38-fn", "flux2-klein-nvfp4-api"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatal("protected core was stopped", name)
		}
	}
}

func TestDocumentWorkDoesNotUseIdleRSSAsPeak(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	seedAuxiliary(t, dir, "sglang-qwen38-fn")
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 2, FreeGiB: 2} }
	if release, err := c.AcquireWorkload(context.Background(), "flash-next", "extra-documents", 1.5); err == nil || release != nil {
		t.Fatal("document work admitted using its 0.4 GiB idle budget")
	}
	if _, err := os.Stat(filepath.Join(dir, "sparktalk-extra-documents")); !os.IsNotExist(err) {
		t.Fatal("document worker started before admission")
	}
	for id, want := range map[string]float64{"extra-documents": 1, "extra-media": 2, "extra-collector": 2} {
		x, _ := c.Catalog().Component(id)
		got, err := supportRequestMemoryGiB(x)
		if err != nil || got != want {
			t.Fatalf("%s: %v %v", id, got, err)
		}
	}
}

func TestAuxiliarySharedDocumentsLeaseProtectsActiveWorkAndReusesService(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	seedAuxiliary(t, dir, "sglang-qwen38-fn", "flux2-klein-nvfp4-api")
	lease, err := c.AcquireWorkload(context.Background(), "flash-next", "extra-documents", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.ComponentAction("extra-documents", "stop", "flash-next"); err == nil {
		t.Fatal("active document job was interrupted")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := c.AcquireWorkload(ctx, "flash-next", "extra-collector", 1.5); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("waiter: %v", err)
	}
	if err = lease(); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(filepath.Join(dir, "sparktalk-extra-documents")); err != nil {
		t.Fatal("service not retained for reuse")
	}
	if err = lease(); err != nil {
		t.Fatal("release not idempotent", err)
	}
}

func TestAuxiliaryPressureReclaimsExtraBeforeCore(t *testing.T) {
	c, dir, actions := offlineAuxiliaryController(t)
	seedAuxiliary(t, dir, "sglang-qwen38-fn", "flux2-klein-nvfp4-api", "sparktalk-extra-collector")
	c.memoryProbe = func() SystemMemory {
		available := 4.0
		if _, err := os.Stat(filepath.Join(dir, "sparktalk-extra-collector")); os.IsNotExist(err) {
			available = 9
		}
		return SystemMemory{AvailableGiB: available, FreeGiB: available}
	}
	lease, err := c.AcquireWorkload(context.Background(), "flash-next", "nemotron-asr", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	if err = lease(); err != nil {
		t.Fatal(err)
	}
	for _, a := range *actions {
		if strings.Contains(a, "/reclaim") {
			t.Fatal("core modules reclaimed before enough idle Extra memory")
		}
	}
	for _, name := range []string{"sglang-qwen38-fn", "flux2-klein-nvfp4-api"} {
		if _, err = os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatal("protected model stopped", name)
		}
	}
}

func TestAuxiliaryInsufficientMemoryPreservesCoreAndReleasesQueue(t *testing.T) {
	c, dir, actions := offlineAuxiliaryController(t)
	seedAuxiliary(t, dir, "sglang-qwen38-fn", "flux2-klein-nvfp4-api")
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 2, FreeGiB: 2} }
	if release, err := c.AcquireWorkload(context.Background(), "flash-next", "nemotron-asr", 1.5); err == nil || release != nil {
		t.Fatal("oversized workload admitted")
	}
	reclaims := 0
	for _, a := range *actions {
		if a == "POST /v1/runtime/reclaim" {
			reclaims++
		}
		if a == "POST /v1/runtime/cancel" {
			t.Fatal("admission failure canceled an unowned image job")
		}
	}
	if reclaims != 1 {
		t.Fatalf("module reclamation count %d", reclaims)
	}
	for _, name := range []string{"sglang-qwen38-fn", "flux2-klein-nvfp4-api"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatal("protected model stopped", name)
		}
	}
	// The next request has enough room for Chromium's bounded workload, so
	// a failure here would indicate a stranded lease rather than admission.
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 4, FreeGiB: 4} }
	lease, err := c.AcquireWorkload(context.Background(), "flash-next", "extra-collector", 1.5)
	if err != nil {
		t.Fatal("queue stranded", err)
	}
	if err = lease(); err != nil {
		t.Fatal(err)
	}
}

func TestAuxiliaryCanceledImageRetainsCore(t *testing.T) {
	c, dir, actions := offlineAuxiliaryController(t)
	seedAuxiliary(t, dir, "sglang-qwen38-fn", "flux2-klein-nvfp4-api")
	ctx, cancel := context.WithCancel(context.Background())
	lease, err := c.AcquireWorkload(ctx, "flash-next", "flux2", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	cancel()
	if err = lease(); err != nil {
		t.Fatal(err)
	}
	want := []string{"POST /v1/runtime/cancel", "POST /v1/runtime/reclaim"}
	joined := strings.Join(*actions, "\n")
	for _, a := range want {
		if !strings.Contains(joined, a) {
			t.Fatal("missing action", a)
		}
	}
	if _, err = os.Stat(filepath.Join(dir, "flux2-klein-nvfp4-api")); err != nil {
		t.Fatal("core lost on cancellation")
	}
}

func TestAuxiliaryBusySharedServiceIsNotReclaimed(t *testing.T) {
	c, _, _ := offlineAuxiliaryController(t)
	x, _ := c.Catalog().Component("extra-documents")
	c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"status":"ok","busy":true}`))}, nil
	})
	idle, err := c.supportIdle(context.Background(), x)
	if err != nil || idle {
		t.Fatalf("busy: %v %v", idle, err)
	}
	c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})
	if idle, err = c.supportIdle(context.Background(), x); err == nil || idle {
		t.Fatal("unknown state credited as idle")
	}
}

func TestAuxiliaryImageWorkspaceIsCheckedAfterCorePreload(t *testing.T) {
	c, dir, actions := offlineAuxiliaryController(t)
	seedAuxiliary(t, dir, "sglang-qwen38-fn")
	c.memoryProbe = func() SystemMemory {
		available := 7.0
		for _, action := range *actions {
			if action == "POST /v1/runtime/prepare" {
				available = 5.0
			}
		}
		return SystemMemory{AvailableGiB: available, FreeGiB: available}
	}
	lease, err := c.AcquireWorkload(context.Background(), "flash-next", "flux2", 1.5)
	if err == nil || lease != nil {
		t.Fatal("cold headroom reused after core allocation")
	}
	if _, err = os.Stat(filepath.Join(dir, "flux2-klein-nvfp4-api")); err != nil {
		t.Fatal("preloaded core was killed on admission failure")
	}
}
