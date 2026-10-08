package orchestrator

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func nativeImageController(t *testing.T) (*Controller, string, Component) {
	t.Helper()
	c, dir, _ := offlineAuxiliaryController(t)
	defaults, _ := LoadCatalog()
	x, _ := defaults.Component("flux2")
	for i := range c.catalog.Components {
		if c.catalog.Components[i].ID == x.ID {
			x.Endpoint = c.catalog.Components[i].Endpoint
			x.HealthURL = x.Endpoint + "/health"
			x.AutoAddress = false
			c.catalog.Components[i] = x
			c.catalog.byComponent[x.ID] = x
		}
	}
	for i := range c.catalog.Bundles {
		delete(c.catalog.Bundles[i].Bindings, "flux2")
	}
	c.idleDuration = time.Hour // Tests explicitly trigger reclamation.
	seedAuxiliary(t, dir, x.Container)
	t.Cleanup(c.Close)
	return c, dir, x
}

func triggerImageIdle(c *Controller, x Component) {
	c.workloadActivity(x)()
	c.idleMu.Lock()
	epoch := c.idleLeases[x.DeploymentKey()].epoch
	c.idleMu.Unlock()
	c.reapIdleWorkload(x.DeploymentKey(), epoch)
}

func TestQwenImageIdleReclaimRequiresConfirmedIdleAndProtectsNewWaiter(t *testing.T) {
	for _, scenario := range []string{"idle", "busy", "queued", "unknown", "recent-direct-work", "new-waiter"} {
		t.Run(scenario, func(t *testing.T) {
			c, dir, x := nativeImageController(t)
			stops := 0
			var endWaiter func()
			c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
				body := `{"status":"ok","busy":false,"idle_for_seconds":7200}`
				if strings.HasSuffix(r.URL.Path, "/memory") {
					switch scenario {
					case "busy":
						body = `{"status":"ok","busy":true}`
					case "queued":
						body = `{"status":"ok","busy":false,"queued":1}`
					case "unknown":
						body = `{"status":"ok"}`
					case "recent-direct-work":
						body = `{"status":"ok","busy":false,"idle_for_seconds":1}`
					case "new-waiter":
						endWaiter = c.workloadActivity(x)
					}
				}
				if strings.HasSuffix(r.URL.Path, "/quiesce") {
					stops++
				}
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body))}, nil
			})
			triggerImageIdle(c, x)
			if endWaiter != nil {
				endWaiter()
			}
			_, err := os.Stat(filepath.Join(dir, x.Container))
			if scenario == "idle" {
				if stops != 1 || !os.IsNotExist(err) {
					t.Fatalf("idle process not reclaimed: quiesce=%d %v", stops, err)
				}
			} else if stops != 0 || err != nil {
				t.Fatalf("protected process reclaimed: %d %v", stops, err)
			}
		})
	}
}

func TestQwenImageConsecutiveLeasesRetainProcessAndPressureReclaims(t *testing.T) {
	c, dir, x := nativeImageController(t)
	for i := 0; i < 2; i++ {
		release, err := c.AcquireWorkload(context.Background(), "flash-next", x.ID, 1.5)
		if err != nil {
			t.Fatal(err)
		}
		if err = release(); err != nil {
			t.Fatal(err)
		}
		if _, err = os.Stat(filepath.Join(dir, x.Container)); err != nil {
			t.Fatal("warm process stopped", err)
		}
	}
	release, err := c.AcquireWorkload(context.Background(), "flash-next", x.ID, 1.5)
	if err != nil {
		t.Fatal(err)
	}
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 1, FreeGiB: 1} }
	if err = release(); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
		t.Fatal("pressure did not return process memory")
	}
}

func TestQwenImageCanceledQueuedRequestDoesNotStopActiveLease(t *testing.T) {
	c, dir, x := nativeImageController(t)
	release, err := c.AcquireWorkload(context.Background(), "flash-next", x.ID, 1.5)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err = c.AcquireWorkload(ctx, "flash-next", x.ID, 1.5); err == nil {
		t.Fatal("canceled request admitted")
	}
	c.idleMu.Lock()
	users := c.idleLeases[x.DeploymentKey()].users
	c.idleMu.Unlock()
	if users != 1 {
		t.Fatalf("active lease count %d", users)
	}
	if _, err = os.Stat(filepath.Join(dir, x.Container)); err != nil {
		t.Fatal("active process stopped")
	}
	if err = release(); err != nil {
		t.Fatal(err)
	}
}

func TestQwenImagePressureFromOtherWorkProtectsQueuedTalkImages(t *testing.T) {
	c, dir, x := nativeImageController(t)
	end := c.workloadActivity(x)
	defer end()
	if err := c.fluxRuntimeAction(context.Background(), x, "reclaim"); err == nil {
		t.Fatal("queued image was treated as idle")
	}
	if _, err := os.Stat(filepath.Join(dir, x.Container)); err != nil {
		t.Fatal("queued image process stopped", err)
	}
}
