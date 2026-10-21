package orchestrator

import (
	"context"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSetSwitchReleasesOnlyUnselectedIdleImage(t *testing.T) {
	for _, busy := range []bool{false, true} {
		t.Run(map[bool]string{false: "idle", true: "busy"}[busy], func(t *testing.T) {
			c, dir, _ := offlineAuxiliaryController(t)
			t.Cleanup(c.Close)
			x, _ := c.Catalog().ResolveComponent("qwen38fn_exl3", "qwim-mmh3")
			seedAuxiliary(t, dir, x.Container, "sparktalk-qwen38fn_exl3")
			current, _ := c.Catalog().Bundle("qwen38fn_exl3")
			if err := c.stopUnselectedManagedImages(context.Background(), current); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(filepath.Join(dir, x.Container)); err != nil {
				t.Fatal("selected resident image must survive", err)
			}
			if busy {
				base := c.client.Transport
				c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
					if r.URL.Path == "/v1/runtime/quiesce" {
						return &http.Response{StatusCode: 409, Body: io.NopCloser(strings.NewReader(`{"detail":"queued work"}`))}, nil
					}
					return base.RoundTrip(r)
				})
			}
			next, _ := c.Catalog().Bundle("flash-next-fp4-local")
			err := c.stopUnselectedManagedImages(context.Background(), next)
			if busy {
				if err == nil {
					t.Fatal("busy image must block switching before the old LLM is stopped")
				}
				if _, err := os.Stat(filepath.Join(dir, x.Container)); err != nil {
					t.Fatal("busy image was stopped", err)
				}
			} else if err != nil {
				t.Fatal(err)
			} else if _, err := os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
				t.Fatal("unselected idle image was retained", err)
			}
			if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen38fn_exl3")); err != nil {
				t.Fatal("image cleanup must not stop the LLM", err)
			}
		})
	}
}

func TestSetSwitchCreditsOnlyImageItWillRelease(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	t.Cleanup(c.Close)
	docker := "#!/bin/sh\ncase \"$*\" in\n *State.Status*sparktalk-qwim-mmh3*) printf running;;\n *State.Status*) printf exited;;\n *top*sparktalk-qwim-mmh3*) printf 'PID\\n999999999\\n';;\n esac\n"
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(docker), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "nvidia-smi"), []byte("#!/bin/sh\nprintf '999999999, 17920\\n'\n"), 0700); err != nil {
		t.Fatal(err)
	}
	b, _ := c.Catalog().Bundle("flash-next-fp4-local")
	plan := c.bundleMemoryPlan(context.Background(), b)
	if math.Abs(plan.FreedGiB-17.5) > .001 {
		t.Fatalf("unselected image's measured GPU allocation was not credited once: %+v", plan)
	}
	b, _ = c.Catalog().Bundle("qwen38fn_exl3")
	if plan := c.bundleMemoryPlan(context.Background(), b); plan.FreedGiB != 0 {
		t.Fatalf("selected image must not be credited as released: %+v", plan)
	}
}
