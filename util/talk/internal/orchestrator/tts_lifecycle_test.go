package orchestrator

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestManagedAuxiliaryColdFreeSeparateFromAvailableAndWarmBudget(t *testing.T) {
	memory := SystemMemory{AvailableGiB: 18, FreeGiB: 5}
	plan := memoryPlan{NeededGiB: 3, RequiresCUDAStart: true, MinimumCUDAFreeGiB: managedAuxiliaryCUDAFreeGiB}
	var cold *cudaStartMemoryError
	if err := validateMemoryHeadroom(memory, plan, 1.5); !errors.As(err, &cold) {
		t.Fatalf("cached memory admitted a cold CUDA context: %v", err)
	}
	memory.FreeGiB = managedAuxiliaryCUDAFreeGiB
	if err := validateMemoryHeadroom(memory, plan, 1.5); err != nil {
		t.Fatal(err)
	}
	plan.RequiresCUDAStart = false
	memory.FreeGiB = 2
	if err := validateMemoryHeadroom(memory, plan, 1.5); err != nil {
		t.Fatalf("warm service unnecessarily required cold pages: %v", err)
	}
}

func TestASRInferenceLeaseDoesNotRestartDecodedMedia(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	t.Cleanup(c.Close)
	release, err := c.AcquireWorkload(context.Background(), "flash-next", "nemotron-asr", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	asr, _ := c.Catalog().ResolveComponent("flash-next", "nemotron-asr")
	media, _ := c.Catalog().ResolveSupport("flash-next", "extra-media")
	if _, err = os.Stat(filepath.Join(dir, asr.Container)); err != nil {
		t.Fatal("ASR inference not started", err)
	}
	if _, err = os.Stat(filepath.Join(dir, media.Container)); !os.IsNotExist(err) {
		t.Fatal("decoded Media unnecessarily restarted before ASR", err)
	}
}

// Opt-in only: exercise pressure cleanup against the real idle native API and
// Docker process without allocating memory to force pressure on a resident LLM.
func TestTTSLifecyclePressureLive(t *testing.T) {
	path := os.Getenv("SPARKTALK_TTS_PRESSURE_LIVE_CATALOG")
	if path == "" {
		t.Skip("live pressure runtime not requested")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var catalog Catalog
	if err = json.Unmarshal(data, &catalog); err != nil {
		t.Fatal(err)
	}
	c, err := NewControllerWithCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(c.Close)
	c.ConfigurePaths(os.Getenv("SPARKTALK_TTS_PRESSURE_LIVE_DATA"), os.Getenv("SPARKTALK_TTS_PRESSURE_LIVE_CACHE"))
	c.idleDuration = time.Hour
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	release, err := c.AcquireWorkload(ctx, "flash-next", "qwen3-tts", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := c.Catalog().ResolveComponent("flash-next", "qwen3-tts")
	before, err := exec.Command("docker", "inspect", "-f", "{{.State.Pid}}", x.Container).Output()
	if err != nil || strings.TrimSpace(string(before)) == "0" {
		t.Fatalf("TTS not running: %s %v", before, err)
	}
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 1, FreeGiB: 1} }
	if err = release(); err != nil {
		t.Fatal(err)
	}
	after, err := exec.Command("docker", "inspect", "-f", "{{.State.Pid}}", x.Container).Output()
	if err != nil || strings.TrimSpace(string(after)) != "0" {
		t.Fatalf("idle TTS pressure reclaim: %s %v", after, err)
	}
	t.Logf("simulated memory probe pressure reclaimed real idle TTS PID %s", strings.TrimSpace(string(before)))
}

func nativeTTSController(t *testing.T) (*Controller, string, Component) {
	t.Helper()
	c, dir, _ := offlineAuxiliaryController(t)
	x, _ := c.Catalog().Component("qwen3-tts")
	c.idleDuration = time.Hour
	seedAuxiliary(t, dir, x.Container)
	t.Cleanup(c.Close)
	return c, dir, x
}

func TestTTSWarmReuseCanceledWaiterAndPressureReclaim(t *testing.T) {
	c, dir, x := nativeTTSController(t)
	for i := 0; i < 2; i++ {
		release, err := c.AcquireWorkload(context.Background(), "flash-next", x.ID, 1.5)
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err = c.AcquireWorkload(ctx, "flash-next", x.ID, 1.5); err == nil {
			t.Fatal("canceled waiter admitted")
		}
		c.idleMu.Lock()
		users := c.idleLeases[x.DeploymentKey()].users
		c.idleMu.Unlock()
		if users != 1 {
			t.Fatalf("active users %d", users)
		}
		if err = release(); err != nil {
			t.Fatal(err)
		}
		if _, err = os.Stat(filepath.Join(dir, x.Container)); err != nil {
			t.Fatal("warm TTS stopped", err)
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
		t.Fatal("idle TTS not reclaimed under pressure")
	}
}

func TestTTSIdleReclaimProtectsNativeAndTalkQueues(t *testing.T) {
	for _, scenario := range []string{"idle", "busy", "queued", "unknown", "new-waiter"} {
		t.Run(scenario, func(t *testing.T) {
			c, dir, x := nativeTTSController(t)
			var end func()
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
					case "new-waiter":
						end = c.workloadActivity(x)
					}
				}
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(body))}, nil
			})
			triggerImageIdle(c, x)
			if end != nil {
				end()
			}
			_, err := os.Stat(filepath.Join(dir, x.Container))
			if scenario == "idle" {
				if !os.IsNotExist(err) {
					t.Fatal("TTS idle process retained")
				}
			} else if err != nil {
				t.Fatal("protected TTS stopped", err)
			}
		})
	}
}
