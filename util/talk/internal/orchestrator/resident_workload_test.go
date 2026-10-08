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

func residentAuxiliaryController(t *testing.T) (*Controller, string) {
	t.Helper()
	c, dir, _ := nativeImageController(t)
	defaults, _ := LoadCatalog()
	native, _ := defaults.Bundle("qwen38fn_exl3")
	for i := range c.catalog.Bundles {
		if c.catalog.Bundles[i].ID == native.ID {
			// Keep exercising the shared QWIM editor's residency policy.
			for j, id := range c.catalog.Bundles[i].Components {
				if id == "qwim-mmh3" {
					c.catalog.Bundles[i].Components[j] = "flux2"
				}
			}
			keep, budget := true, 24.0
			c.catalog.Bundles[i].Bindings["flux2"] = Deployment{KeepResident: &keep, MemoryGiB: &budget, StartAfterLLM: &keep}
			for _, id := range []string{"nemotron-asr", "qwen3-tts"} {
				c.catalog.Bundles[i].Bindings[id] = native.Bindings[id]
			}
		}
	}
	validated, err := ValidateCatalog(c.catalog)
	if err != nil {
		t.Fatal(err)
	}
	c.catalog = validated
	base := c.client.Transport
	c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path == "/v1/model" {
			return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"id":"qwen38fn_exl3","parameters":{"max_seq_len":1048576,"cache_size":1048576,"cache_mode":"Q8","use_vision":true}}`))}, nil
		}
		return base.RoundTrip(r)
	})
	seedAuxiliary(t, dir, "sparktalk-qwen38fn_exl3", "sparktalk-qwen-image21", "sparktalk-nemotron-asr", "sparktalk-qwen3-tts")
	return c, dir
}

func TestResidentAuxiliaryIdlePolicyFollowsSelectedBundle(t *testing.T) {
	for _, id := range []string{"flux2", "qwen3-tts"} {
		t.Run(id, func(t *testing.T) {
			c, dir := residentAuxiliaryController(t)
			x, _ := c.Catalog().ResolveComponent("qwen38fn_exl3", id)
			c.workloadActivity(x)()
			c.idleMu.Lock()
			lease := c.idleLeases[x.DeploymentKey()]
			if lease.timer != nil {
				t.Fatal("resident service scheduled automatic shutdown")
			}
			// A timer created before a policy change must not reclaim it either.
			c.scheduleWorkloadIdleLocked(x.DeploymentKey(), lease, time.Hour)
			epoch := lease.epoch
			c.idleMu.Unlock()
			c.reapIdleWorkload(x.DeploymentKey(), epoch)
			if _, err := os.Stat(filepath.Join(dir, x.Container)); err != nil {
				t.Fatal("resident service was reaped", err)
			}
			qad, _ := c.Catalog().ResolveComponent("flash-next", id)
			triggerImageIdle(c, qad)
			if _, err := os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
				t.Fatal("QAD idle reclamation was disabled", err)
			}
		})
	}
}

func TestStartupWorkloadSweepPreservesResidentMembers(t *testing.T) {
	c, dir := residentAuxiliaryController(t)
	b, _ := c.Catalog().Bundle("qwen38fn_exl3")
	if err := c.stopWorkloads(context.Background(), c.Catalog().StartBundleMembers(b)); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"sparktalk-qwen38fn_exl3", "sparktalk-qwen-image21", "sparktalk-nemotron-asr", "sparktalk-qwen3-tts"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatal("startup sweep stopped an already accounted resident", name, err)
		}
	}
}

func TestResidentWorkloadRetainsModulesAndRejectsPressure(t *testing.T) {
	c, dir := residentAuxiliaryController(t)
	for _, id := range []string{"flux2", "nemotron-asr", "qwen3-tts"} {
		release, err := c.AcquireWorkload(context.Background(), "qwen38fn_exl3", id, 1.5)
		if err != nil {
			t.Fatal(id, err)
		}
		c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 1, FreeGiB: 1} }
		if err := release(); err != nil {
			t.Fatal(err)
		}
		for _, name := range []string{"sparktalk-qwen38fn_exl3", "sparktalk-qwen-image21", "sparktalk-nemotron-asr", "sparktalk-qwen3-tts"} {
			if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
				t.Fatal("resident module was stopped", name, err)
			}
		}
		if release, err := c.AcquireWorkload(context.Background(), "qwen38fn_exl3", id, 1.5); err == nil || release != nil {
			t.Fatal("low-memory request admitted", id)
		}
		c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 120, FreeGiB: 110} }
	}
}

func TestResidentBundleStartsAuxiliariesAfterExistingLLM(t *testing.T) {
	c, dir := residentAuxiliaryController(t)
	if err := c.StartBundle(context.Background(), "qwen38fn_exl3", 1.5); err != nil {
		t.Fatal(err)
	}
	waitSupportOperation(t, c)
	if c.Operation().State != "complete" {
		t.Fatal(c.Operation())
	}
	seen := map[string]bool{}
	for _, step := range c.Operation().Steps {
		seen[step.ComponentID] = true
	}
	for _, id := range []string{"flux2", "nemotron-asr", "qwen3-tts"} {
		if !seen[id] {
			t.Fatal("resident auxiliary skipped at bundle startup", id)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen38fn_exl3")); err != nil {
		t.Fatal("LLM was stopped", err)
	}
}

func TestResidentBundleMemoryIncludesRetainedModelsAndOneWorkspace(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	b, _ := catalog.Bundle("qwen38fn_exl3")
	core, _ := catalog.ResolveComponent(b.ID, "qwen38fn_exl3")
	// Joint image/video startup floor 18 + ASR 3.5 + TTS 2.5 + workspace 10.
	if b.MemoryGiB != core.MemoryGiB+34 {
		t.Fatalf("resident stack not reserved: %+v", b)
	}
	var states []ComponentStatus
	for id, resident := range map[string]float64{"qwim-mmh3": 24, "nemotron-asr": 2, "qwen3-tts": 2} {
		x, _ := catalog.ResolveComponent(b.ID, id)
		states = append(states, ComponentStatus{Component: x, ResidentMemoryGiB: resident, MemoryMeasured: true})
	}
	if got := observedBundleBudget(catalog, b, states); got != core.MemoryGiB+36 {
		t.Fatalf("observed residency/workspace lost or double-counted: %v", got)
	}
	qad, _ := catalog.Bundle("flash-next")
	for _, id := range []string{"flux2", "nemotron-asr", "qwen3-tts"} {
		x, _ := catalog.ResolveComponent(qad.ID, id)
		if x.KeepResident {
			t.Fatal("resident policy leaked into QAD", id)
		}
	}
}
