package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestQAD512ResidentStartupUsesQualifiedLoadOrder(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	keep, serving, peak, imageStartup, imageWorkspace := true, 94.0, 108.0, 7.0, 6.0
	opts := map[string]string{"MODEL_VARIANT": "huihui_lil", "MTP_TOKENS": "3", "MAX_MODEL_LEN": "524288"}
	imageOpts := map[string]string{"IMAGE_RESIDENCY": "dit"}
	for i := range catalog.Bundles {
		b := &catalog.Bundles[i]
		if b.ID != "flash-next" {
			continue
		}
		b.ContextTokens = 524288
		b.Bindings["flash-next"] = Deployment{RuntimeOptions: &opts, MemoryGiB: &serving, StartupMemoryGiB: &peak}
		b.Bindings["flux2"] = Deployment{RuntimeOptions: &imageOpts, KeepResident: &keep, StartAfterLLM: &keep, StartupMemoryGiB: &imageStartup, WorkspaceMemoryGiB: &imageWorkspace}
		for _, id := range []string{"nemotron-asr", "qwen3-tts"} {
			b.Bindings[id] = Deployment{KeepResident: &keep, StartAfterLLM: &keep}
		}
	}
	catalog, err = ValidateCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := catalog.Bundle("flash-next")
	if !catalog.qad512DiTResident(b) {
		t.Fatal("qualified profile was not recognized")
	}
	order := catalog.startupOrder(catalog.StartBundleMembers(b))
	var cores []string
	for _, id := range order {
		x, _ := catalog.ResolveComponent(b.ID, id)
		if !x.IsSupport() {
			cores = append(cores, id)
		}
	}
	if !reflect.DeepEqual(cores, []string{"flash-next", "nemotron-asr", "qwen3-tts", "flux2"}) {
		t.Fatal(order)
	}
	llm, _ := catalog.ResolveComponent(b.ID, "flash-next")
	if llm.MemoryGiB != 94 || llm.StartupMemoryGiB != 108 {
		t.Fatal(llm)
	}
	image, _ := catalog.ResolveComponent(b.ID, "flux2")
	if image.WorkspaceMemoryGiB != 6 {
		t.Fatal(image.WorkspaceMemoryGiB)
	}
	other, _ := catalog.ResolveComponent("qwen38fn_exl3", "qwim-mmh3")
	if other.RuntimeOptions["IMAGE_RESIDENCY"] == "dit" {
		t.Fatal("changed another image deployment")
	}
	dir := t.TempDir()
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	if err = os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\ncase \"$*\" in *inspect*) printf exited;; esac\n"), 0700); err != nil {
		t.Fatal(err)
	}
	c, err := NewControllerWithCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	plan := c.bundleMemoryPlan(context.Background(), b)
	if plan.NeededGiB != 108 {
		t.Fatalf("loading peak must not include later auxiliaries: %+v", plan)
	}
	if err = validateMemoryHeadroom(SystemMemory{AvailableGiB: 118, FreeGiB: 80}, plan, 1.5); err != nil {
		t.Fatal(err)
	}
	if err = validateMemoryHeadroom(SystemMemory{AvailableGiB: 109, FreeGiB: 80}, plan, 1.5); err == nil {
		t.Fatal("unsafe startup admitted")
	}
	// An unqualified 640K setting retains its conservative 108 GiB fallback.
	llm.RuntimeOptions["MAX_MODEL_LEN"] = "655360"
	if llm.runtimeMemoryEstimate().MemoryGiB != 108 {
		t.Fatal("other profiles were relaxed")
	}
}

func TestOrderedStartupPeakIncludesEarlierResidents(t *testing.T) {
	p := startupMemoryPhases{ordered: true}
	p.add(Component{MemoryGiB: 7, StartupMemoryGiB: 7}, 0)
	p.add(Component{MemoryGiB: 94, StartupMemoryGiB: 108}, 0)
	if p.needed() != 115 {
		t.Fatal(p.needed())
	}
}
