package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestNVFP4ResidentStartupAccountsForTemporaryLoaderPeak(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	t.Cleanup(c.Close)
	b, _ := c.Catalog().Bundle("flash-next-radixark")
	asr, _ := c.Catalog().ResolveComponent(b.ID, "nemotron-asr")
	embed, _ := c.Catalog().ResolveComponent(b.ID, "extra-embedding")
	if !asr.KeepResident || !embed.KeepResident {
		t.Fatal("NVFP4 core modules must stay resident")
	}
	order := c.Catalog().startupOrder(c.Catalog().StartBundleMembers(b))
	pos := map[string]int{}
	for i, id := range order {
		pos[id] = i
	}
	if !(pos["flash-next-radixark"] < pos[asr.ID] && pos[asr.ID] < pos[embed.ID]) {
		t.Fatal("CUDA bootstrap order", order)
	}
	plan := c.bundleMemoryPlan(context.Background(), b)
	if plan.NeededGiB != 110 {
		t.Fatal("loader peak accumulated with deferred weights", plan)
	}
	seedAuxiliary(t, dir, "sglang-qwen38-fn-radixark", embed.Container)
	plan = c.bundleMemoryPlan(context.Background(), b)
	if plan.NeededGiB != 7.5 {
		t.Fatal("ASR recovery omitted embedding reload", plan)
	}
	if err := c.StopBundle(b.ID); err != nil {
		t.Fatal(err)
	}
	waitSupportOperation(t, c)
	if _, err := os.Stat(filepath.Join(dir, embed.Container)); !os.IsNotExist(err) {
		t.Fatal("model stop retained embedding")
	}
}
