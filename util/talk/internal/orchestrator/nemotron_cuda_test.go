package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNemotronPostPreparationGuardPreservesResidentModels(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	t.Cleanup(c.Close)
	seedAuxiliary(t, dir, "sglang-qwen38-fn-radixark", "sparktalk-embedding")
	x, _ := c.Catalog().ResolveComponent("flash-next-radixark", "nemotron-asr")
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 13, FreeGiB: 6.3} }
	err := c.prepareOrStartComponent(context.Background(), x, false)
	if err == nil || !strings.Contains(err.Error(), "ASR 준비 완료 후 CUDA") {
		t.Fatal("cold ASR bypassed post-preparation guard", err)
	}
	if _, err = os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
		t.Fatal("ASR started without bootstrap headroom")
	}
	for _, name := range []string{"sglang-qwen38-fn-radixark", "sparktalk-embedding"} {
		if _, err = os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatal("resident model stopped", name)
		}
	}
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 13, FreeGiB: 7.3} }
	if err = c.prepareOrStartComponent(context.Background(), x, false); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(filepath.Join(dir, x.Container)); err != nil {
		t.Fatal("qualified ASR not started", err)
	}
}
