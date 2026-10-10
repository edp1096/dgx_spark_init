package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestResidentEmbeddingUsesSharedQueueAndStopsOutsideItsSet(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	t.Cleanup(c.Close)
	x, ok := c.Catalog().ResolveComponent("qwen38fn_exl3", "extra-embedding")
	if !ok || !isCUDAComponent(x) || !x.KeepResident {
		t.Fatal("embedding GPU classification missing")
	}
	seedAuxiliary(t, dir, x.Container, "sparktalk-qwen38fn_exl3", "sparktalk-qwen3-tts", "sparktalk-nemotron-asr")
	c.idleDuration = time.Hour
	release, err := c.AcquireWorkload(context.Background(), "qwen38fn_exl3", x.ID, 1.5)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err = c.AcquireWorkload(ctx, "qwen38fn_exl3", "extra-documents", 1.5); err == nil {
		t.Fatal("second workload bypassed shared lease")
	}
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 1, FreeGiB: 1} }
	if err = release(); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(filepath.Join(dir, x.Container)); err != nil {
		t.Fatal("resident embedding reclaimed")
	}
	qad, _ := c.Catalog().Bundle("flash-next")
	if _, err = c.AcquireWorkload(context.Background(), qad.ID, x.ID, 1.5); err == nil {
		t.Fatal("QAD admitted embedding")
	}
	if err = c.stopUnselectedEmbedding(context.Background(), qad); err != nil {
		t.Fatal(err)
	}
	if _, err = os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
		t.Fatal("embedding survived switch outside its set")
	}
	for _, name := range []string{"sparktalk-qwen38fn_exl3", "sparktalk-qwen3-tts", "sparktalk-nemotron-asr"} {
		if _, err = os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatal("resident core stopped", name)
		}
	}
	if err = os.Remove(filepath.Join(dir, "sparktalk-qwen38fn_exl3")); err != nil {
		t.Fatal(err)
	}
	if _, err = c.AcquireWorkload(context.Background(), "qwen38fn_exl3", x.ID, 1.5); err == nil {
		t.Fatal("stopped model set restarted embedding")
	}
	if _, err = os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
		t.Fatal("embedding restarted after set stop")
	}
}
