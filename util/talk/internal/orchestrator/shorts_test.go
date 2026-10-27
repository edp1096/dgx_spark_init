package orchestrator

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFishLeaseRestoresDisplacedResidentImage(t *testing.T) {
	c, dir := residentAuxiliaryController(t)
	seedAuxiliary(t, dir, "sparktalk-fish-drama3")
	c.memoryProbe = func() SystemMemory {
		available := 20.
		if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen-image21")); os.IsNotExist(err) {
			available = 90
		}
		return SystemMemory{AvailableGiB: available, FreeGiB: available}
	}
	release, err := c.AcquireWorkload(context.Background(), "qwen38fn_exl3", "extra-fish", 1.5)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen-image21")); !os.IsNotExist(err) {
		t.Fatal("image was not displaced", err)
	}
	if err = release(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "sparktalk-fish-drama3")); !os.IsNotExist(err) {
		t.Fatal("Fish retained over restored image", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen-image21")); err != nil {
		t.Fatal("resident image not restored", err)
	}
	for _, name := range []string{"sparktalk-qwen38fn_exl3", "sparktalk-qwen3-tts"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatal("default model interrupted", name, err)
		}
	}
}

func TestFishDoesNotDisplaceBusyImage(t *testing.T) {
	c, dir := residentAuxiliaryController(t)
	seedAuxiliary(t, dir, "sparktalk-fish-drama3")
	c.memoryProbe = func() SystemMemory { return SystemMemory{AvailableGiB: 20, FreeGiB: 20} }
	base := c.client.Transport
	c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
		if r.URL.Host == "flux2.test" && strings.HasSuffix(r.URL.Path, "/quiesce") {
			return &http.Response{StatusCode: 409, Body: io.NopCloser(strings.NewReader(`{"busy":true}`))}, nil
		}
		return base.RoundTrip(r)
	})
	if release, err := c.AcquireWorkload(context.Background(), "qwen38fn_exl3", "extra-fish", 1.5); err == nil || release != nil {
		t.Fatal("busy image was displaced")
	}
	if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen-image21")); err != nil {
		t.Fatal("busy image stopped", err)
	}
}

func TestCanceledFishRestoresImageWithoutRuntimeErrorToast(t *testing.T) {
	c, dir := residentAuxiliaryController(t)
	seedAuxiliary(t, dir, "sparktalk-fish-drama3")
	c.memoryProbe = func() SystemMemory {
		available := 20.
		if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen-image21")); os.IsNotExist(err) {
			available = 90
		}
		return SystemMemory{AvailableGiB: available, FreeGiB: available}
	}
	ctx, cancel := context.WithCancel(context.Background())
	release, err := c.AcquireWorkload(ctx, "qwen38fn_exl3", "extra-fish", 1.5)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	cancel()
	if err = release(); err != nil {
		t.Fatal(err)
	}
	if c.Operation().State != "complete" {
		t.Fatalf("expected cancellation shown as failure: %+v", c.Operation())
	}
	if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen-image21")); err != nil {
		t.Fatal("image not restored after cancellation", err)
	}
}

func TestStoppedCoreDoesNotAutoStartShortsFromAStatusRead(t *testing.T) {
	c, dir := residentAuxiliaryController(t)
	if err := os.Remove(filepath.Join(dir, "sparktalk-qwen38fn_exl3")); err != nil {
		t.Fatal(err)
	}
	// This fixture has no running language-model container and no Shorts worker.
	c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 503, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})
	err := c.EnsureShortsWorker(context.Background(), "qwen38fn_exl3", 4)
	if err == nil || !strings.Contains(err.Error(), "세트가 중지") {
		t.Fatal(err)
	}
	if c.Operation().State == "running" {
		t.Fatal("stopped set started a background service")
	}
}
