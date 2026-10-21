package orchestrator

import (
	"context"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestTensorFoldAdmissionKeepsFourGiBFloor(t *testing.T) {
	c, _, _ := offlineAuxiliaryController(t)
	for _, id := range []string{"flash-next-fp4-local", "qwen38fn_exl3"} {
		b, _ := c.Catalog().Bundle(id)
		plan := memoryPlan{NeededGiB: 103, RequiresCUDAStart: true}
		for _, available := range []float64{106.9, 107} {
			m := SystemMemory{TotalGiB: 121.6, AvailableGiB: available, FreeGiB: 112}
			if err := c.validateBundleStartupMemory(b, m, plan, 1.5); (err == nil) != (available >= 107) {
				t.Fatalf("%s available=%v err=%v", id, available, err)
			}
		}
		x, _ := c.Catalog().Component(id)
		if !c.managedQwenTensorFold(x) {
			t.Fatal("watchdog missing", id)
		}
		x.Host = "worker"
		if c.managedQwenTensorFold(x) {
			t.Fatal("remote runtime uses local watchdog")
		}
	}
}

func fp4StartupTestController(t *testing.T, recoverOnRetry bool) (*Controller, Component, string) {
	t.Helper()
	c, dir, _ := offlineAuxiliaryController(t)
	t.Cleanup(c.Close)
	script := `#!/bin/sh
last=""; file=""; prev=""
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
 run*) echo reclaim-or-prepare >> "$AUX_TEST_ROOT/actions"; cat >/dev/null;;
esac
`
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	x, _ := c.Catalog().Component("flash-next-fp4-local")
	starts := func() int {
		data, _ := os.ReadFile(filepath.Join(dir, "actions"))
		return strings.Count(string(data), "start "+x.Container)
	}
	c.memoryProbe = func() SystemMemory {
		m := SystemMemory{TotalGiB: 121.6, AvailableGiB: 116.8, FreeGiB: 112.9}
		if _, err := os.Stat(filepath.Join(dir, x.Container)); err == nil {
			m.AvailableGiB, m.FreeGiB = 1, 1
			if recoverOnRetry && starts() >= 2 {
				m.AvailableGiB, m.FreeGiB = 16, 12
			}
		}
		return m
	}
	c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
		if _, err := os.Stat(filepath.Join(dir, x.Container)); err != nil || !recoverOnRetry || starts() < 2 {
			return nil, errors.New("still loading")
		}
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"status":"ok"}`))}, nil
	})
	return c, x, dir
}

func TestFP4PersistentPressureStopsOnlyItsModelAndDoesNotLoop(t *testing.T) {
	c, x, dir := fp4StartupTestController(t, false)
	seedAuxiliary(t, dir, "protected-service")
	err := c.startWithMemoryWatch(context.Background(), x, 1.5)
	var pressure *qwenFP4StartupPressure
	if !errors.As(err, &pressure) {
		t.Fatalf("expected measured pressure failure, got %v", err)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "actions"))
	if strings.Count(string(data), "start "+x.Container) != 1 || strings.Count(string(data), "stop "+x.Container) != 1 {
		t.Fatalf("retry count is not bounded: %s", data)
	}
	if _, err := os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
		t.Fatal("failed model was left running")
	}
	if _, err := os.Stat(filepath.Join(dir, "protected-service")); err != nil {
		t.Fatal("unrelated service was terminated")
	}
}

func TestFP4CanceledStartupCleansUpWithoutRetry(t *testing.T) {
	c, x, dir := fp4StartupTestController(t, false)
	c.memoryProbe = func() SystemMemory { return SystemMemory{TotalGiB: 121.6, AvailableGiB: 120, FreeGiB: 118} }
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		for i := 0; i < 500; i++ {
			if _, err := os.Stat(filepath.Join(dir, x.Container)); err == nil {
				cancel()
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
		cancel()
	}()
	defer cancel()
	if err := c.startWithMemoryWatch(ctx, x, 1.5); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation was lost: %v", err)
	}
	data, _ := os.ReadFile(filepath.Join(dir, "actions"))
	if got := strings.Count(string(data), "start "+x.Container); got != 1 {
		t.Fatal("canceled start retried", strconv.Itoa(got), string(data))
	}
	if _, err := os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
		t.Fatal("canceled model was left running")
	}
}
