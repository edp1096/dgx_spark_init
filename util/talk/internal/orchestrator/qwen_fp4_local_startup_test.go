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

func TestFP4GuardedAdmissionKeepsCapacityAndUserReserve(t *testing.T) {
	c, _, _ := offlineAuxiliaryController(t)
	b, _ := c.Catalog().Bundle("flash-next-fp4-local")
	plan := memoryPlan{NeededGiB: 116, RequiresCUDAStart: true}
	for _, tc := range []struct {
		name                   string
		total, available, free float64
		reserve                float64
		allow                  bool
	}{
		{"reported failure", 121.6, 116.8, 112.9, 1.5, true},
		{"larger shortage", 121.6, 115, 112, 1.5, false},
		{"larger user reserve", 121.6, 116.8, 112.9, 4, false},
		{"not enough physical memory", 116, 115.9, 114, 1.5, false},
		{"immediate CUDA shortage", 121.6, 116.8, .5, 1.5, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := SystemMemory{TotalGiB: tc.total, AvailableGiB: tc.available, FreeGiB: tc.free}
			if got := c.validateBundleStartupMemory(b, m, plan, tc.reserve); (got == nil) != tc.allow {
				t.Fatalf("allow=%v, error=%v", tc.allow, got)
			}
		})
	}
	m := SystemMemory{TotalGiB: 121.6, AvailableGiB: 116.8, FreeGiB: 112.9}
	other, _ := c.Catalog().Bundle("flash-next-radixark")
	if c.validateBundleStartupMemory(other, m, plan, 1.5) == nil {
		t.Fatal("guarded exception escaped into an unrelated model")
	}
	x, _ := c.Catalog().Component(b.ID)
	x.StartupMemoryGiB = 117
	if c.guardedQwenFP4Local(x) {
		t.Fatal("custom startup reservation was relaxed")
	}
	x, _ = c.Catalog().Component(b.ID)
	x.Host = "worker"
	if c.guardedQwenFP4Local(x) {
		t.Fatal("remote startup used a local memory watchdog")
	}
	cat := c.Catalog()
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == b.ID {
			keep := true
			binding := cat.Bundles[i].Bindings["nemotron-asr"]
			binding.KeepResident = &keep
			cat.Bundles[i].Bindings["nemotron-asr"] = binding
		}
	}
	custom, err := NewControllerWithCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	b, _ = custom.Catalog().Bundle(b.ID)
	if custom.validateBundleStartupMemory(b, m, plan, 1.5) == nil {
		t.Fatal("custom resident auxiliary profile was relaxed")
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

func TestFP4StartButtonAutomaticallyRecoversAndRetriesOnce(t *testing.T) {
	c, x, dir := fp4StartupTestController(t, true)
	// The reported 116.8 GiB case must pass through the public start path.
	if err := c.CheckBundleStart(context.Background(), x.ID, 1.5); err != nil {
		t.Fatal(err)
	}
	if actions, _ := os.ReadFile(filepath.Join(dir, "actions")); len(actions) != 0 {
		t.Fatalf("a read-only probe mutated runtime state: %s", actions)
	}
	if err := c.StartBundle(context.Background(), x.ID, 1.5); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(15 * time.Second)
	for c.Operation().State == "running" && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if op := c.Operation(); op.State != "complete" {
		t.Fatalf("automatic startup did not finish: %+v", op)
	}
	actions, _ := os.ReadFile(filepath.Join(dir, "actions"))
	if strings.Count(string(actions), "start "+x.Container) != 2 || strings.Count(string(actions), "stop "+x.Container) != 1 {
		t.Fatalf("expected one cleaned attempt and one successful retry: %s", actions)
	}
	if _, err := os.Stat(filepath.Join(dir, x.Container)); err != nil {
		t.Fatal("successful retry was stopped", err)
	}
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
	if strings.Count(string(data), "start "+x.Container) != 2 || strings.Count(string(data), "stop "+x.Container) != 2 {
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
