package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestHealthHistoryDistinguishesLatencyFromStartup(t *testing.T) {
	now := time.Now()
	instance := containerObservation{identity: "one", state: "running", started: now.Add(-time.Hour)}
	h := healthHistory{}
	h.reset(instance.identity)
	h.record(now, 10*time.Millisecond, nil)
	h.record(now.Add(time.Second), 3*time.Second, errors.New("timed out"))
	if got := h.health(now.Add(time.Second), instance, time.Minute); got != "online" {
		t.Fatal(got)
	}
	h.record(now.Add(4*time.Second), 3*time.Second, errors.New("timed out"))
	if got := h.health(now.Add(4*time.Second), instance, time.Minute); got != "online" {
		t.Fatal(got)
	}
	if got := h.health(now.Add(healthGrace), instance, time.Minute); got != "unresponsive" {
		t.Fatal("grace never expires", got)
	}
	h.record(now.Add(8*time.Second), 3*time.Second, errors.New("timed out"))
	if got := h.health(now.Add(8*time.Second), instance, time.Minute); got != "unresponsive" {
		t.Fatal(got)
	}
	h.record(now.Add(9*time.Second), time.Millisecond, nil)
	if h.failures != 0 || h.detail != "" || h.health(now.Add(9*time.Second), instance, time.Minute) != "online" {
		t.Fatal(h)
	}
	h.reset("restarted")
	instance.started = now
	h.record(now, 3*time.Second, errors.New("connection failed"))
	if got := h.health(now, instance, time.Minute); got != "starting" {
		t.Fatal(got)
	}
	if got := h.health(now.Add(2*time.Minute), instance, time.Minute); got != "unresponsive" {
		t.Fatal(got)
	}
	instance.oom = true
	if got := h.health(now, instance, time.Minute); got != "failed" {
		t.Fatal(got)
	}
	instance.oom = false
	instance.state = "exited"
	if got := h.health(now, instance, time.Minute); got != "offline" {
		t.Fatal(got)
	}
}

func statusFixture(t *testing.T, handler http.HandlerFunc) (*Controller, Component, string) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	dir := t.TempDir()
	state := filepath.Join(dir, "state")
	script := "#!/bin/sh\ncase \"$1\" in inspect) cat \"$STATUS_STATE\";; logs) printf 'model starting';; esac\n"
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("STATUS_STATE", state)
	c := newController(Catalog{Hosts: map[string]Host{"local": {}}})
	component := Component{ID: "test", Host: "local", Container: "test", HealthURL: server.URL, StartupTimeoutSeconds: 60}
	writeInstance(t, state, "one", "running", time.Now().Add(-time.Hour), 0, false)
	return c, component, state
}
func writeInstance(t *testing.T, file, id, state string, started time.Time, restarts int, oom bool) {
	t.Helper()
	if err := os.WriteFile(file, []byte(fmt.Sprintf("%s|%s|%s|%d|%t", id, state, started.Format(time.RFC3339Nano), restarts, oom)), 0600); err != nil {
		t.Fatal(err)
	}
}
func TestStatusProbeCoalescesAndResetsOnLifecycle(t *testing.T) {
	var requests atomic.Int32
	var failure atomic.Bool
	c, component, state := statusFixture(t, func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		time.Sleep(20 * time.Millisecond)
		if failure.Load() {
			w.WriteHeader(503)
		}
	})
	var wg sync.WaitGroup
	for i := 0; i < 12; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if got := c.observedStatus(context.Background(), component); got.Health != "online" {
				t.Errorf("%+v", got)
			}
		}()
	}
	wg.Wait()
	if requests.Load() != 1 {
		t.Fatalf("duplicate probes: %d", requests.Load())
	}
	failure.Store(true)
	// A restart must bypass the HTTP cache and prior successful state immediately.
	writeInstance(t, state, "one", "running", time.Now(), 1, false)
	got := c.observedStatus(context.Background(), component)
	if got.Health != "starting" || got.LastHealthyAt != nil || got.HealthFailures != 1 || got.HealthError != "health HTTP 503" || requests.Load() != 2 {
		t.Fatalf("%+v requests=%d", got, requests.Load())
	}
	writeInstance(t, state, "one", "exited", time.Now(), 1, true)
	got = c.observedStatus(context.Background(), component)
	if got.Health != "failed" || got.Phase != "OOM" || requests.Load() != 2 {
		t.Fatal(got)
	}
	writeInstance(t, state, "two", "exited", time.Now(), 0, false)
	if got = c.observedStatus(context.Background(), component); got.Health != "offline" {
		t.Fatal(got)
	}
}
func TestStatusProbeTimeoutIsNotLoading(t *testing.T) {
	var slow atomic.Bool
	c, component, _ := statusFixture(t, func(w http.ResponseWriter, r *http.Request) {
		if slow.Load() {
			<-r.Context().Done()
		}
	})
	c.client.Timeout = 20 * time.Millisecond
	if got := c.observedStatus(context.Background(), component); got.Health != "online" {
		t.Fatal(got)
	}
	slow.Store(true)
	monitor := c.statusMonitorFor(component)
	for i := 1; i <= 3; i++ {
		monitor.history.checked = time.Now().Add(-healthProbeInterval)
		got := c.observedStatus(context.Background(), component)
		want := "online"
		if i == 3 {
			want = "unresponsive"
		}
		if got.Health != want || got.HealthFailures != i || !strings.Contains(got.HealthError, "timed out") || got.HealthLatencyMS < 10 {
			t.Fatalf("%+v", got)
		}
		if i == 3 && got.Phase != "응답 지연" {
			t.Fatal(got)
		}
	}
	before := monitor.history
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	c.observedStatus(ctx, component)
	if monitor.history != before {
		t.Fatal("caller cancellation polluted health history")
	}
}
func TestOldProcessWithNoHistoryIsNotStarting(t *testing.T) {
	c, component, _ := statusFixture(t, func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(503) })
	if got := c.observedStatus(context.Background(), component); got.Health != "unresponsive" {
		t.Fatal(got)
	}
}
