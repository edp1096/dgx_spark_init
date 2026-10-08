package plugins_test

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/plugins"
)

var ctx = context.Background()

func fixture(id string) plugins.Definition {
	return plugins.Definition{Manifest: plugins.Manifest{ID: id, Name: id, Version: "1.0.0", API: 1, DataVersion: 1, Permissions: []string{"storage", "tools", "jobs"}, Operations: []plugins.Operation{{Name: "echo", Description: "Echo", Parameters: json.RawMessage(`{"type":"object"}`), Tool: true, Background: true, TimeoutSeconds: 2}}}, Handle: func(c context.Context, h plugins.Host, _ string, r plugins.Request) (json.RawMessage, error) {
		return r.Input, nil
	}}
}
func database(t *testing.T) *db.DB {
	t.Helper()
	d, err := db.Open(filepath.Join(t.TempDir(), "talk.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Close() })
	return d
}
func manager(t *testing.T, d *db.DB, defs ...plugins.Definition) *plugins.Manager {
	t.Helper()
	m, err := plugins.New(ctx, d, map[string]plugins.Service{"model.complete": func(context.Context, plugins.Request) (json.RawMessage, error) { return json.RawMessage(`{}`), nil }}, defs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c, cancel := context.WithTimeout(ctx, time.Second); defer cancel(); m.Close(c) })
	return m
}
func enable(t *testing.T, m *plugins.Manager, id string) {
	t.Helper()
	var grants []string
	for _, v := range m.List() {
		if v.Manifest.ID == id {
			grants = v.Manifest.Permissions
		}
	}
	if err := m.Configure(ctx, id, json.RawMessage(`{}`), grants); err != nil {
		t.Fatal(err)
	}
	if err := m.Enable(ctx, id); err != nil {
		t.Fatal(err)
	}
}
func waitRun(t *testing.T, m *plugins.Manager, id, rid string) plugins.Run {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		runs, err := m.Runs(ctx, id)
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range runs {
			if r.ID == rid && r.Status != "running" {
				return r
			}
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("run did not settle")
	return plugins.Run{}
}
func TestCapabilitiesNamespaceAndRevocation(t *testing.T) {
	d := database(t)
	a, b := fixture("alpha"), fixture("beta")
	var host plugins.Host
	a.Start = func(c context.Context, h plugins.Host) error {
		host = h
		return h.Put(c, "key", json.RawMessage(`"alpha"`))
	}
	m := manager(t, d, a, b)
	if !errors.Is(m.Enable(ctx, "alpha"), plugins.ErrPermission) {
		t.Fatal("ungranted activation accepted")
	}
	enable(t, m, "alpha")
	enable(t, m, "beta")
	if err := host.Put(ctx, "../beta/key", json.RawMessage(`"still-alpha"`)); err != nil {
		t.Fatal(err)
	}
	if _, err := d.PluginValue(ctx, "beta", "key"); !errors.Is(err, plugins.ErrNotFound) {
		t.Fatal("namespace leaked")
	}
	if _, err := host.Call(ctx, "model.complete", plugins.Request{Input: json.RawMessage(`{}`)}); !errors.Is(err, plugins.ErrPermission) {
		t.Fatal("ungranted service accepted")
	}
	if len(m.Tools()) != 2 {
		t.Fatal("tools not exposed")
	}
	if err := m.Disable(ctx, "alpha"); err != nil {
		t.Fatal(err)
	}
	if err := host.Put(ctx, "key", json.RawMessage(`1`)); !errors.Is(err, plugins.ErrDisabled) {
		t.Fatal("stale host can write")
	}
	if _, err := m.Call(ctx, "alpha", "echo", plugins.Request{Input: json.RawMessage(`{}`)}); !errors.Is(err, plugins.ErrDisabled) {
		t.Fatal("disabled tool callable")
	}
	if len(m.Tools()) != 1 {
		t.Fatal("disabled tools exposed")
	}
}
func TestJobsIdempotencyCancellationAndPanicIsolation(t *testing.T) {
	d := database(t)
	f := fixture("worker")
	var calls atomic.Int32
	started := make(chan struct{}, 2)
	f.Handle = func(c context.Context, h plugins.Host, _ string, r plugins.Request) (json.RawMessage, error) {
		calls.Add(1)
		switch string(r.Input) {
		case `{"panic":true}`:
			panic("fixture")
		case `{"wait":true}`:
			started <- struct{}{}
			<-c.Done()
			return nil, c.Err()
		}
		return r.Input, nil
	}
	m := manager(t, d, f, fixture("other"))
	enable(t, m, "worker")
	enable(t, m, "other")
	r := plugins.Request{Input: json.RawMessage(`{"ok":true}`)}
	run, err := m.Submit(ctx, "worker", "echo", "once", r)
	if err != nil {
		t.Fatal(err)
	}
	if got := waitRun(t, m, "worker", run.ID); got.Status != "completed" {
		t.Fatal(got)
	}
	// Wait for lifecycle lease release as the ledger write precedes release.
	for m.List()[1].Active > 0 {
		time.Sleep(time.Millisecond)
	}
	again, err := m.Submit(ctx, "worker", "echo", "once", r)
	if err != nil || again.ID != run.ID || calls.Load() != 1 {
		t.Fatalf("duplicate execution: %+v %v %d", again, err, calls.Load())
	}
	if _, err = m.Submit(ctx, "worker", "echo", "once", plugins.Request{Input: json.RawMessage(`{}`)}); !errors.Is(err, plugins.ErrConflict) {
		t.Fatal("idempotency key changed meaning")
	}
	if _, err = m.Call(ctx, "worker", "echo", plugins.Request{Input: json.RawMessage(`{"panic":true}`)}); err == nil {
		t.Fatal("panic not recovered")
	}
	if _, err = m.Call(ctx, "other", "echo", r); err != nil {
		t.Fatal("other plugin affected", err)
	}
	run, err = m.Submit(ctx, "worker", "echo", "wait", plugins.Request{Input: json.RawMessage(`{"wait":true}`)})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	again, err = m.Submit(ctx, "worker", "echo", "wait", plugins.Request{Input: json.RawMessage(`{"wait":true}`)})
	if err != nil || again.ID != run.ID {
		t.Fatal("running retry was not deduplicated", again, err)
	}
	if err = m.Cancel("other", run.ID); !errors.Is(err, plugins.ErrNotFound) {
		t.Fatal("cross-plugin cancel")
	}
	if err = m.Disable(ctx, "worker"); err != nil {
		t.Fatal(err)
	}
	if got := waitRun(t, m, "worker", run.ID); got.Status != "canceled" {
		t.Fatal(got)
	}
}
func TestRestartRecoveryAndMigrationRollback(t *testing.T) {
	d := database(t)
	f := fixture("fixture")
	m := manager(t, d, f)
	enable(t, m, "fixture")
	if err := d.PutPluginValue(ctx, "fixture", "original", json.RawMessage(`1`)); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(ctx); err != nil {
		t.Fatal(err)
	}
	m2 := manager(t, d, f)
	if m2.List()[0].Status != "active" {
		t.Fatal("enabled state not restored")
	}
	if err := m2.Disable(ctx, "fixture"); err != nil {
		t.Fatal(err)
	}
	m2.Close(ctx)
	_, _, err := d.CreatePluginRun(ctx, plugins.Run{ID: "interrupted", Key: "interrupted", PluginID: "fixture", Operation: "echo", Request: plugins.Request{Input: json.RawMessage(`{}`)}, Status: "running", CreatedAt: time.Now(), UpdatedAt: time.Now()})
	if err != nil {
		t.Fatal(err)
	}
	f.Manifest.Version = "2.0.0"
	f.Manifest.DataVersion = 2
	fail := true
	f.Migrate = func(_ context.Context, from, to int, values map[string]json.RawMessage) error {
		delete(values, "original")
		values["new"] = json.RawMessage(`2`)
		if fail {
			return errors.New("rollback")
		}
		return nil
	}
	m3 := manager(t, d, f)
	if !errors.Is(m3.Enable(ctx, "fixture"), plugins.ErrConflict) {
		t.Fatal("unmigrated plugin active")
	}
	if got := waitRun(t, m3, "fixture", "interrupted"); got.Status != "interrupted" {
		t.Fatal(got)
	}
	if m3.Migrate(ctx, "fixture") == nil {
		t.Fatal("failed migration accepted")
	}
	value, err := d.PluginValue(ctx, "fixture", "original")
	if err != nil || string(value) != "1" {
		t.Fatal("partial migration escaped")
	}
	fail = false
	if err = m3.Migrate(ctx, "fixture"); err != nil {
		t.Fatal(err)
	}
	if err = m3.Enable(ctx, "fixture"); err != nil {
		t.Fatal(err)
	}
	s, err := d.PluginSettings(ctx, "fixture")
	if err != nil || s.DataVersion != 2 {
		t.Fatal(s, err)
	}
}
func TestManifestValidationAndPayloadFailure(t *testing.T) {
	d := database(t)
	f := fixture("fixture")
	bad := f
	bad.Manifest.API = 2
	if _, err := plugins.New(ctx, d, nil, []plugins.Definition{bad}); err == nil {
		t.Fatal("incompatible API accepted")
	}
	if _, err := plugins.New(ctx, d, nil, []plugins.Definition{f, f}); err == nil {
		t.Fatal("duplicate accepted")
	}
	f.Handle = func(context.Context, plugins.Host, string, plugins.Request) (json.RawMessage, error) {
		return json.RawMessage(`broken`), nil
	}
	m := manager(t, d, f)
	enable(t, m, "fixture")
	if _, err := m.Call(ctx, "fixture", "echo", plugins.Request{Input: json.RawMessage(`{}`)}); err == nil {
		t.Fatal("invalid output accepted")
	}
	runs, err := m.Runs(ctx, "fixture")
	if err != nil || len(runs) != 1 || runs[0].Status != "failed" {
		t.Fatal("failure not persisted", runs, err)
	}
}

func TestSynchronousCancellationAndTimeoutLedger(t *testing.T) {
	d := database(t)
	f := fixture("fixture")
	f.Manifest.Operations[0].TimeoutSeconds = 1
	started := make(chan struct{}, 2)
	f.Handle = func(c context.Context, _ plugins.Host, _ string, _ plugins.Request) (json.RawMessage, error) {
		started <- struct{}{}
		<-c.Done()
		return nil, c.Err()
	}
	m := manager(t, d, f)
	enable(t, m, "fixture")
	finished := make(chan error, 1)
	go func() {
		_, err := m.Call(ctx, "fixture", "echo", plugins.Request{Input: json.RawMessage(`{}`)})
		finished <- err
	}()
	<-started
	runs, err := m.Runs(ctx, "fixture")
	if err != nil || len(runs) != 1 {
		t.Fatal(runs, err)
	}
	if err = m.Cancel("fixture", runs[0].ID); err != nil {
		t.Fatal(err)
	}
	if err = <-finished; !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if got := waitRun(t, m, "fixture", runs[0].ID); got.Status != "canceled" {
		t.Fatal(got)
	}
	if _, err = m.Call(ctx, "fixture", "echo", plugins.Request{Input: json.RawMessage(`{}`)}); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	runs, err = m.Runs(ctx, "fixture")
	if err != nil || runs[0].Status != "timed_out" {
		t.Fatal(runs, err)
	}
}
func TestReadyAndDurableReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "persistent.db")
	d, err := db.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	f := fixture("fixture")
	var h plugins.Host
	f.Start = func(_ context.Context, host plugins.Host) error {
		h = host
		select {
		case <-host.Ready():
			t.Error("ready before durable activation")
		default:
		}
		return nil
	}
	m, err := plugins.New(ctx, d, nil, []plugins.Definition{f})
	if err != nil {
		t.Fatal(err)
	}
	enable(t, m, "fixture")
	select {
	case <-h.Ready():
	default:
		t.Fatal("ready not published")
	}
	if err = h.Put(ctx, "persist", json.RawMessage(`{"v":3}`)); err != nil {
		t.Fatal(err)
	}
	if err = m.Close(ctx); err != nil {
		t.Fatal(err)
	}
	d.Close()
	d, err = db.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	m, err = plugins.New(ctx, d, nil, []plugins.Definition{f})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close(ctx)
	if m.List()[0].Status != "active" {
		t.Fatal("not restored")
	}
	value, err := h.Get(ctx, "persist")
	if err != nil || string(value) != `{"v":3}` {
		t.Fatal(string(value), err)
	}
}
func TestLifecyclePanicDoesNotActivateOrBlockOtherPlugins(t *testing.T) {
	d := database(t)
	bad := fixture("bad")
	bad.Start = func(context.Context, plugins.Host) error { panic("bad start") }
	m := manager(t, d, bad, fixture("good"))
	if err := m.Configure(ctx, "bad", json.RawMessage(`{}`), bad.Manifest.Permissions); err != nil {
		t.Fatal(err)
	}
	if m.Enable(ctx, "bad") == nil {
		t.Fatal("panic accepted")
	}
	if m.List()[0].Status != "failed" {
		t.Fatal("failure not visible")
	}
	enable(t, m, "good")
	if _, err := m.Call(ctx, "good", "echo", plugins.Request{Input: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
}

func TestUpgradeRemovesUndeclaredGrants(t *testing.T) {
	d := database(t)
	f := fixture("fixture")
	f.Manifest.Permissions = append(f.Manifest.Permissions, "model.complete")
	var host plugins.Host
	f.Start = func(_ context.Context, h plugins.Host) error { host = h; return nil }
	m := manager(t, d, f)
	enable(t, m, "fixture")
	if _, err := host.Call(ctx, "model.complete", plugins.Request{Input: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
	m.Close(ctx)
	f.Manifest.Version = "1.0.1"
	f.Manifest.Permissions = f.Manifest.Permissions[:3]
	next := manager(t, d, f)
	if next.List()[0].Status != "active" {
		t.Fatal("compatible version did not restore")
	}
	if _, err := host.Call(ctx, "model.complete", plugins.Request{Input: json.RawMessage(`{}`)}); !errors.Is(err, plugins.ErrPermission) {
		t.Fatal("removed permission retained", err)
	}
	for _, permission := range next.List()[0].Settings.Grants {
		if permission == "model.complete" {
			t.Fatal("stale grant persisted")
		}
	}
}

func TestToolNamespaceCollisionAndInvalidSchemaRejected(t *testing.T) {
	d := database(t)
	a, b := fixture("a__b"), fixture("a")
	a.Manifest.Operations[0].Name = "c"
	b.Manifest.Operations[0].Name = "b__c"
	if _, err := plugins.New(ctx, d, nil, []plugins.Definition{a, b}); err == nil {
		t.Fatal("ambiguous tool namespace accepted")
	}
	bad := fixture("bad")
	bad.Manifest.Operations[0].Parameters = json.RawMessage(`{"type":"string"}`)
	if _, err := plugins.New(ctx, d, nil, []plugins.Definition{bad}); err == nil {
		t.Fatal("non-object tool schema accepted")
	}
}
