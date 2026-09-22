package plugins_test

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/plugins"
)

func buildExternal(t *testing.T, version string) []byte {
	t.Helper()
	dir := t.TempDir()
	talk, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	// A separate module imports the public SDK. No internal Talk packages and
	// no recompilation or registration in the host are required.
	mod := "module thirdparty.example/plugin\n\ngo 1.25.0\nrequire sparktalk v0.0.0\nreplace sparktalk => " + talk + "\n"
	os.WriteFile(filepath.Join(dir, "go.mod"), []byte(mod), 0600)
	source, err := os.ReadFile("testdata/fixture/main.go")
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "main.go"), source, 0600)
	cmd := exec.Command("go", "build", "-ldflags", "-X main.version="+version, "-o", "plugin", ".")
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0", "GOWORK=off")
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build third-party: %v %s", err, b)
	}
	binary, err := os.ReadFile(filepath.Join(dir, "plugin"))
	if err != nil {
		t.Fatal(err)
	}
	return binary
}
func externalManifest(version string, data int) plugins.PackageManifest {
	m := fixture("example").Manifest
	m.Version = version
	m.DataVersion = data
	for _, op := range []string{"storage", "isolation", "spawn", "secret", "hang", "crash", "oversize"} {
		m.Operations = append(m.Operations, plugins.Operation{Name: op, Parameters: json.RawMessage(`{"type":"object"}`), TimeoutSeconds: 1})
	}
	return plugins.PackageManifest{Manifest: m, Executable: "plugin", Platform: runtime.GOOS + "/" + runtime.GOARCH}
}
func archiveExternal(t *testing.T, m plugins.PackageManifest, binary []byte, extra string) []byte {
	t.Helper()
	var b bytes.Buffer
	w := zip.NewWriter(&b)
	meta, _ := json.Marshal(m)
	for name, data := range map[string][]byte{"plugin.json": meta, "plugin": binary} {
		f, e := w.Create(name)
		if e != nil {
			t.Fatal(e)
		}
		f.Write(data)
	}
	if extra != "" {
		f, _ := w.Create(extra)
		f.Write([]byte("bad"))
	}
	if e := w.Close(); e != nil {
		t.Fatal(e)
	}
	return b.Bytes()
}
func TestExternalPackageLifecycle(t *testing.T) {
	if runtime.GOOS != "linux" || (runtime.GOARCH != "arm64" && runtime.GOARCH != "amd64") {
		t.Skip("external sandbox is supported on Linux arm64/amd64")
	}
	d := database(t)
	m := manager(t, d)
	root := t.TempDir()
	if e := m.OpenPackages(ctx, root); e != nil {
		t.Fatal(e)
	}
	v1 := archiveExternal(t, externalManifest("1.0.0", 1), buildExternal(t, "1.0.0"), "")
	if e := m.InstallPackage(ctx, bytes.NewReader(v1)); e != nil {
		t.Fatal(e)
	}
	if !m.List()[0].External || m.List()[0].Settings.Enabled {
		t.Fatal("installation must default disabled")
	}
	enable(t, m, "example")
	r := plugins.Request{Input: json.RawMessage(`{"retained":42}`)}
	out, e := m.Call(ctx, "example", "storage", r)
	if e != nil || !bytes.Equal(out, r.Input) {
		t.Fatalf("storage: %s %v", out, e)
	}
	if e = m.InstallPackage(ctx, bytes.NewReader(v1)); !errors.Is(e, plugins.ErrBusy) {
		t.Fatalf("live update accepted: %v", e)
	}
	if e = m.Disable(ctx, "example"); e != nil {
		t.Fatal(e)
	}
	v2 := archiveExternal(t, externalManifest("2.0.0", 2), buildExternal(t, "2.0.0"), "")
	if e = m.InstallPackage(ctx, bytes.NewReader(v2)); e != nil {
		t.Fatal(e)
	}
	if got := m.List()[0]; got.Manifest.Version != "2.0.0" || !got.Rollback || len(got.Settings.Grants) != 0 || got.Settings.DataVersion != 2 {
		t.Fatalf("update: %+v", got)
	}
	value, e := d.PluginValue(ctx, "example", "schema")
	if e != nil || string(value) != "2" {
		t.Fatalf("migration: %s %v", value, e)
	}
	v3 := archiveExternal(t, externalManifest("3.0.0", 3), buildExternal(t, "3.0.0"), "")
	if e = m.InstallPackage(ctx, bytes.NewReader(v3)); e == nil {
		t.Fatal("broken migration accepted")
	}
	if m.List()[0].Manifest.Version != "2.0.0" {
		t.Fatal("failed update changed current version")
	}
	if _, e = d.PluginValue(ctx, "example", "corrupt"); !errors.Is(e, plugins.ErrNotFound) {
		t.Fatal("failed migration changed data")
	}
	if e = m.RollbackPackage(ctx, "example"); e != nil {
		t.Fatal(e)
	}
	if _, e = d.PluginValue(ctx, "example", "schema"); !errors.Is(e, plugins.ErrNotFound) {
		t.Fatal("rollback did not restore data")
	}
	enable(t, m, "example")
	if e = m.Close(ctx); e != nil {
		t.Fatal(e)
	}
	reopened := manager(t, d)
	if e = reopened.OpenPackages(ctx, root); e != nil {
		t.Fatal(e)
	}
	if reopened.List()[0].Status != "active" {
		t.Fatalf("restart: %+v", reopened.List())
	}
	if e = reopened.Disable(ctx, "example"); e != nil {
		t.Fatal(e)
	}
	if e = reopened.RemovePackage(ctx, "example", false); e != nil {
		t.Fatal(e)
	}
	if len(reopened.List()) != 0 {
		t.Fatal("remove retained catalog")
	}
	if _, e = d.PluginValue(ctx, "example", "example"); e != nil {
		t.Fatal("remove lost retained data")
	}
	if e = reopened.InstallPackage(ctx, bytes.NewReader(v1)); e != nil {
		t.Fatal("reinstall:", e)
	}
	if e = reopened.RemovePackage(ctx, "example", true); e != nil {
		t.Fatal(e)
	}
	if _, e = d.PluginValue(ctx, "example", "example"); !errors.Is(e, plugins.ErrNotFound) {
		t.Fatal("purge retained data")
	}
	if _, e = d.PluginRunByKey(ctx, "example", "missing"); !errors.Is(e, plugins.ErrNotFound) {
		t.Fatal(e)
	}
}
func TestExternalUntrustedOperations(t *testing.T) {
	if runtime.GOOS != "linux" || (runtime.GOARCH != "arm64" && runtime.GOARCH != "amd64") {
		t.Skip("external sandbox is supported on Linux arm64/amd64")
	}
	d := database(t)
	m := manager(t, d)
	if e := m.OpenPackages(ctx, t.TempDir()); e != nil {
		t.Fatal(e)
	}
	binary := buildExternal(t, "1.0.0")
	manifest := externalManifest("1.0.0", 1)
	archive := archiveExternal(t, manifest, binary, "")
	for _, bad := range [][]byte{[]byte("invalid zip"), archiveExternal(t, manifest, binary, "../escape"), archiveExternal(t, manifest, []byte("not ELF"), "")} {
		if e := m.InstallPackage(ctx, bytes.NewReader(bad)); e == nil {
			t.Fatal("invalid package accepted")
		}
	}
	mismatch := manifest
	mismatch.Version = "9.0.0"
	if e := m.InstallPackage(ctx, bytes.NewReader(archiveExternal(t, mismatch, binary, ""))); e == nil {
		t.Fatal("handshake mismatch accepted")
	}
	if e := m.InstallPackage(ctx, bytes.NewReader(archive)); e != nil {
		t.Fatal(e)
	}
	t.Setenv("TALK_PLUGIN_TEST_SECRET", "never forward")
	enable(t, m, "example")
	request := plugins.Request{Input: json.RawMessage(`{}`)}
	for _, op := range []string{"isolation", "spawn", "secret"} {
		out, e := m.Call(ctx, "example", op, request)
		if e != nil {
			t.Fatal(op, e)
		}
		switch op {
		case "isolation":
			if !strings.Contains(string(out), `"file_denied":true`) || !strings.Contains(string(out), "operation not permitted") {
				t.Fatal(string(out))
			}
		case "spawn":
			if string(out) != `{"denied":true}` {
				t.Fatal(string(out))
			}
		case "secret":
			if string(out) != `{"secret":""}` {
				t.Fatal(string(out))
			}
		}
	}
	for _, op := range []string{"hang", "crash", "oversize"} {
		started := time.Now()
		_, e := m.Call(ctx, "example", op, request)
		if e == nil {
			t.Fatal(op, "unexpected success")
		}
		if time.Since(started) > 3*time.Second {
			t.Fatal("unbounded operation")
		}
		deadline := time.Now().Add(time.Second)
		for m.List()[0].Status != "failed" && time.Now().Before(deadline) {
			time.Sleep(time.Millisecond)
		}
		if m.List()[0].Status != "failed" {
			t.Fatal("dead process remained active")
		}
		if e = m.Disable(ctx, "example"); e != nil {
			t.Fatal(e)
		}
		enable(t, m, "example")
	}
	// A separate pending call is forcibly stopped by disable.
	done := make(chan error, 1)
	go func() { _, e := m.Call(context.Background(), "example", "hang", request); done <- e }()
	deadline := time.Now().Add(time.Second)
	for m.List()[0].Active == 0 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if e := m.Disable(ctx, "example"); e != nil {
		t.Fatal(e)
	}
	if e := <-done; e == nil {
		t.Fatal("disable did not cancel operation")
	}
}

type failingPackages struct {
	*db.DB
	fail bool
}

func (s *failingPackages) CommitPluginPackage(ctx context.Context, id string, p *plugins.PackageRecord, settings plugins.Settings, values map[string]json.RawMessage, purge bool) error {
	if s.fail {
		return errors.New("simulated catalog write failure")
	}
	return s.DB.CommitPluginPackage(ctx, id, p, settings, values, purge)
}
func TestPackageCommitFailureAndChecksum(t *testing.T) {
	if runtime.GOOS != "linux" || (runtime.GOARCH != "arm64" && runtime.GOARCH != "amd64") {
		t.Skip("external sandbox is supported on Linux arm64/amd64")
	}
	d := database(t)
	store := &failingPackages{DB: d}
	m, err := plugins.New(ctx, store, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close(ctx)
	root := t.TempDir()
	if err = m.OpenPackages(ctx, root); err != nil {
		t.Fatal(err)
	}
	binary := buildExternal(t, "1.0.0")
	manifest := externalManifest("1.0.0", 1)
	// Remove storage authority while retaining the operation that requests it.
	manifest.Permissions = []string{"tools", "jobs"}
	packageBytes := archiveExternal(t, manifest, binary, "")
	store.fail = true
	if err = m.InstallPackage(ctx, bytes.NewReader(packageBytes)); err == nil {
		t.Fatal("commit failure ignored")
	}
	if len(m.List()) != 0 {
		t.Fatal("failed commit registered plugin")
	}
	records, err := d.PluginPackages(ctx)
	if err != nil || len(records) != 0 {
		t.Fatal("failed commit persisted catalog")
	}
	store.fail = false
	if err = m.InstallPackage(ctx, bytes.NewReader(packageBytes)); err != nil {
		t.Fatal(err)
	}
	enable(t, m, "example")
	if _, err = m.Call(ctx, "example", "storage", plugins.Request{Input: json.RawMessage(`{}`)}); err == nil {
		t.Fatal("external process bypassed storage grant")
	}
	if err = m.Disable(ctx, "example"); err != nil {
		t.Fatal(err)
	}
	records, _ = d.PluginPackages(ctx)
	executable := filepath.Join(root, "example", records[0].Digest, "plugin")
	if err = os.Chmod(executable, 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(executable, []byte("corrupted"), 0600); err != nil {
		t.Fatal(err)
	}
	if err = m.Enable(ctx, "example"); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Fatalf("tampered executable was not rejected: %v", err)
	}
}
