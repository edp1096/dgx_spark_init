package plugins

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

type testHost struct{ ready chan struct{} }

func (h testHost) Ready() <-chan struct{}                               { return h.ready }
func (h testHost) Config() json.RawMessage                              { return json.RawMessage(`{}`) }
func (h testHost) Get(context.Context, string) (json.RawMessage, error) { return nil, ErrPermission }
func (h testHost) Put(context.Context, string, json.RawMessage) error   { return ErrPermission }
func (h testHost) Delete(context.Context, string) error                 { return ErrPermission }
func (h testHost) Call(context.Context, string, Request) (json.RawMessage, error) {
	return nil, ErrPermission
}
func (h testHost) Submit(context.Context, string, string, Request) (Run, error) {
	return Run{}, ErrPermission
}
func (h testHost) Runs(context.Context) ([]Run, error)  { return nil, ErrPermission }
func (h testHost) Cancel(context.Context, string) error { return ErrPermission }
func TestExternalSandbox(t *testing.T) {
	if runtime.GOOS != "linux" || (runtime.GOARCH != "arm64" && runtime.GOARCH != "amd64") {
		t.Skip("external sandbox is supported on Linux arm64/amd64")
	}
	dir := t.TempDir()
	binary := filepath.Join(dir, "plugin")
	cmd := exec.Command("go", "build", "-o", binary, "./testdata/fixture")
	cmd.Env = append(os.Environ(), "CGO_ENABLED=0")
	if b, e := cmd.CombinedOutput(); e != nil {
		t.Fatalf("build %v %s", e, b)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	h := testHost{make(chan struct{})}
	close(h.ready)
	p := &processPlugin{manifest: Manifest{ID: "example", Version: "1.0.0"}, executable: binary}
	if e := p.start(ctx, h); e != nil {
		t.Fatal(e)
	}
	defer p.close(ctx)
	out, e := p.handle(ctx, h, "echo", Request{Input: json.RawMessage(`{"hello":true}`)})
	if e != nil {
		t.Fatal(e)
	}
	t.Log(string(out))
	out, e = p.handle(ctx, h, "isolation", Request{Input: json.RawMessage(`{}`)})
	if e != nil {
		t.Fatal(e)
	}
	t.Log(string(out))
	var v struct {
		File    bool   `json:"file_denied"`
		Network string `json:"network_error"`
	}
	json.Unmarshal(out, &v)
	if !v.File || v.Network == "" {
		t.Fatal("isolation failed")
	}
}
