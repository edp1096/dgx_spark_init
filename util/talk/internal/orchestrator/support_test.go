package orchestrator

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func waitSupportOperation(t *testing.T, c *Controller) {
	t.Helper()
	for i := 0; i < 100; i++ {
		op := c.Operation()
		if op.State != "running" {
			if op.State != "complete" {
				t.Fatalf("operation: %+v", op)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("operation timed out")
}
func fakeSupportDocker(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	t.Setenv("MOCK_LOG", log)
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	body := `#!/bin/sh
printf '%s\n' "$*" >> "$MOCK_LOG"
case "$*" in
 "image inspect "*) exit 0;;
 *" config") cat;;
 *inspect*) printf 'running';;
esac
`
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(body), 0700); err != nil {
		t.Fatal(err)
	}
	return log
}
func TestModelStopLeavesSharedServicesRunning(t *testing.T) {
	log := fakeSupportDocker(t)
	cat, _ := LoadCatalog()
	c, _ := NewControllerWithCatalog(cat)
	if err := c.StopBundle("flash-next"); err != nil {
		t.Fatal(err)
	}
	waitSupportOperation(t, c)
	raw, _ := os.ReadFile(log)
	if !strings.Contains(string(raw), "sglang-qwen38-fn") {
		t.Fatal("model was not stopped")
	}
	if strings.Contains(string(raw), "sparktalk-extra-") {
		t.Fatalf("shared service touched: %s", raw)
	}
}
func TestPreparingDocumentImageDoesNotRestartRunningService(t *testing.T) {
	log := fakeSupportDocker(t)
	c, _ := NewController()
	dir := t.TempDir()
	c.ConfigurePaths(dir, filepath.Join(dir, "models"))
	if err := c.ComponentAction("extra-documents", "prepare", "flash-next"); err != nil {
		t.Fatal(err)
	}
	waitSupportOperation(t, c)
	raw, _ := os.ReadFile(log)
	for _, action := range []string{"stop ", "rm ", " up -d"} {
		if strings.Contains(string(raw), action) {
			t.Fatalf("prepare mutated service: %s", raw)
		}
	}
	if _, err := os.Stat(filepath.Join(dir, "runtime/extra-documents/build/calc/go.mod")); err != nil {
		t.Fatal("Go build inputs missing", err)
	}
}
func TestSupportInventoryIncludesGlobalDocumentsAndSetBindings(t *testing.T) {
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	defer api.Close()
	cat, _ := LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].IsSupport() {
			cat.Components[i].Controller = "external"
			cat.Components[i].HealthURL = api.URL
		}
	}
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "glm53-worker-extra" {
			for _, id := range []string{"extra-media", "extra-collector", "extra-ssh"} {
				d := cat.Bundles[i].Bindings[id]
				d.Controller = pointer("external")
				d.HealthURL = pointer(api.URL)
				cat.Bundles[i].Bindings[id] = d
			}
		}
	}
	c, e := NewControllerWithCatalog(cat)
	if e != nil {
		t.Fatal(e)
	}
	rows := c.SupportSnapshot(context.Background(), "glm53-worker-extra")
	if len(rows) != 4 {
		t.Fatalf("support count=%d", len(rows))
	}
	for _, r := range rows {
		if r.Health != "online" {
			t.Fatalf("%+v", r)
		}
		if r.Key == "collector" && r.Host != "worker" {
			t.Fatal("worker binding lost")
		}
		if r.Key == "documents" && r.Host != "local" {
			t.Fatal("global document binding lost")
		}
	}
}
