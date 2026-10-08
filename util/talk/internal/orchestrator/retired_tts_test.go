package orchestrator

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRetiredTTSDrainsOwnedImageAndProtectsBusyRuntime(t *testing.T) {
	for _, scenario := range []string{"old-image", "idle", "busy", "foreign-image", "absent", "absent-lowercase-object", "absent-lowercase-container", "inspect-error"} {
		t.Run(scenario, func(t *testing.T) {
			dir := t.TempDir()
			log := filepath.Join(dir, "calls")
			image := "sparktalk-magpie-tts:v2607-longform3-lifecycle"
			if scenario == "old-image" {
				image = "sparktalk-magpie-tts:v2607-longform2"
			}
			if scenario == "foreign-image" {
				image = "another-service:latest"
			}
			inspect := "printf '%s\\n' '" + image + "|running'"
			if scenario == "absent" {
				inspect = "echo 'No such object' >&2; exit 1"
			}
			if scenario == "absent-lowercase-object" {
				inspect = "echo 'error: no such object: sparktalk-magpie-tts' >&2; exit 1"
			}
			if scenario == "absent-lowercase-container" {
				inspect = "echo 'error: no such container: sparktalk-magpie-tts' >&2; exit 1"
			}
			if scenario == "inspect-error" {
				inspect = "echo 'permission denied' >&2; exit 1"
			}
			script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '" + log + "'\ncase \"$1\" in\ninspect) " + inspect + " ;;\nstop|rm) exit 0 ;;\n*) exit 2 ;;\nesac\n"
			if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0o755); err != nil {
				t.Fatal(err)
			}
			t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/runtime/quiesce" {
					t.Error("wrong retirement endpoint", r.URL.Path)
				}
				if scenario == "busy" {
					w.WriteHeader(409)
				}
				w.Write([]byte(`{"status":"ok"}`))
			}))
			defer server.Close()
			cat, _ := LoadCatalog()
			c := newController(cat)
			t.Cleanup(c.Close)
			x, _ := cat.Component("qwen3-tts")
			x.Endpoint = server.URL
			err := c.retireLegacyTTS(context.Background(), x)
			calls, _ := os.ReadFile(log)
			mutated := strings.Contains(string(calls), "stop -t 30 sparktalk-magpie-tts") && strings.Contains(string(calls), "rm sparktalk-magpie-tts")
			shouldStop := scenario == "idle" || scenario == "old-image"
			if mutated != shouldStop {
				t.Fatalf("unsafe or missing retirement %s: %s", scenario, calls)
			}
			shouldFail := scenario == "busy" || scenario == "foreign-image" || scenario == "inspect-error"
			if (err != nil) != shouldFail {
				t.Fatalf("unexpected retirement result: %v", err)
			}
		})
	}
}
