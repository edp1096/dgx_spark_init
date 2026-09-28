package orchestrator

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func TestQADCapacityRejectsSilentKVTruncation(t *testing.T) {
	for _, capacity := range []int{844992, 1048576} {
		t.Run(fmt.Sprint(capacity), func(t *testing.T) {
			api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/server_info" {
					t.Error(r.URL.Path)
				}
				fmt.Fprintf(w, `{"context_length":1048576,"max_total_num_tokens":%d}`, capacity)
			}))
			defer api.Close()
			c, _ := NewController()
			component := Component{ComposeAsset: "compose.flash-next.yaml", HealthURL: api.URL + "/health"}
			err := c.checkSGLangCapacity(context.Background(), component)
			if capacity < 1048576 {
				if err == nil || !strings.Contains(err.Error(), "844992") {
					t.Fatal(err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestGLMSGLangCapacityAndVLLMBypass(t *testing.T) {
	for _, capacity := range []int{433792, 1048576} {
		var calls atomic.Int64
		api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			calls.Add(1)
			fmt.Fprintf(w, `{"context_length":1048576,"max_total_num_tokens":%d}`, capacity)
		}))
		c, _ := NewController()
		component := Component{ID: "glm53", Controller: "glm53-cluster", ProgressKind: "sglang", HealthURL: api.URL + "/health"}
		err := c.checkSGLangCapacity(context.Background(), component)
		if capacity < 1048576 {
			if err == nil || !strings.Contains(err.Error(), "GLM") || !strings.Contains(err.Error(), "433792") {
				t.Fatalf("undersized GLM accepted: %v", err)
			}
		} else if err != nil {
			t.Fatal(err)
		}
		component.ProgressKind = "service"
		if err := c.checkSGLangCapacity(context.Background(), component); err != nil || calls.Load() != 1 {
			t.Fatalf("vLLM queried for SGLang capacity: calls=%d err=%v", calls.Load(), err)
		}
		api.Close()
	}
}

func TestHealthyButUndersizedQADRequiresRestart(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "stops")
	t.Setenv("KV_STOP_LOG", logPath)
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	script := "#!/bin/sh\ncase \"$1\" in inspect) case \"$*\" in *\"|\"*) printf 'running|1';; *) printf running;; esac;; stop) echo stopped > \"$KV_STOP_LOG\"; echo simulated-stop-error >&2; exit 1;; esac\n"
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	var capacity atomic.Int64
	capacity.Store(1013056)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/server_info" {
			fmt.Fprintf(w, `{"context_length":1048576,"max_total_num_tokens":%d}`, capacity.Load())
		}
	}))
	defer server.Close()
	c := newController(Catalog{Hosts: map[string]Host{"local": {}}})
	component := Component{ID: "flash-next", Host: "local", Container: "test-qad", ComposeAsset: "compose.flash-next.yaml", HealthURL: server.URL + "/health"}
	if !c.componentNeedsStart(context.Background(), component) {
		t.Fatal("HTTP health masked undersized KV")
	}
	if err := c.startAndWait(component); err == nil || !strings.Contains(err.Error(), "simulated-stop-error") {
		t.Fatalf("did not attempt restart: %v", err)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Fatal("old container was not stopped", err)
	}
	capacity.Store(1048576)
	if c.componentNeedsStart(context.Background(), component) {
		t.Fatal("adequate healthy KV must not restart")
	}
}

func TestHealthProbeRejectsHTTPHealthyButUndersizedKV(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/server_info" {
			fmt.Fprint(w, `{"context_length":1048576,"max_total_num_tokens":758656}`)
		}
	}))
	defer server.Close()
	c, _ := NewController()
	_, err := c.probeHealth(context.Background(), Component{ComposeAsset: "compose.flash-next.yaml", HealthURL: server.URL + "/health"})
	if err == nil || !strings.Contains(err.Error(), "758656") {
		t.Fatalf("undersized online: %v", err)
	}
}
