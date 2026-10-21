package orchestrator

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestQwen38FNEXL3VerifiesTensorFoldCapacityAndRuntimeProfile(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "docker-call")
	t.Setenv("NATIVE_TEST_LOG", log)
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$NATIVE_TEST_LOG\"\nprintf '%s' \"$NATIVE_TEST_PROFILE\"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	good := `{"model":"qwen38fn_exl3","context":1048576,"kv":"int8","depth":5,"confidence":0.4,"draft_vocabulary":86218,"expert_tiles":"compact-parallel-v1","moe_window":2048,"vision":true}`
	for _, tc := range []struct {
		name     string
		context  int
		profile  string
		accepted bool
	}{
		{"undersized-context", 262144, good, false},
		{"qualified", 1048576, good, true},
		{"old-runtime", 1048576, `{"model":"qwen38fn_exl3","context":1048576,"kv":"int8","depth":3,"vision":true}`, false},
		{"wrong-confidence", 1048576, strings.Replace(good, `"confidence":0.4`, `"confidence":0`, 1), false},
		{"wrong-vocabulary", 1048576, strings.Replace(good, `86218`, `65536`, 1), false},
		{"missing-kernel-profile", 1048576, strings.Replace(good, `compact-parallel-v1`, `none`, 1), false},
		{"wrong-workspace", 1048576, strings.Replace(good, `"moe_window":2048`, `"moe_window":1024`, 1), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			os.Remove(log)
			t.Setenv("NATIVE_TEST_PROFILE", tc.profile)
			api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/health" {
					t.Error(r.URL.Path)
				}
				fmt.Fprintf(w, `{"ok":true,"backend":"tensorfold","context_length":%d}`, tc.context)
			}))
			defer api.Close()
			controller, _ := NewController()
			component := Component{ID: "qwen38fn_exl3", Endpoint: api.URL, Model: "qwen38fn_exl3", Container: "native-test"}
			err := controller.prepareQwen38FNEXL3Headroom(context.Background(), component)
			if (err == nil) != tc.accepted {
				t.Fatalf("accepted=%v error=%v", tc.accepted, err)
			}
			call, readErr := os.ReadFile(log)
			if tc.context < 1048576 {
				if !os.IsNotExist(readErr) {
					t.Fatal("profile checked before actual capacity")
				}
			} else if readErr != nil || strings.TrimSpace(string(call)) != "exec native-test cat /runtime/runtime-profile.json" {
				t.Fatalf("unexpected check: %s %v", call, readErr)
			}
		})
	}
}
