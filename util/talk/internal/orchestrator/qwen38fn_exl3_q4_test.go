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

func TestQwen38FNEXL3Q4VerifiesCapacityBeforeClosedFileCacheAdvice(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "docker-call")
	t.Setenv("NATIVE_TEST_LOG", log)
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$NATIVE_TEST_LOG\"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	for _, cache := range []int{262144, 1048576} {
		api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/v1/models" {
				t.Error(r.URL.Path)
			}
			fmt.Fprintf(w, `{"data":[{"id":"qwen38fn_exl3_q4","max_model_len":%d}]}`, cache)
		}))
		controller, _ := NewController()
		component := Component{ID: "qwen38fn_exl3_q4", Endpoint: api.URL, Model: "qwen38fn_exl3_q4", Container: "native-test"}
		err := controller.prepareQwen38FNEXL3Q4Headroom(context.Background(), component)
		api.Close()
		if cache < 1048576 {
			if err == nil {
				t.Fatal("undersized actual cache accepted")
			}
			if _, err := os.Stat(log); !os.IsNotExist(err) {
				t.Fatal("cache advice ran before actual capacity was qualified")
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
			call, _ := os.ReadFile(log)
			if strings.TrimSpace(string(call)) != "exec native-test python /opt/velogb10/release_weight_cache.py" {
				t.Fatalf("unexpected cache advice: %s", call)
			}
		}
	}
}
