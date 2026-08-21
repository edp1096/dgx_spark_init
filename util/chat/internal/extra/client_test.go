package extra

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestExecuteStreamsRawOutput(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		fmt.Fprintln(w, `{"type":"stdout","data":"hello\n"}`)
		fmt.Fprintln(w, `{"type":"stderr","data":"warning\n"}`)
		fmt.Fprintln(w, `{"type":"exit","exit_code":0,"duration_ms":12}`)
	}))
	defer server.Close()
	result, err := New(server.URL).Execute(context.Background(), ExecRequest{Command: "test"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.Stdout != "hello\n" || result.Stderr != "warning\n" || result.ExitCode != 0 || result.DurationMS != 12 {
		t.Fatalf("unexpected result: %+v", result)
	}
}
