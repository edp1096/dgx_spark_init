package main

import (
	"bytes"
	"encoding/json"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"
)

func TestLongCommandTransport(t *testing.T) {
	for _, size := range []int{9000, 150000, 900000} {
		payload := strings.Repeat("한글~'\"\\$()", size/20)
		script := "cat <<'PAYLOAD'\n" + payload + "\nPAYLOAD\nread ignored\nprintf '\\nDONE\\n'\nexit 7\n"
		command, input := commandTransport(script)
		cmd := exec.Command("/bin/sh", "-c", command)
		temp := t.TempDir()
		cmd.Env = append(os.Environ(), "TMPDIR="+temp, "SHELL=/bin/bash")
		cmd.Stdin = input
		got, err := cmd.CombinedOutput()
		if err == nil || cmd.ProcessState.ExitCode() != 7 {
			t.Fatalf("exit: %v", err)
		}
		if string(got) != payload+"\n\nDONE\n" {
			t.Fatalf("output mismatch size=%d", size)
		}
		entries, _ := os.ReadDir(temp)
		if len(entries) != 0 {
			t.Fatal("temporary script leaked")
		}
	}
}
func TestCommandJSONLimits(t *testing.T) {
	body, _ := json.Marshal(execRequest{Command: strings.Repeat("\t", maxCommandBytes)})
	var req execRequest
	if err := decodeJSONLimit(httptest.NewRecorder(), httptest.NewRequest("POST", "/", bytes.NewReader(body)), &req, 6*maxCommandBytes+65536); err != nil {
		t.Fatal(err)
	}
	for _, command := range []string{strings.Repeat("x", maxCommandBytes+1), "echo\x00bad", "  "} {
		body, _ = json.Marshal(execRequest{Command: command})
		response := httptest.NewRecorder()
		newAPI(config{}).execute(response, httptest.NewRequest("POST", "/", bytes.NewReader(body)))
		if response.Code != 400 {
			t.Fatalf("status %d", response.Code)
		}
	}
}
