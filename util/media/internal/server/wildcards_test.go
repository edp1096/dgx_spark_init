package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestRandomPromptWildcardUsesNoCameraMuseAndStyle(t *testing.T) {
	dataDir := t.TempDir()
	wildcardDir := filepath.Join(dataDir, "prompt-wildcards", "crocody-mymuse")
	if err := os.MkdirAll(wildcardDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wildcardDir, "muse_no_camera.txt"), []byte("a woman beside a window\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wildcardDir, "Style.txt"), []byte("casual smartphone snapshot\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: dataDir}, store, nil).Handler()
	request := httptest.NewRequest(http.MethodGet, "/api/prompts/wildcard", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	var result wildcardPromptResult
	if err := json.NewDecoder(response.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.MuseVariant != "muse(no_camera).txt" || result.MuseIndex != 1 || result.StyleIndex != 1 {
		t.Fatalf("unexpected metadata: %#v", result)
	}
	if result.Prompt != "a woman beside a window casual smartphone snapshot" {
		t.Fatalf("unexpected prompt: %q", result.Prompt)
	}
	if strings.Contains(result.Prompt, "Sony A7R") {
		t.Fatalf("camera-locked muse was used: %q", result.Prompt)
	}
}

func TestReadWildcardLinesSkipsEmptyLinesAndBOM(t *testing.T) {
	path := filepath.Join(t.TempDir(), "wildcards.txt")
	if err := os.WriteFile(path, []byte("\ufefffirst\n\n second \n"), 0o644); err != nil {
		t.Fatal(err)
	}
	lines, err := readWildcardLines(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(lines) != 2 || lines[0] != "first" || lines[1] != "second" {
		t.Fatalf("unexpected lines: %#v", lines)
	}
}
