package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestGenerateListAndDeleteKey(t *testing.T) {
	dir := t.TempDir()
	a := newAPI(config{KeyDir: dir, KnownHostsPath: filepath.Join(dir, "known_hosts"), MaxConcurrency: 1, MaxOutputBytes: 1024, CommandTimeout: time.Second})
	handler := a.routes()

	request := httptest.NewRequest(http.MethodPost, "/v1/ssh/keys/generate", bytes.NewBufferString(`{"key_id":"dgx-main"}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("generate status=%d body=%s", response.Code, response.Body.String())
	}
	var generated storedKey
	if err := json.Unmarshal(response.Body.Bytes(), &generated); err != nil {
		t.Fatal(err)
	}
	if generated.ID != "dgx-main" || generated.Fingerprint == "" || generated.PublicKey == "" {
		t.Fatalf("unexpected key metadata: %+v", generated)
	}
	info, err := os.Stat(filepath.Join(dir, "dgx-main"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("private key permissions=%v", info.Mode().Perm())
	}

	response = httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/v1/ssh/keys", nil))
	if response.Code != http.StatusOK || !bytes.Contains(response.Body.Bytes(), []byte(`"id":"dgx-main"`)) {
		t.Fatalf("list status=%d body=%s", response.Code, response.Body.String())
	}

	response = httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodDelete, "/v1/ssh/keys/dgx-main", nil))
	if response.Code != http.StatusNoContent {
		t.Fatalf("delete status=%d body=%s", response.Code, response.Body.String())
	}
	if _, err := os.Stat(filepath.Join(dir, "dgx-main")); !os.IsNotExist(err) {
		t.Fatalf("expected deleted key, err=%v", err)
	}
}
