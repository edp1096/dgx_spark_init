package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestStorageStatusAndManualCleanupProxy(t *testing.T) {
	media := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/media/storage":
			writeJSON(w, http.StatusOK, mediaStorageStatus{TemporaryDirectories: 3, TemporaryBytes: 100, ReclaimableDirectories: 2, ReclaimableBytes: 80})
		case r.Method == http.MethodDelete && r.URL.Path == "/v1/media/storage/temp":
			writeJSON(w, http.StatusOK, mediaCleanupResult{RemovedDirectories: 2, RemovedBytes: 80})
		default:
			http.NotFound(w, r)
		}
	}))
	defer media.Close()
	store, err := jobs.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"media": {Endpoint: media.URL}}}, store, nil).Handler()

	statusResponse := httptest.NewRecorder()
	handler.ServeHTTP(statusResponse, httptest.NewRequest(http.MethodGet, "/api/storage", nil))
	if statusResponse.Code != http.StatusOK || !strings.Contains(statusResponse.Body.String(), `"reclaimable_bytes":80`) {
		t.Fatalf("storage status = %d %s", statusResponse.Code, statusResponse.Body.String())
	}
	cleanupResponse := httptest.NewRecorder()
	handler.ServeHTTP(cleanupResponse, httptest.NewRequest(http.MethodDelete, "/api/storage/temp", nil))
	if cleanupResponse.Code != http.StatusOK {
		t.Fatalf("cleanup = %d %s", cleanupResponse.Code, cleanupResponse.Body.String())
	}
	var result mediaCleanupResult
	if err := json.Unmarshal(cleanupResponse.Body.Bytes(), &result); err != nil || result.RemovedBytes != 80 {
		t.Fatalf("cleanup result = %#v, %v", result, err)
	}
}

func TestStartupCleanupUsesRetentionSetting(t *testing.T) {
	requestedQuery := ""
	media := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestedQuery = r.URL.RawQuery
		writeJSON(w, http.StatusOK, mediaCleanupResult{})
	}))
	defer media.Close()
	store, err := jobs.New(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	s := New(config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"media": {Endpoint: media.URL}},
		Storage: config.Storage{CleanupOnStartup: true, TempRetentionHours: 48},
	}, store, nil)
	if _, err := s.CleanupStaleMediaTemp(); err != nil {
		t.Fatal(err)
	}
	if requestedQuery != "older_than_hours=48" {
		t.Fatalf("query = %q", requestedQuery)
	}
}
