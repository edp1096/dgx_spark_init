package server

import (
	"encoding/json"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestJobTagAPIUpdatesAndListsCatalog(t *testing.T) {
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Save(jobs.Job{ID: "image-1", Kind: "image", Status: "completed", CreatedAt: time.Now()}); err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: dataDir}, store, nil).Handler()

	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPut, "/api/jobs/image-1/tags", strings.NewReader(`{"tags":[" Portrait ","야간","portrait"]}`))
	request.Header.Set("Content-Type", "application/json")
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("update tags = %d %s", response.Code, response.Body.String())
	}
	var updated jobs.Job
	if err := json.Unmarshal(response.Body.Bytes(), &updated); err != nil {
		t.Fatal(err)
	}
	if len(updated.Tags) != 2 || updated.Tags[0] != "Portrait" || updated.Tags[1] != "야간" {
		t.Fatalf("updated tags = %#v", updated.Tags)
	}

	response = httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/tags", nil))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"name":"Portrait"`) || !strings.Contains(response.Body.String(), `"count":1`) {
		t.Fatalf("tag catalog = %d %s", response.Code, response.Body.String())
	}
}
