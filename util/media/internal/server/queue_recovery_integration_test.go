package server

import (
	"encoding/json"
	"fmt"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRestartRecoveryFailsJobsWithoutRecoverableInputs(t *testing.T) {
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, job := range []jobs.Job{
		{ID: "image-job", Kind: "image", Status: "running", Params: map[string]any{}, CreatedAt: time.Now()},
		{ID: "missing-file", Kind: "recognition", Status: "running", Params: map[string]any{"source": "file"}, CreatedAt: time.Now()},
	} {
		if err := store.Save(job); err != nil {
			t.Fatal(err)
		}
	}

	mediaServer := New(config.Config{DataDir: dataDir}, store, nil)
	resumed, failed := mediaServer.ResumeInterruptedJobs()
	if resumed != 0 || failed != 2 {
		t.Fatalf("resumed=%d failed=%d", resumed, failed)
	}
	for _, job := range store.List() {
		if job.Status != "failed" || job.Error == "" {
			t.Fatalf("job was not reconciled after restart: %#v", job)
		}
	}
}

func TestGenerationQueueRunsFIFOAndContinuesAfterFailure(t *testing.T) {
	calls := make(chan string, 2)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		_ = json.NewDecoder(r.Body).Decode(&request)
		input, _ := request["input"].(string)
		calls <- input
		if input == "first" {
			http.Error(w, "intentional failure", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("RIFF-test-wave"))
	}))
	defer worker.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	base := time.Now()
	for index, prompt := range []string{"first", "second"} {
		job := jobs.Job{
			ID: fmt.Sprintf("speech-%d", index), Kind: "speech", Status: "queued", Prompt: prompt,
			Params:    map[string]any{"language": "Korean", "speaker": "Sohee", "queued_at": base.Add(time.Duration(index) * time.Millisecond).Format(time.RFC3339Nano)},
			CreatedAt: base.Add(time.Duration(index) * time.Millisecond),
		}
		if err := store.Save(job); err != nil {
			t.Fatal(err)
		}
	}
	mediaServer := New(config.Config{DataDir: dataDir, Engines: map[string]config.Engine{"speech": {Endpoint: worker.URL}}}, store, nil)
	mediaServer.wakeGenerationQueue()

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		first, _ := store.Get("speech-0")
		second, _ := store.Get("speech-1")
		if first.Status == "failed" && second.Status == "completed" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	first, _ := store.Get("speech-0")
	second, _ := store.Get("speech-1")
	if first.Status != "failed" || second.Status != "completed" {
		t.Fatalf("queue did not continue after failure: first=%s second=%s", first.Status, second.Status)
	}
	if got := <-calls; got != "first" {
		t.Fatalf("first call=%q", got)
	}
	if got := <-calls; got != "second" {
		t.Fatalf("second call=%q", got)
	}
}

func TestRestartRecoveryCancelsActiveMediaPreparationBeforeResume(t *testing.T) {
	cancelled := make(chan string, 1)
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/v1/media/prepare/") {
			cancelled <- strings.TrimPrefix(r.URL.Path, "/v1/media/prepare/")
			w.WriteHeader(http.StatusAccepted)
			return
		}
		http.NotFound(w, r)
	}))
	defer mediaWorker.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, job := range []jobs.Job{
		{ID: "running-subtitle", Kind: "recognition", Status: "running", CreatedAt: time.Now()},
		{ID: "completed-subtitle", Kind: "recognition", Status: "completed", CreatedAt: time.Now()},
		{ID: "running-image", Kind: "image", Status: "running", CreatedAt: time.Now()},
	} {
		if err := store.Save(job); err != nil {
			t.Fatal(err)
		}
	}

	mediaServer := New(config.Config{
		DataDir: dataDir,
		Engines: map[string]config.Engine{"media": {Endpoint: mediaWorker.URL}},
	}, store, nil)
	if count := mediaServer.CancelActiveMediaPreparations(); count != 1 {
		t.Fatalf("cancelled=%d want=1", count)
	}
	requeued, ok := store.Get("running-subtitle")
	if !ok || requeued.Status != "queued" || requeued.Params["stage"] != "queued" {
		t.Fatalf("running subtitle was not durably requeued: %#v", requeued)
	}
	select {
	case id := <-cancelled:
		if id != "running-subtitle" {
			t.Fatalf("cancelled id=%q", id)
		}
	case <-time.After(time.Second):
		t.Fatal("stale media preparation cancellation was not sent")
	}
}

func TestCancelSubtitleJobStopsMediaPreparation(t *testing.T) {
	cancelled := make(chan string, 1)
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/v1/media/prepare/") {
			cancelled <- strings.TrimPrefix(r.URL.Path, "/v1/media/prepare/")
			_ = json.NewEncoder(w).Encode(map[string]string{"status": "cancelling"})
			return
		}
		http.NotFound(w, r)
	}))
	defer mediaWorker.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	job := jobs.Job{
		ID: "subtitle-cancel-test", Kind: "recognition", Status: "running",
		Params: map[string]any{"stage": "media", "media_eta_seconds": 30}, CreatedAt: time.Now(),
	}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{
		DataDir: dataDir,
		Engines: map[string]config.Engine{"media": {Endpoint: mediaWorker.URL}},
	}, store, nil).Handler()

	request := httptest.NewRequest(http.MethodPost, "/api/jobs/"+job.ID+"/cancel", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	select {
	case id := <-cancelled:
		if id != job.ID {
			t.Fatalf("cancelled id=%q", id)
		}
	case <-time.After(time.Second):
		t.Fatal("media cancellation request was not sent")
	}
	persisted, ok := store.Get(job.ID)
	if !ok || persisted.Status != "cancelled" || persisted.Params["stage"] != "cancelled" {
		t.Fatalf("job was not cancelled: %#v", persisted)
	}
	if _, ok := persisted.Params["media_eta_seconds"]; ok {
		t.Fatalf("stale ETA remained after cancellation: %#v", persisted.Params)
	}
}

func TestCancelRunningGenerationInvokesRequestCancellation(t *testing.T) {
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	job := jobs.Job{
		ID: "running-image-cancel-test", Kind: "image", Status: "running",
		Params: map[string]any{"stage": "running"}, CreatedAt: time.Now(),
	}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	server := New(config.Config{DataDir: dataDir}, store, nil)
	cancelled := make(chan struct{}, 1)
	server.generationCancels[job.ID] = func() { cancelled <- struct{}{} }

	request := httptest.NewRequest(http.MethodPost, "/api/jobs/"+job.ID+"/cancel", nil)
	response := httptest.NewRecorder()
	server.Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("running generation request was not cancelled")
	}
	persisted, ok := store.Get(job.ID)
	if !ok || persisted.Status != "cancelled" || persisted.Params["stage"] != "cancelled" {
		t.Fatalf("job was not cancelled: %#v", persisted)
	}
}
