package server

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"fmt"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSubtitleQueueRunsCompletePipelinesInFIFOOrder(t *testing.T) {
	started := make(chan string, 2)
	releaseFirst := make(chan struct{})
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/media/prepare" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		sourceURL := r.FormValue("url")
		started <- sourceURL
		if strings.Contains(sourceURL, "first") {
			<-releaseFirst
		}
		w.Header().Set("Content-Type", "application/zip")
		archive := zip.NewWriter(w)
		manifest, _ := archive.Create("manifest.json")
		_, _ = manifest.Write([]byte(`{"source_name":"queued.mp4","segments":[{"name":"segment-00000.wav","start":0,"end":1,"duration":1}]}`))
		segment, _ := archive.Create("segment-00000.wav")
		_, _ = segment.Write([]byte("fake audio"))
		_ = archive.Close()
	}))
	defer mediaWorker.Close()

	asrWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{
			"text": "queued result", "language": "English",
			"timestamps": []map[string]any{{"text": "queued result", "start": 0.0, "end": 0.5}},
		})
	}))
	defer asrWorker.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	server := New(config.Config{
		DataDir: dataDir,
		Engines: map[string]config.Engine{
			"media": {Endpoint: mediaWorker.URL}, "recognition": {Endpoint: asrWorker.URL},
		},
		Recognition: config.Recognition{
			Model: "test-asr", MaxUploadMB: 1, SegmentSeconds: 30,
			DefaultLanguage: "English", DefaultOutputFormats: []string{"txt"}, DefaultTranslationMode: "none",
		},
	}, store, nil)
	handler := server.Handler()

	submit := func(sourceURL string) {
		t.Helper()
		var body bytes.Buffer
		form := multipart.NewWriter(&body)
		_ = form.WriteField("url", sourceURL)
		_ = form.WriteField("output_formats", "txt")
		_ = form.WriteField("translation_mode", "none")
		_ = form.Close()
		request := httptest.NewRequest(http.MethodPost, "/api/jobs/recognition", &body)
		request.Header.Set("Content-Type", form.FormDataContentType())
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusAccepted {
			t.Fatalf("submit status=%d body=%s", response.Code, response.Body.String())
		}
	}

	submit("https://example.com/first")
	select {
	case got := <-started:
		if got != "https://example.com/first" {
			t.Fatalf("first started=%q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("first queued subtitle did not start")
	}
	submit("https://example.com/second")
	select {
	case got := <-started:
		t.Fatalf("second pipeline overlapped first: %q", got)
	case <-time.After(150 * time.Millisecond):
	}
	close(releaseFirst)
	select {
	case got := <-started:
		if got != "https://example.com/second" {
			t.Fatalf("second started=%q", got)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("second queued subtitle did not start after first completed")
	}

	deadline := time.Now().Add(2 * time.Second)
	allCompleted := false
	for time.Now().Before(deadline) {
		completed := 0
		for _, job := range store.List() {
			if job.Status == "completed" {
				completed++
			}
		}
		if completed == 2 {
			allCompleted = true
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	if !allCompleted {
		t.Fatalf("queued subtitle jobs did not complete: %#v", store.List())
	}
	for _, job := range store.List() {
		if _, ok := job.Params["started_at"].(string); !ok {
			t.Fatalf("completed subtitle job has no execution timestamp: %#v", job.Params)
		}
		if _, ok := job.Params["stage_started_at"]; ok {
			t.Fatalf("completed subtitle job retained a stale stage timestamp: %#v", job.Params)
		}
	}
}

func TestMediaOptionsAndSubtitleSelectionAreForwarded(t *testing.T) {
	selectionReceived := make(chan struct{}, 1)
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/media/options":
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Fatal(err)
			}
			if r.FormValue("url") != "https://supjav.com/206680.html" {
				t.Fatalf("options url = %q", r.FormValue("url"))
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"site":"supjav.com","parts":[{"id":"1","label":"1","sources":[{"id":"ST","label":"ST"}]}]}`))
		case "/v1/media/prepare":
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Fatal(err)
			}
			if r.FormValue("url") != "https://supjav.com/206680.html" || r.FormValue("media_part") != "2" || r.FormValue("media_source") != "DS" {
				t.Fatalf("unexpected selection fields: %#v", r.MultipartForm.Value)
			}
			selectionReceived <- struct{}{}
			http.Error(w, "test stop", http.StatusUnprocessableEntity)
		default:
			http.NotFound(w, r)
		}
	}))
	defer mediaWorker.Close()

	cfg := config.Config{
		DataDir:     t.TempDir(),
		Engines:     map[string]config.Engine{"media": {Endpoint: mediaWorker.URL}},
		Recognition: config.Recognition{MaxUploadMB: 1, SegmentSeconds: 30, DefaultOutputFormats: []string{"txt"}, DefaultTranslationMode: "none"},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	optionsReq := httptest.NewRequest(http.MethodPost, "/api/media/options", strings.NewReader("url=https%3A%2F%2Fsupjav.com%2F206680.html"))
	optionsReq.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	optionsRes := httptest.NewRecorder()
	handler.ServeHTTP(optionsRes, optionsReq)
	if optionsRes.Code != http.StatusOK || !strings.Contains(optionsRes.Body.String(), `"site":"supjav.com"`) {
		t.Fatalf("options status=%d body=%s", optionsRes.Code, optionsRes.Body.String())
	}

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("url", "https://supjav.com/206680.html")
	_ = form.WriteField("media_part", "2")
	_ = form.WriteField("media_source", "DS")
	_ = form.WriteField("output_formats", "txt")
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/recognition", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	select {
	case <-selectionReceived:
	case <-time.After(time.Second):
		t.Fatal("media selection was not forwarded")
	}
	deadline := time.Now().Add(time.Second)
	var job jobs.Job
	for time.Now().Before(deadline) {
		job = store.List()[0]
		if job.Status == "failed" {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if job.Status != "failed" {
		t.Fatalf("job did not finish: %#v", job)
	}
	if job.Params["media_part"] != "2" || job.Params["media_source"] != "DS" {
		t.Fatalf("selection not persisted: %#v", job.Params)
	}
}

func TestRecoverSubtitleSegmentUsesMediaAPISubsegments(t *testing.T) {
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/media/prepare" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if got := r.FormValue("segment_seconds"); got != "10" {
			t.Fatalf("segment_seconds = %q", got)
		}
		w.Header().Set("Content-Type", "application/zip")
		archive := zip.NewWriter(w)
		manifest, _ := archive.Create("manifest.json")
		_, _ = manifest.Write([]byte(`{"source_name":"retry.wav","segments":[{"name":"segment-00000.wav","start":0,"end":10,"duration":10},{"name":"segment-00001.wav","start":10,"end":20,"duration":10},{"name":"segment-00002.wav","start":20,"end":30,"duration":10}]}`))
		for index := 0; index < 3; index++ {
			segment, _ := archive.Create(fmt.Sprintf("segment-%05d.wav", index))
			_, _ = segment.Write([]byte("fake audio"))
		}
		_ = archive.Close()
	}))
	defer mediaWorker.Close()

	requestCount := 0
	asrWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		_ = json.NewEncoder(w).Encode(map[string]any{
			"text": "Shadow line.", "language": "English",
			"timestamps": []map[string]any{{"text": "Shadow", "start": 1.0, "end": 1.5}, {"text": "line", "start": 1.5, "end": 2.0}},
		})
	}))
	defer asrWorker.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	server := New(config.Config{
		DataDir: dataDir,
		Engines: map[string]config.Engine{
			"media": {Endpoint: mediaWorker.URL}, "recognition": {Endpoint: asrWorker.URL},
		},
		Recognition: config.Recognition{Model: "test-asr"},
	}, store, nil)
	inputDir := filepath.Join(dataDir, "inputs", "retry-test")
	if err := os.MkdirAll(inputDir, 0o755); err != nil {
		t.Fatal(err)
	}
	source := filepath.Join(inputDir, "source.wav")
	if err := os.WriteFile(source, []byte("fake audio"), 0o644); err != nil {
		t.Fatal(err)
	}
	cues, detected, err := server.recoverSubtitleSegment(inputDir, source, 210, "English", "")
	if err != nil {
		t.Fatal(err)
	}
	if requestCount != 3 || detected != "English" || len(cues) != 3 {
		t.Fatalf("requests=%d detected=%q cues=%#v", requestCount, detected, cues)
	}
	for index, want := range []float64{211, 221, 231} {
		if cues[index].Start != want {
			t.Fatalf("cue %d start=%f want=%f", index, cues[index].Start, want)
		}
	}
}

func TestPrepareMediaPollsDownloadProgress(t *testing.T) {
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/media/prepare":
			if err := r.ParseMultipartForm(1 << 20); err != nil {
				t.Fatal(err)
			}
			time.Sleep(1200 * time.Millisecond)
			_, _ = w.Write([]byte("prepared"))
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/media/progress/"):
			_ = json.NewEncoder(w).Encode(map[string]any{
				"stage": "downloading", "downloaded_bytes": 50, "total_bytes": 100,
				"percent": 50.0, "eta_seconds": 3,
			})
		case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/v1/media/progress/"):
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	defer mediaWorker.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	server := New(config.Config{DataDir: dataDir, Engines: map[string]config.Engine{"media": {Endpoint: mediaWorker.URL}}}, store, nil)
	job := jobs.Job{ID: "progress-test", Params: map[string]any{}}
	output := filepath.Join(dataDir, "prepared.zip")
	err = server.prepareMediaWithProgress(&job, mediaWorker.URL+"/v1/media/prepare", map[string]string{"request_id": job.ID}, nil, output)
	if err != nil {
		t.Fatal(err)
	}
	if job.Params["media_stage"] != "downloading" || job.Params["media_percent"] != 50.0 || job.Params["media_eta_seconds"] != 3 {
		t.Fatalf("progress not applied: %#v", job.Params)
	}
}
