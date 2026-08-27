package server

import (
	"bytes"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestSpeechOutputCanDriveA2VVideo(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_, _ = io.WriteString(w, `{"status":"ready","busy":false}`)
			return
		}
		if err := r.ParseMultipartForm(4 << 20); err != nil {
			t.Fatal(err)
		}
		if got := len(r.MultipartForm.File["audios"]); got != 1 {
			t.Fatalf("audio files=%d", got)
		}
		if got := r.FormValue("audio_start_times"); got != "[0]" {
			t.Fatalf("audio_start_times=%q", got)
		}
		if r.FormValue("audio_max_duration") == "" {
			t.Fatal("audio_max_duration was not sent")
		}
		_, _ = io.WriteString(w, "a2v video")
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	speech := jobs.Job{ID: "speech-source", Kind: "speech", Status: "completed", OutputURL: "/api/outputs/speech-source.wav", Params: map[string]any{}, CreatedAt: time.Now()}
	if err := os.WriteFile(store.OutputPath("speech-source.wav"), []byte("fake wav"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.Save(speech); err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{DataDir: dataDir, Engines: map[string]config.Engine{"video": {Endpoint: engine.URL}}, Video: config.Video{DefaultWidth: 768, DefaultHeight: 512, DefaultFrames: 121, DefaultFPS: 24}}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "a singer performs to the supplied voice")
	_ = form.WriteField("reuse_audio_job", speech.ID)
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/video", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		for _, job := range store.List() {
			if job.Kind != "video" {
				continue
			}
			if job.Status == "completed" {
				if !boolParam(job.Params, "audio", false) || stringParam(job.Params, "mode", "") != "a2v" {
					t.Fatalf("A2V metadata missing: %#v", job.Params)
				}
				return
			}
			if job.Status == "failed" {
				t.Fatalf("A2V job failed: %s", job.Error)
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("A2V job did not complete")
}

func TestMultipleSpeechOutputsPreserveTimelineForA2V(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			_, _ = io.WriteString(w, `{"status":"ready","busy":false}`)
			return
		}
		if err := r.ParseMultipartForm(8 << 20); err != nil {
			t.Fatal(err)
		}
		if got := len(r.MultipartForm.File["audios"]); got != 2 {
			t.Fatalf("audio files=%d", got)
		}
		if got := r.FormValue("audio_start_times"); got != "[0,3.25]" {
			t.Fatalf("audio_start_times=%q", got)
		}
		_, _ = io.WriteString(w, "multi-audio video")
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"speech-a", "speech-b"} {
		if err := os.WriteFile(store.OutputPath(id+".wav"), []byte("fake wav "+id), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := store.Save(jobs.Job{ID: id, Kind: "speech", Status: "completed", OutputURL: "/api/outputs/" + id + ".wav", Params: map[string]any{}, CreatedAt: time.Now()}); err != nil {
			t.Fatal(err)
		}
	}
	cfg := config.Config{DataDir: dataDir, Engines: map[string]config.Engine{"video": {Endpoint: engine.URL}}, Video: config.Video{DefaultWidth: 768, DefaultHeight: 512, DefaultFrames: 241, DefaultFPS: 24}}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "two speakers perform in sequence")
	_ = form.WriteField("num_frames", "241")
	_ = form.WriteField("fps", "24")
	_ = form.WriteField("audio_count", "2")
	_ = form.WriteField("reuse_audio_job_0", "speech-a")
	_ = form.WriteField("audio_start_0", "0")
	_ = form.WriteField("reuse_audio_job_1", "speech-b")
	_ = form.WriteField("audio_start_1", "3.25")
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/video", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		for _, job := range store.List() {
			if job.Kind != "video" {
				continue
			}
			if job.Status == "completed" {
				var clips []savedVideoAudioClip
				decodeParam(job.Params, "audio_clips", &clips)
				if len(clips) != 2 || clips[1].Start != 3.25 {
					t.Fatalf("audio clips=%#v", clips)
				}
				return
			}
			if job.Status == "failed" {
				t.Fatalf("A2V job failed: %s", job.Error)
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("multi-audio A2V job did not complete")
}

func TestDownloadedVideoRemainsUsableAfterTranscriptionFailure(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/media/assets/saved-video":
			_, _ = io.WriteString(w, "downloaded video")
		case r.Method == http.MethodPost && r.URL.Path == "/v1/videos/upscale":
			_, _ = io.WriteString(w, "upscaled video")
		default:
			http.NotFound(w, r)
		}
	}))
	defer engine.Close()
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	source := jobs.Job{
		ID: "failed-transcription", Kind: "recognition", Status: "failed", Error: "translation failed",
		MediaAssetID: "saved-video", MediaURL: "/api/media/assets/saved-video", CreatedAt: time.Now(),
		Params: map[string]any{"media": map[string]any{"media_type": "video", "width": 1280, "height": 720, "duration": 12.0}},
	}
	if err := store.Save(source); err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: dataDir, Engines: map[string]config.Engine{"media": {Endpoint: engine.URL}, "upscale": {Endpoint: engine.URL}}}, store, nil).Handler()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/failed-transcription/video-upscale", bytes.NewBufferString(`{"scale":1.5,"start_time":0,"end_time":5}`))
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		for _, job := range store.List() {
			if job.Kind == "video" && job.ID != source.ID && job.Status == "completed" {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("failed-transcription source upscale did not complete")
}
