package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestFailedVideoCanBeGeneratedAgainWithSavedScenes(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/videos/generations" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			t.Fatal(err)
		}
		if got := r.FormValue("prompt"); got != "enhanced scene motion" {
			t.Fatalf("prompt=%q", got)
		}
		if got := r.FormValue("frame_indices"); got != "[0,40,80]" {
			t.Fatalf("frame_indices=%q", got)
		}
		if got := len(r.MultipartForm.File["images"]); got != 3 {
			t.Fatalf("images=%d", got)
		}
		_, _ = io.WriteString(w, "video")
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	job := jobs.Job{
		ID: "interrupted-video", Kind: "video", Status: "failed", Prompt: "장면을 자연스럽게 연결",
		Error: "앱 재시작으로 작업이 중단되었습니다.", CreatedAt: time.Now(),
		Params: map[string]any{
			"width": 768, "height": 512, "num_frames": 81, "fps": 24.0, "seed": int64(7),
			"enhanced_prompt": "enhanced scene motion",
			"video_conditions": []savedVideoCondition{
				{Role: "start", FrameIdx: 0, Strength: 1},
				{Role: "keyframe", Index: 0, FrameIdx: 40, Strength: .8},
				{Role: "end", FrameIdx: 80, Strength: 1},
			},
		},
	}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	for _, role := range []string{"start", "keyframe-0", "end"} {
		dir := filepath.Join(dataDir, "inputs", job.ID, role)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "scene.png"), testPNG(t, 16, 16), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	cfg := config.Config{
		DataDir: dataDir,
		Engines: map[string]config.Engine{"video": {Endpoint: engine.URL}},
		Video:   config.Video{DefaultWidth: 768, DefaultHeight: 512, DefaultFrames: 81, DefaultFPS: 24},
	}
	handler := New(cfg, store, nil).Handler()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/interrupted-video/retry", strings.NewReader(""))
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		current, _ := store.Get(job.ID)
		if current.Status == "completed" {
			if current.OutputURL == "" || imageIntParam(current.Params, "retry_count", 0) != 1 {
				t.Fatalf("completed job=%#v", current)
			}
			return
		}
		if current.Status == "failed" {
			t.Fatalf("retry failed: %s", current.Error)
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("retried video did not complete")
}
