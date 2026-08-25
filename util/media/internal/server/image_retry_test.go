package server

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestFailedImageCanBeGeneratedAgainWithSavedSettings(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Prompt    string `json:"prompt"`
			Size      string `json:"size"`
			Seed      int64  `json:"seed"`
			Steps     int    `json:"steps"`
			Sampler   string `json:"sampler_name"`
			Scheduler string `json:"scheduler"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request.Prompt != "enhanced Seoul skyline" || request.Size != "1024x768" || request.Seed != 42 || request.Steps != 8 || request.Sampler != "euler" || request.Scheduler != "simple" {
			t.Fatalf("request=%#v", request)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"seed": int64(42),
			"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(testPNG(t, 32, 24))}},
		})
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	job := jobs.Job{
		ID: "interrupted-image", Kind: "image", Status: "failed", Prompt: "서울 야경",
		Error: "앱 재시작으로 작업이 중단되었습니다.", CreatedAt: time.Now(),
		Params: map[string]any{
			"mode": "create", "width": 1024, "height": 768, "seed": int64(42),
			"enhanced_prompt": "enhanced Seoul skyline", "steps": 8,
			"sampling_preset": "default", "sampler": "euler", "scheduler": "simple",
			"filter_mode": "balanced", "filter_strength": 1.0,
			"prompt_enhancer_strength": 1.0, "prompt_text_scale": 1.75,
		},
	}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{
		DataDir: dataDir,
		Image: config.Image{
			DefaultWidth: 1024, DefaultHeight: 768, DefaultMode: "create",
			Backends: map[string]config.ImageBackend{"create": {Endpoint: engine.URL, Model: "krea-test"}},
		},
	}
	handler := New(cfg, store, nil).Handler()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/interrupted-image/retry", strings.NewReader(""))
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
	t.Fatal("retried image did not complete")
}
