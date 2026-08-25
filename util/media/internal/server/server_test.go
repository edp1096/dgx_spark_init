package server

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"mime/multipart"
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

func testPNG(t *testing.T, width, height int) []byte {
	t.Helper()
	var output bytes.Buffer
	picture := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			picture.Set(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 120, A: 255})
		}
	}
	if err := png.Encode(&output, picture); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}

func TestCompletedImageOutputCanBeReusedAsInput(t *testing.T) {
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	job := jobs.Job{ID: "parent", Kind: "image", Status: "completed", OutputURL: "/api/outputs/parent.png", Params: map[string]any{}, CreatedAt: time.Now()}
	if err := os.WriteFile(store.OutputPath("parent.png"), testPNG(t, 32, 24), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	server := New(config.Config{DataDir: dataDir}, store, nil)
	files, err := server.imageInputFiles("parent", "output")
	if err != nil || len(files) != 1 || filepath.Base(files[0]) != "parent.png" {
		t.Fatalf("files=%v err=%v", files, err)
	}
}

func TestImageJobCompletesThroughEngine(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/images/generations":
			var request struct {
				Prompt     string `json:"prompt"`
				Size       string `json:"size"`
				Checkpoint string `json:"checkpoint"`
				Sampler    string `json:"sampler_name"`
				Scheduler  string `json:"scheduler"`
				Steps      int    `json:"steps"`
			}
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatal(err)
			}
			if request.Prompt != "green glass sphere" || request.Size != "512x512" || request.Checkpoint != "moody-v7" || request.Sampler != "euler_ancestral" || request.Scheduler != "beta" || request.Steps != 8 {
				t.Fatalf("unexpected request %#v", request)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"seed": int64(987654321), "data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("fake png"))}}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}},
		Image:   config.Image{Model: "test-image", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "green glass sphere")
	_ = form.WriteField("checkpoint", "moody-v7")
	_ = form.WriteField("sampling_preset", "moody")
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Params["seed"] != int64(987654321) {
				t.Fatalf("actual seed was not recorded: %#v", list[0].Params)
			}
			file, err := os.Open(store.OutputPath(list[0].ID + ".png"))
			if err != nil {
				t.Fatal(err)
			}
			got, err := io.ReadAll(file)
			_ = file.Close()
			if err != nil || string(got) != "fake png" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestImageSequenceUsesPreviousResultAsIdentity(t *testing.T) {
	type engineRequest struct {
		Prompt           string  `json:"prompt"`
		SourceImage      string  `json:"source_image"`
		Steps            int     `json:"steps"`
		IdentityStrength float64 `json:"identity_strength"`
	}
	requests := make(chan engineRequest, 2)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		var request engineRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		requests <- request
		result := "first scene png"
		if request.SourceImage != "" {
			result = "second scene png"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte(result))}}})
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}},
		Image: config.Image{
			DefaultMode: "create", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4,
			Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}},
		},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "same woman in a quiet room")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("steps", "8")
	_ = form.WriteField("sequence_prompts", `["same woman in a quiet room","she walks toward the window"]`)
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var response struct {
		SequenceID string     `json:"sequence_id"`
		Jobs       []jobs.Job `json:"jobs"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &response); err != nil || len(response.Jobs) != 2 || response.SequenceID == "" {
		t.Fatalf("response=%#v err=%v body=%s", response, err, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		completed := 0
		for _, job := range store.List() {
			if job.Status == "completed" {
				completed++
			}
		}
		if completed == 2 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	list := store.List()
	if len(list) != 2 || list[0].Status != "completed" || list[1].Status != "completed" {
		t.Fatalf("sequence did not complete: %#v", list)
	}
	firstRequest := <-requests
	secondRequest := <-requests
	if firstRequest.SourceImage != "" || firstRequest.Prompt != "same woman in a quiet room" || firstRequest.Steps != 8 {
		t.Fatalf("unexpected first request: %#v", firstRequest)
	}
	source, err := base64.StdEncoding.DecodeString(secondRequest.SourceImage)
	if err != nil || string(source) != "first scene png" {
		t.Fatalf("second source=%q err=%v request=%#v", source, err, secondRequest)
	}
	if !strings.Contains(secondRequest.Prompt, "Change: she walks toward the window") || !strings.Contains(secondRequest.Prompt, "Preserve:") || secondRequest.Steps != 10 || secondRequest.IdentityStrength != 0.8 {
		t.Fatalf("unexpected second request: %#v", secondRequest)
	}
	second := response.Jobs[1]
	if second.Params["sequence_previous_job_id"] != response.Jobs[0].ID || second.Params["sequence_index"] != float64(2) {
		t.Fatalf("unexpected sequence params: %#v", second.Params)
	}
}

func TestSequenceAnyPaintMaskUsesNormalizedArmRegion(t *testing.T) {
	dataDir := t.TempDir()
	source := filepath.Join(dataDir, "source.png")
	if err := os.WriteFile(source, testPNG(t, 100, 80), 0o644); err != nil {
		t.Fatal(err)
	}
	server := &Server{dataDir: dataDir}
	job := jobs.Job{ID: "masked-scene", Params: map[string]any{"sequence_region": "left-arm"}}
	if err := server.materializeSequenceAnyPaintMask(job, source); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(filepath.Join(dataDir, "inputs", job.ID, "anypaint-mask", "0.png"))
	if err != nil {
		t.Fatal(err)
	}
	mask, err := png.Decode(file)
	_ = file.Close()
	if err != nil {
		t.Fatal(err)
	}
	if color.GrayModel.Convert(mask.At(10, 20)).(color.Gray).Y != 255 || color.GrayModel.Convert(mask.At(90, 20)).(color.Gray).Y != 0 || color.GrayModel.Convert(mask.At(45, 70)).(color.Gray).Y != 255 {
		t.Fatalf("unexpected normalized left-arm mask pixels")
	}
}

func TestImageSequenceCanContinueFromCompletedImageWithPaintedMask(t *testing.T) {
	type engineRequest struct {
		Prompt string `json:"prompt"`
		Image  string `json:"anypaint_image"`
		Mask   string `json:"anypaint_mask"`
	}
	requests := make(chan engineRequest, 1)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request engineRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		requests <- request
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(testPNG(t, 64, 48))}}})
	}))
	defer worker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 64, DefaultHeight: 48, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	baseBytes := testPNG(t, 64, 48)
	baseName := "sequence-base.png"
	if err := os.WriteFile(store.OutputPath(baseName), baseBytes, 0o644); err != nil {
		t.Fatal(err)
	}
	base := jobs.Job{ID: "existing-image", Kind: "image", Status: "completed", Prompt: "a robot standing still", OutputURL: "/api/outputs/" + baseName, CreatedAt: time.Now().Add(-time.Minute)}
	if err := store.Save(base); err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", base.Prompt)
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("sequence_prompts", `["a robot standing still","raise only the robot's left arm"]`)
	_ = form.WriteField("sequence_regions", `["all","custom"]`)
	_ = form.WriteField("sequence_base_job_id", base.ID)
	part, err := form.CreateFormFile("sequence_mask_1", "painted-mask.png")
	if err != nil {
		t.Fatal(err)
	}
	maskBytes := testPNG(t, 64, 48)
	_, _ = part.Write(maskBytes)
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var response struct {
		Jobs []jobs.Job `json:"jobs"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &response); err != nil || len(response.Jobs) != 1 {
		t.Fatalf("response=%#v err=%v body=%s", response, err, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		job, ok := store.Get(response.Jobs[0].ID)
		if ok && job.Status == "completed" {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	select {
	case request := <-requests:
		imageBytes, imageErr := base64.StdEncoding.DecodeString(request.Image)
		maskResult, maskErr := base64.StdEncoding.DecodeString(request.Mask)
		if imageErr != nil || maskErr != nil || !bytes.Equal(imageBytes, baseBytes) || !bytes.Equal(maskResult, maskBytes) {
			t.Fatalf("unexpected painted sequence inputs: imageErr=%v maskErr=%v", imageErr, maskErr)
		}
		if !strings.Contains(request.Prompt, "raise only the robot's left arm") {
			t.Fatalf("unexpected prompt: %q", request.Prompt)
		}
	default:
		t.Fatal("painted sequence request was not sent")
	}
}

func TestCompletedImageCanBeUpscaledThroughSeedVR2(t *testing.T) {
	upscaled := testPNG(t, 64, 48)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/upscale" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Model string `json:"model"`
			Image string `json:"image"`
			Scale int    `json:"scale"`
			Seed  int64  `json:"seed"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.Image)
		if err != nil {
			t.Fatal(err)
		}
		input, _, err := image.DecodeConfig(bytes.NewReader(decoded))
		if err != nil || input.Width != 32 || input.Height != 24 {
			t.Fatalf("input=%#v err=%v", input, err)
		}
		if request.Model != "seedvr2-3b-fp8" || request.Scale != 2 || request.Seed != 77 {
			t.Fatalf("unexpected upscale request: %#v", request)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(upscaled)}}})
	}))
	defer worker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"upscale": {Endpoint: worker.URL}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	source := jobs.Job{ID: "source-image", Kind: "image", Status: "completed", Prompt: "source prompt", Params: map[string]any{}, OutputURL: "/api/outputs/source-image.png", CreatedAt: time.Now()}
	if err := os.WriteFile(store.OutputPath("source-image.png"), testPNG(t, 32, 24), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.Save(source); err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/source-image/upscale", strings.NewReader(`{"scale":2,"seed":77}`))
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	var created jobs.Job
	if err := json.Unmarshal(res.Body.Bytes(), &created); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		current, ok := store.Get(created.ID)
		if ok && current.Status == "completed" {
			if current.Params["source_job_id"] != source.ID || current.Params["width"] != 64 || current.Params["height"] != 48 {
				t.Fatalf("unexpected completed job: %#v", current)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("upscale job did not complete: %#v", store.List())
}

func TestCompletedImageCanBeReinterpretedThroughKreaDetailEnhancer(t *testing.T) {
	result := testPNG(t, 512, 512)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Model       string  `json:"model"`
			Image       string  `json:"detail_enhance_image"`
			Strength    float64 `json:"detail_strength"`
			VAE         string  `json:"detail_vae"`
			Steps       int     `json:"steps"`
			FilterMode  string  `json:"filter_mode"`
			FilterLevel float64 `json:"filter_strength"`
			Sampler     string  `json:"sampler_name"`
			Scheduler   string  `json:"scheduler"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.Image)
		if err != nil {
			t.Fatal(err)
		}
		input, _, err := image.DecodeConfig(bytes.NewReader(decoded))
		if err != nil || input.Width != 512 || input.Height != 512 {
			t.Fatalf("input=%#v err=%v", input, err)
		}
		if request.Model != "krea-test" || request.Strength != 1 || request.VAE != "wan" || request.Steps != 10 || request.FilterMode != "balanced" || request.FilterLevel != 1 || request.Sampler != "er_sde" || request.Scheduler != "simple" {
			t.Fatalf("unexpected detail request: %#v", request)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(result)}}})
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Image: config.Image{Backends: map[string]config.ImageBackend{
			"create": {Endpoint: worker.URL, Model: "krea-test"},
		}},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	source := jobs.Job{ID: "detail-source", Kind: "image", Status: "completed", Prompt: "portrait", Params: map[string]any{}, OutputURL: "/api/outputs/detail-source.png", CreatedAt: time.Now()}
	if err := os.WriteFile(store.OutputPath("detail-source.png"), testPNG(t, 512, 512), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := store.Save(source); err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/detail-source/detail-enhance", strings.NewReader(`{"strength":1,"seed":-1,"vae":"wan"}`))
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var created jobs.Job
	if err := json.Unmarshal(res.Body.Bytes(), &created); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		current, ok := store.Get(created.ID)
		if ok && current.Status == "completed" {
			if current.Params["source_job_id"] != source.ID || current.Params["mode"] != "detail_enhance" {
				t.Fatalf("unexpected completed job: %#v", current)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("detail job did not complete: %#v", store.List())
}

func TestControlImageRoutesToZImageBackend(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Model           string  `json:"model"`
			ControlImage    string  `json:"control_image"`
			ControlStrength float64 `json:"control_strength"`
			ControlStrategy string  `json:"control_strategy"`
			ControlType     string  `json:"control_type"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.ControlImage)
		if err != nil || string(decoded) != "control png" {
			t.Fatalf("control image=%q err=%v", decoded, err)
		}
		if request.Model != "z-image-test" || request.ControlStrength != 0.65 || request.ControlStrategy != "split4" || request.ControlType != "canny" {
			t.Fatalf("unexpected control request: %#v", request)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("controlled png"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}},
		Image: config.Image{
			Model: "legacy", DefaultMode: "control", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4,
			Backends: map[string]config.ImageBackend{"control": {Endpoint: worker.URL, Model: "z-image-test"}},
		},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "preserve the outline")
	_ = form.WriteField("mode", "control")
	_ = form.WriteField("control_type", "canny")
	_ = form.WriteField("control_strength", "0.65")
	part, _ := form.CreateFormFile("references", "control.png")
	_, _ = part.Write([]byte("control png"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".png"))
			if err != nil || string(got) != "controlled png" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			if list[0].Params["mode"] != "control" || list[0].Params["control_type"] != "canny" {
				t.Fatalf("job params=%#v", list[0].Params)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestStoredImageInputCanBePreviewedAndReused(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			ControlImage string `json:"control_image"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.ControlImage)
		if err != nil || string(decoded) != "reusable control image" {
			t.Fatalf("control image=%q err=%v", decoded, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("result"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}},
		Image: config.Image{
			DefaultMode: "control", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4,
			Backends: map[string]config.ImageBackend{"control": {Endpoint: worker.URL, Model: "z-image-test"}},
		},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	create := func(referenceField, referenceValue string, upload bool) jobs.Job {
		t.Helper()
		var body bytes.Buffer
		form := multipart.NewWriter(&body)
		_ = form.WriteField("prompt", "reuse this structure")
		_ = form.WriteField("mode", "control")
		if upload {
			part, partErr := form.CreateFormFile(referenceField, "control.png")
			if partErr != nil {
				t.Fatal(partErr)
			}
			_, _ = part.Write([]byte(referenceValue))
		} else {
			_ = form.WriteField(referenceField, referenceValue)
		}
		_ = form.Close()
		req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
		req.Header.Set("Content-Type", form.FormDataContentType())
		res := httptest.NewRecorder()
		handler.ServeHTTP(res, req)
		if res.Code != http.StatusAccepted {
			t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
		}
		var job jobs.Job
		if err := json.Unmarshal(res.Body.Bytes(), &job); err != nil {
			t.Fatal(err)
		}
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			current, ok := store.Get(job.ID)
			if ok && current.Status == "completed" {
				return current
			}
			time.Sleep(10 * time.Millisecond)
		}
		t.Fatalf("job %s did not complete", job.ID)
		return jobs.Job{}
	}

	first := create("references", "reusable control image", true)
	inputsReq := httptest.NewRequest(http.MethodGet, "/api/jobs/"+first.ID+"/inputs", nil)
	inputsRes := httptest.NewRecorder()
	handler.ServeHTTP(inputsRes, inputsReq)
	if inputsRes.Code != http.StatusOK {
		t.Fatalf("inputs status=%d body=%s", inputsRes.Code, inputsRes.Body.String())
	}
	var inputs []imageJobInputInfo
	if err := json.Unmarshal(inputsRes.Body.Bytes(), &inputs); err != nil {
		t.Fatal(err)
	}
	if len(inputs) != 1 || inputs[0].Role != "reference" || inputs[0].Ref == "" {
		t.Fatalf("unexpected inputs: %#v", inputs)
	}
	previewReq := httptest.NewRequest(http.MethodGet, inputs[0].URL, nil)
	previewRes := httptest.NewRecorder()
	handler.ServeHTTP(previewRes, previewReq)
	if previewRes.Code != http.StatusOK || previewRes.Body.String() != "reusable control image" {
		t.Fatalf("preview status=%d body=%q", previewRes.Code, previewRes.Body.String())
	}

	second := create("reuse_references", inputs[0].Ref, false)
	reused, err := os.ReadFile(filepath.Join(cfg.DataDir, "inputs", second.ID, "0.png"))
	if err != nil || string(reused) != "reusable control image" {
		t.Fatalf("reused input=%q err=%v", reused, err)
	}
}

func TestEnhancementPreservesIdentityEditContract(t *testing.T) {
	original := "Change: Replace all clothing with the outfit from the supporting reference and apply the pose from the Depth image. Preserve: identity. Do not preserve original clothing or pose."
	good := "Replace the clothing with the reference outfit and follow the Depth-controlled posture while preserving identity; do not retain the source garments or pose."
	badPose := "Use the supporting reference outfit and Depth image while preserving the original pose and clothing."
	badReference := "Change the outfit and posture while preserving identity."
	if !enhancementPreservesEditContract("edit_control", original, good) {
		t.Fatal("valid edit-control enhancement was rejected")
	}
	if enhancementPreservesEditContract("edit_control", original, badPose) {
		t.Fatal("contradictory pose preservation was accepted")
	}
	if enhancementPreservesEditContract("edit_control", original, badReference) {
		t.Fatal("missing reference and depth controls were accepted")
	}
	if !enhancementPreservesEditContract("t2i", original, badPose) {
		t.Fatal("non-edit modes must not use the edit contract")
	}
}

func TestKreaModulesRouteToCreateBackend(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			http.NotFound(w, r)
			return
		}
		var request struct {
			Model            string   `json:"model"`
			SourceImage      string   `json:"source_image"`
			ReferenceImages  []string `json:"reference_images"`
			ControlImage     string   `json:"control_image"`
			IdentityStrength float64  `json:"identity_strength"`
			RefBoost         float64  `json:"ref_boost"`
			SourceRefBoost   float64  `json:"source_ref_boost"`
			GroundingPX      int      `json:"grounding_px"`
			ControlStrength  float64  `json:"control_strength"`
			Style            string   `json:"style"`
			StyleStrength    float64  `json:"style_strength"`
			Styles           []struct {
				Name     string  `json:"name"`
				Strength float64 `json:"strength"`
			} `json:"styles"`
			Steps             int     `json:"steps"`
			FilterMode        string  `json:"filter_mode"`
			FilterStrength    float64 `json:"filter_strength"`
			PromptEnhancer    bool    `json:"prompt_enhancer"`
			PromptEnhStrength float64 `json:"prompt_enhancer_strength"`
			PromptTextScale   float64 `json:"prompt_text_scale"`
			IdentityFitMode   string  `json:"identity_fit_mode"`
			VAEMode           string  `json:"vae_mode"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decode := func(value string) string {
			decoded, err := base64.StdEncoding.DecodeString(value)
			if err != nil {
				t.Fatal(err)
			}
			return string(decoded)
		}
		if request.Model != "krea-test" || decode(request.SourceImage) != "identity png" || len(request.ReferenceImages) != 2 || decode(request.ReferenceImages[0]) != "person png" || decode(request.ReferenceImages[1]) != "outfit png" || decode(request.ControlImage) != "pose png" {
			t.Fatalf("unexpected Krea images: %#v", request)
		}
		if request.IdentityStrength != 1 || request.RefBoost != 4 || request.SourceRefBoost != 1.5 || request.GroundingPX != 768 || request.ControlStrength != 0.8 || request.Style != "retroanime" || request.StyleStrength != 0.8 || request.Steps != 10 {
			t.Fatalf("unexpected Krea settings: %#v", request)
		}
		if request.FilterMode != "strong" || request.FilterStrength != 1 || !request.PromptEnhancer || request.PromptEnhStrength != 1.25 || request.PromptTextScale != 2 {
			t.Fatalf("unexpected Krea filter settings: %#v", request)
		}
		if request.IdentityFitMode != "crop" || request.VAEMode != "wan" {
			t.Fatalf("unexpected Krea identity input settings: %#v", request)
		}
		if len(request.Styles) != 2 || request.Styles[0].Name != "retroanime" || request.Styles[0].Strength != 0.8 || request.Styles[1].Name != "softwatercolor" || request.Styles[1].Strength != 0.6 {
			t.Fatalf("unexpected stacked Krea styles: %#v", request.Styles)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("krea modules png"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}},
		Image: config.Image{
			Model: "legacy", DefaultMode: "create", DefaultWidth: 1024, DefaultHeight: 1024, MaxReferenceImages: 4,
			Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}},
		},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "restage this person")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("identity_strength", "1")
	_ = form.WriteField("ref_boost", "4")
	_ = form.WriteField("source_ref_boost", "1.5")
	_ = form.WriteField("grounding_px", "768")
	_ = form.WriteField("depth_strength", "0.8")
	_ = form.WriteField("styles", `[{"name":"retroanime","strength":0.8},{"name":"softwatercolor","strength":0.6}]`)
	_ = form.WriteField("steps", "10")
	_ = form.WriteField("filter_mode", "strong")
	_ = form.WriteField("filter_strength", "1")
	_ = form.WriteField("prompt_enhancer", "true")
	_ = form.WriteField("prompt_enhancer_strength", "1.25")
	_ = form.WriteField("prompt_text_scale", "2")
	_ = form.WriteField("identity_fit_mode", "crop")
	_ = form.WriteField("vae_mode", "wan")
	for _, upload := range []struct{ field, name, content string }{
		{"identity_image", "identity.png", "identity png"},
		{"identity_reference", "person.png", "person png"},
		{"identity_reference", "outfit.png", "outfit png"},
		{"depth_image", "pose.png", "pose png"},
	} {
		part, partErr := form.CreateFormFile(upload.field, upload.name)
		if partErr != nil {
			t.Fatal(partErr)
		}
		_, _ = part.Write([]byte(upload.content))
	}
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			got, readErr := os.ReadFile(store.OutputPath(list[0].ID + ".png"))
			if readErr != nil || string(got) != "krea modules png" {
				t.Fatalf("output=%q err=%v", got, readErr)
			}
			if list[0].Params["identity"] != true || list[0].Params["depth"] != true || list[0].Params["style"] != "retroanime" {
				t.Fatalf("job params=%#v", list[0].Params)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestKreaNK2ERoutesAndStoresReference(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model        string  `json:"model"`
			NK2EImage    string  `json:"nk2e_image"`
			NK2EMode     string  `json:"nk2e_mode"`
			NK2EStrength float64 `json:"nk2e_strength"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.NK2EImage)
		if err != nil || string(decoded) != "nk2e png" || request.Model != "krea-test" || request.NK2EMode != "canny" || request.NK2EStrength != 0.8 {
			t.Fatalf("unexpected NK2E request: %#v decoded=%q err=%v", request, decoded, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("nk2e result"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "turn the dancer into a robot")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("nk2e_mode", "canny")
	_ = form.WriteField("nk2e_strength", "0.8")
	part, _ := form.CreateFormFile("nk2e_image", "pose.png")
	_, _ = part.Write([]byte("nk2e png"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Params["nk2e"] != true || list[0].Params["nk2e_mode"] != "canny" {
				t.Fatalf("job params=%#v", list[0].Params)
			}
			inputsReq := httptest.NewRequest(http.MethodGet, "/api/jobs/"+list[0].ID+"/inputs", nil)
			inputsReq.SetPathValue("id", list[0].ID)
			inputsRes := httptest.NewRecorder()
			handler.ServeHTTP(inputsRes, inputsReq)
			var inputs []imageJobInputInfo
			if err := json.Unmarshal(inputsRes.Body.Bytes(), &inputs); err != nil || len(inputs) != 1 || inputs[0].Role != "nk2e" {
				t.Fatalf("inputs=%#v err=%v", inputs, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestKreaAnyPaintRoutesAndStoresSourceAndMask(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model        string  `json:"model"`
			Image        string  `json:"anypaint_image"`
			Mask         string  `json:"anypaint_mask"`
			Left         int     `json:"outpaint_left"`
			Right        int     `json:"outpaint_right"`
			Strength     float64 `json:"anypaint_strength"`
			Boundary     int     `json:"anypaint_boundary_redraw_px"`
			ReferenceMax int     `json:"anypaint_reference_max_edge"`
			VLMReference bool    `json:"anypaint_vlm_reference"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		image, imageErr := base64.StdEncoding.DecodeString(request.Image)
		mask, maskErr := base64.StdEncoding.DecodeString(request.Mask)
		if imageErr != nil || maskErr != nil || string(image) != "source png" || string(mask) != "mask png" {
			t.Fatalf("unexpected AnyPaint images: %#v image=%q mask=%q", request, image, mask)
		}
		if request.Model != "krea-test" || request.Left != 128 || request.Right != 384 || request.Strength != 1.1 || request.Boundary != 16 || request.ReferenceMax != 384 || !request.VLMReference {
			t.Fatalf("unexpected AnyPaint settings: %#v", request)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("anypaint result"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 1024, DefaultHeight: 512, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "extend the room and replace the masked object")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("outpaint_left", "128")
	_ = form.WriteField("outpaint_right", "384")
	_ = form.WriteField("anypaint_strength", "1.1")
	_ = form.WriteField("anypaint_boundary_redraw_px", "16")
	for _, upload := range []struct{ field, name, content string }{
		{"anypaint_image", "source.png", "source png"},
		{"anypaint_mask", "mask.png", "mask png"},
	} {
		part, partErr := form.CreateFormFile(upload.field, upload.name)
		if partErr != nil {
			t.Fatal(partErr)
		}
		_, _ = part.Write([]byte(upload.content))
	}
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Params["anypaint"] != true || list[0].Params["anypaint_mask"] != true || list[0].Params["outpaint_right"] != 384 {
				t.Fatalf("job params=%#v", list[0].Params)
			}
			inputsReq := httptest.NewRequest(http.MethodGet, "/api/jobs/"+list[0].ID+"/inputs", nil)
			inputsRes := httptest.NewRecorder()
			handler.ServeHTTP(inputsRes, inputsReq)
			var inputs []imageJobInputInfo
			if err := json.Unmarshal(inputsRes.Body.Bytes(), &inputs); err != nil || len(inputs) != 2 || inputs[0].Role != "anypaint" || inputs[1].Role != "anypaint_mask" {
				t.Fatalf("inputs=%#v err=%v", inputs, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("AnyPaint job did not complete: %#v", store.List())
}

func TestKreaAnyPaintOutpaintAllowsBlankPrompt(t *testing.T) {
	const automaticPrompt = "Extend the original image naturally into a complete, coherent composition while preserving its subjects, style, lighting, perspective, and visual continuity."
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Prompt string `json:"prompt"`
			Image  string `json:"anypaint_image"`
			Right  int    `json:"outpaint_right"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		image, err := base64.StdEncoding.DecodeString(request.Image)
		if err != nil || string(image) != "source png" || request.Prompt != automaticPrompt || request.Right != 256 {
			t.Fatalf("unexpected blank-prompt Outpaint request: %#v image=%q err=%v", request, image, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("outpaint result"))}}})
	}))
	defer worker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 1024, DefaultHeight: 512, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("outpaint_right", "256")
	part, err := form.CreateFormFile("anypaint_image", "source.png")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write([]byte("source png"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Prompt != automaticPrompt || list[0].Params["anypaint_mask"] != false {
				t.Fatalf("job=%#v", list[0])
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("blank-prompt Outpaint job did not complete: %#v", store.List())
}

func TestKreaStyleReferenceRoutesAndIsReusable(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Images   []string `json:"style_reference_images"`
			Strength float64  `json:"style_reference_strength"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		decoded, err := base64.StdEncoding.DecodeString(request.Images[0])
		if err != nil || string(decoded) != "style reference" || request.Strength != 0.9 {
			t.Fatalf("unexpected style reference request: %#v decoded=%q err=%v", request, decoded, err)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("styled png"))}}})
	}))
	defer worker.Close()
	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"image": {Endpoint: worker.URL}}, Image: config.Image{DefaultMode: "create", DefaultWidth: 1024, DefaultHeight: 1024, MaxReferenceImages: 4, Backends: map[string]config.ImageBackend{"create": {Endpoint: worker.URL, Model: "krea-test"}}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "use this visual style")
	_ = form.WriteField("mode", "create")
	_ = form.WriteField("style_reference_strength", "0.9")
	part, _ := form.CreateFormFile("style_reference_images", "style.png")
	_, _ = part.Write([]byte("style reference"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/image", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			inputsReq := httptest.NewRequest(http.MethodGet, "/api/jobs/"+list[0].ID+"/inputs", nil)
			inputsRes := httptest.NewRecorder()
			handler.ServeHTTP(inputsRes, inputsReq)
			var inputs []imageJobInputInfo
			if json.Unmarshal(inputsRes.Body.Bytes(), &inputs) != nil || len(inputs) != 1 || inputs[0].Role != "style_reference" {
				t.Fatalf("unexpected stored inputs: %s", inputsRes.Body.String())
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("style reference job did not complete")
}

func TestVideoJobStreamsEngineOutput(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/videos/generations" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if r.FormValue("prompt") != "waves under moonlight" || r.FormValue("width") != "768" || r.FormValue("height") != "512" || r.FormValue("num_frames") != "121" || r.FormValue("fps") != "24" || r.FormValue("seed") != "42" {
			t.Fatalf("unexpected fields: %#v", r.MultipartForm.Value)
		}
		if r.FormValue("frame_indices") != "[0,60,120]" || r.FormValue("image_strengths") != "[0.8,0.7,0.9]" || len(r.MultipartForm.File["images"]) != 3 {
			t.Fatalf("unexpected conditioning: values=%#v files=%#v", r.MultipartForm.Value, r.MultipartForm.File)
		}
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte("fake mp4"))
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"video": {Endpoint: worker.URL}},
		Video:   config.Video{Model: "test-video", DefaultWidth: 768, DefaultHeight: 512, DefaultFrames: 121, DefaultFPS: 24},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "waves under moonlight")
	_ = form.WriteField("seed", "42")
	_ = form.WriteField("image_strength", "0.8")
	_ = form.WriteField("end_image_strength", "0.9")
	_ = form.WriteField("keyframe_count", "1")
	_ = form.WriteField("keyframe_time_0", "2.5")
	_ = form.WriteField("keyframe_strength_0", "0.7")
	for _, field := range []string{"start_image", "keyframe_image_0", "end_image"} {
		part, err := form.CreateFormFile(field, field+".png")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = part.Write([]byte("fake image"))
	}
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
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".mp4"))
			if err != nil || string(got) != "fake mp4" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestConfigUpdatePersistsAndAppliesImmediately(t *testing.T) {
	first := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "offline", http.StatusServiceUnavailable)
	}))
	defer first.Close()
	second := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" || r.URL.Path == "/v1/models" {
			w.WriteHeader(http.StatusOK)
			return
		}
		http.NotFound(w, r)
	}))
	defer second.Close()

	cfg := config.Config{
		Listen:  "127.0.0.1:8686",
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{
			"image": {Endpoint: first.URL}, "video": {Endpoint: first.URL},
			"speech": {Endpoint: first.URL}, "recognition": {Endpoint: first.URL}, "prompt": {Endpoint: first.URL}, "media": {Endpoint: first.URL}, "trainer": {Endpoint: first.URL}, "upscale": {Endpoint: first.URL},
		},
		Image:             config.Image{Model: "image", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4},
		Video:             config.Video{Model: "video", DefaultWidth: 768, DefaultHeight: 512, DefaultFrames: 121, DefaultFPS: 24},
		Speech:            config.Speech{CustomVoiceModel: "speech", DefaultLanguage: "Korean", DefaultSpeaker: "Sohee"},
		Recognition:       config.Recognition{Model: "asr", DefaultLanguage: "Auto", MaxUploadMB: 500, SegmentSeconds: 30, DefaultOutputFormats: []string{"txt"}, DefaultTranslationMode: "none"},
		PromptEnhancement: config.PromptEnhancement{Model: "enhancer", DefaultEnabled: true, MaxTokens: 600},
	}
	configPath := filepath.Join(t.TempDir(), "media.yaml")
	if err := config.Save(configPath, cfg); err != nil {
		t.Fatal(err)
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil, configPath).Handler()

	next := cfg
	next.Engines = map[string]config.Engine{
		"image": {Endpoint: second.URL}, "video": {Endpoint: second.URL},
		"speech": {Endpoint: second.URL}, "recognition": {Endpoint: second.URL}, "prompt": {Endpoint: second.URL}, "media": {Endpoint: second.URL}, "trainer": {Endpoint: second.URL}, "upscale": {Endpoint: second.URL},
	}
	next.Video.DefaultFrames = 65
	next.Image.DefaultPromptEnhancer = true
	next.ImageMetadata = config.ImageMetadata{Creator: " Studio Name ", Copyright: "© 2026 Studio", Website: "https://example.com", Note: "Portfolio image"}
	body, _ := json.Marshal(next)
	req := httptest.NewRequest(http.MethodPut, "/api/config", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	loaded, _, err := config.Load(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Video.DefaultFrames != 65 || loaded.Engines["video"].Endpoint != second.URL || !loaded.Image.DefaultPromptEnhancer || loaded.ImageMetadata.Creator != "Studio Name" {
		t.Fatalf("saved config was not updated: %#v", loaded)
	}

	stateReq := httptest.NewRequest(http.MethodGet, "/api/engines", nil)
	stateRes := httptest.NewRecorder()
	handler.ServeHTTP(stateRes, stateReq)
	var states []struct {
		Kind   string `json:"kind"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal(stateRes.Body.Bytes(), &states); err != nil {
		t.Fatal(err)
	}
	for _, state := range states {
		if state.Status != "online" {
			t.Fatalf("engine %s did not use updated endpoint: %#v", state.Kind, states)
		}
	}
}

func TestPromptEnhancementRejectsI2VWhenVisionDisabled(t *testing.T) {
	cfg := config.Config{
		DataDir:           t.TempDir(),
		Engines:           map[string]config.Engine{"prompt": {Endpoint: "http://example.invalid"}},
		PromptEnhancement: config.PromptEnhancement{Model: "test-e2b", DefaultEnabled: true, VisionEnabled: false, MaxTokens: 600},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "고개를 들어 하늘을 본다")
	_ = form.WriteField("mode", "i2v")
	part, _ := form.CreateFormFile("image", "reference.png")
	_, _ = part.Write([]byte("fake image"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/enhance", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusConflict {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
}

func TestPromptEnhancementCallsOpenAICompatibleEngineForT2V(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Model               string `json:"model"`
			MaxCompletionTokens int    `json:"max_completion_tokens"`
			TopK                int    `json:"top_k"`
			ReasoningEffort     string `json:"reasoning_effort"`
			Messages            []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if r.URL.Path != "/v1/chat/completions" || json.NewDecoder(r.Body).Decode(&request) != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if request.Model != "test-e2b" || request.MaxCompletionTokens != 600 || request.TopK != 1 || request.ReasoningEffort != "none" || len(request.Messages) != 2 || len(request.Messages[0].Content) < 100 {
			t.Fatalf("unexpected enhancer request: %#v", request)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": map[string]string{"content": "*** A cinematic tracking shot follows the subject."}}},
		})
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir:           t.TempDir(),
		Engines:           map[string]config.Engine{"prompt": {Endpoint: worker.URL}},
		PromptEnhancement: config.PromptEnhancement{Model: "test-e2b", DefaultEnabled: true, MaxTokens: 600},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "빗속을 걷는 사람")
	_ = form.WriteField("mode", "t2v")
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/enhance", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var response struct {
		EnhancedPrompt string `json:"enhanced_prompt"`
		ImageUsed      bool   `json:"image_used"`
	}
	if err := json.Unmarshal(res.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.EnhancedPrompt != "A cinematic tracking shot follows the subject." || response.ImageUsed {
		t.Fatalf("unexpected response: %#v", response)
	}
}

func TestCustomVoiceUsesOpenAICompatibleRequest(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path != "/v1/audio/speech" {
			http.NotFound(w, r)
			return
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request["task_type"] != "CustomVoice" || request["voice"] != "sohee" || request["model"] != "test-custom" || request["instructions"] != "Speak warmly and slowly." || request["seed"] != float64(4242) {
			t.Fatalf("unexpected request %#v", request)
		}
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("fake wav"))
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"speech": {Endpoint: worker.URL}},
		Speech:  config.Speech{CustomVoiceModel: "test-custom", DefaultLanguage: "Korean", DefaultSpeaker: "Sohee"},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("text", "generated words")
	_ = form.WriteField("language", "Korean")
	_ = form.WriteField("speaker", "Sohee")
	_ = form.WriteField("instructions", "Speak warmly and slowly.")
	_ = form.WriteField("seed", "4242")
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/speech", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".wav"))
			if err != nil || string(got) != "fake wav" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestRecognitionUsesOpenAICompatibleRequest(t *testing.T) {
	asrWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if r.FormValue("model") != "test-asr" || r.FormValue("language") != "Korean" || r.FormValue("prompt") != "SparkTalk" {
			t.Fatalf("unexpected fields: %#v", r.MultipartForm.Value)
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(file)
		_ = file.Close()
		if string(data) != "fake audio" {
			t.Fatalf("unexpected audio %q", data)
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"text": "인식 결과", "language": "Korean"})
	}))
	defer asrWorker.Close()
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/media/prepare" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		archivePath := filepath.Join(t.TempDir(), "prepared.zip")
		archiveFile, err := os.Create(archivePath)
		if err != nil {
			t.Fatal(err)
		}
		archive := zip.NewWriter(archiveFile)
		manifest, _ := archive.Create("manifest.json")
		_, _ = manifest.Write([]byte(`{"source_name":"sample.mp4","asset":{"id":"0123456789abcdef0123456789abcdef","filename":"video.mp4","media_type":"video","content_type":"video/mp4","size":1024,"duration":1,"width":640,"height":360},"segments":[{"name":"segment-00000.wav","start":0,"end":1,"duration":1}]}`))
		segment, _ := archive.Create("segment-00000.wav")
		_, _ = segment.Write([]byte("fake audio"))
		_ = archive.Close()
		_ = archiveFile.Close()
		w.Header().Set("Content-Type", "application/zip")
		http.ServeFile(w, r, archivePath)
	}))
	defer mediaWorker.Close()

	cfg := config.Config{
		DataDir:     t.TempDir(),
		Engines:     map[string]config.Engine{"recognition": {Endpoint: asrWorker.URL}, "media": {Endpoint: mediaWorker.URL}},
		Recognition: config.Recognition{Model: "test-asr", DefaultLanguage: "Auto", MaxUploadMB: 1, SegmentSeconds: 30, DefaultOutputFormats: []string{"txt"}, DefaultTranslationMode: "none"},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("language", "Korean")
	_ = form.WriteField("context", "SparkTalk")
	part, _ := form.CreateFormFile("audio", "sample.wav")
	_, _ = part.Write([]byte("fake audio"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/recognition", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Params["text"] != "인식 결과" || list[0].Params["detected_language"] != "Korean" {
				t.Fatalf("unexpected result %#v", list[0])
			}
			if list[0].MediaAssetID != "0123456789abcdef0123456789abcdef" || list[0].MediaURL == "" || list[0].CaptionURL == "" {
				t.Fatalf("missing media result %#v", list[0])
			}
			media, ok := list[0].Params["media"].(map[string]any)
			if !ok || media["media_type"] != "video" || media["content_type"] != "video/mp4" {
				t.Fatalf("missing media metadata %#v", list[0].Params["media"])
			}
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".txt"))
			if err != nil || string(got) != "인식 결과\n" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			caption, err := os.ReadFile(store.OutputPath(list[0].ID + ".player.vtt"))
			if err != nil || !bytes.HasPrefix(caption, []byte("WEBVTT\n")) {
				t.Fatalf("caption=%q err=%v", caption, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

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
	for time.Now().Before(deadline) {
		completed := 0
		for _, job := range store.List() {
			if job.Status == "completed" {
				completed++
			}
		}
		if completed == 2 {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("queued subtitle jobs did not complete: %#v", store.List())
}

func TestMediaAssetProxyPreservesRangeAndJobDeleteRemovesAsset(t *testing.T) {
	const assetID = "0123456789abcdef0123456789abcdef"
	deleted := false
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/media/assets/"+assetID {
			http.NotFound(w, r)
			return
		}
		switch r.Method {
		case http.MethodGet:
			if got := r.Header.Get("Range"); got != "bytes=2-4" {
				t.Fatalf("Range = %q", got)
			}
			w.Header().Set("Accept-Ranges", "bytes")
			w.Header().Set("Content-Range", "bytes 2-4/6")
			w.Header().Set("Content-Type", "video/mp4")
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write([]byte("cde"))
		case http.MethodDelete:
			deleted = true
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer mediaWorker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"media": {Endpoint: mediaWorker.URL}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	job := jobs.Job{ID: "media-job", Kind: "recognition", Status: "completed", MediaAssetID: assetID, CreatedAt: time.Now()}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	streamReq := httptest.NewRequest(http.MethodGet, "/api/media/assets/"+assetID, nil)
	streamReq.Header.Set("Range", "bytes=2-4")
	streamRes := httptest.NewRecorder()
	handler.ServeHTTP(streamRes, streamReq)
	if streamRes.Code != http.StatusPartialContent || streamRes.Body.String() != "cde" {
		t.Fatalf("stream status=%d body=%q", streamRes.Code, streamRes.Body.String())
	}
	if got := streamRes.Header().Get("Content-Range"); got != "bytes 2-4/6" {
		t.Fatalf("Content-Range = %q", got)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/api/jobs/"+job.ID, nil)
	deleteRes := httptest.NewRecorder()
	handler.ServeHTTP(deleteRes, deleteReq)
	if deleteRes.Code != http.StatusNoContent || !deleted {
		t.Fatalf("delete status=%d remote deleted=%v", deleteRes.Code, deleted)
	}
	if _, ok := store.Get(job.ID); ok {
		t.Fatal("job remains after delete")
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
