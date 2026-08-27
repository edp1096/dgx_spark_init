package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
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
