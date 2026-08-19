package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestImageJobCompletesThroughEngine(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.WriteHeader(http.StatusOK)
		case "/v1/images/generations":
			var request struct {
				Prompt string `json:"prompt"`
				Size   string `json:"size"`
			}
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Fatal(err)
			}
			if request.Prompt != "green glass sphere" || request.Size != "512x512" {
				t.Fatalf("unexpected request %#v", request)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString([]byte("fake png"))}}})
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
			"speech": {Endpoint: first.URL}, "recognition": {Endpoint: first.URL}, "prompt": {Endpoint: first.URL},
		},
		Image:             config.Image{Model: "image", DefaultWidth: 512, DefaultHeight: 512, MaxReferenceImages: 4},
		Video:             config.Video{Model: "video", DefaultWidth: 768, DefaultHeight: 512, DefaultFrames: 121, DefaultFPS: 24},
		Speech:            config.Speech{CustomVoiceModel: "speech", DefaultLanguage: "Korean", DefaultSpeaker: "Sohee"},
		Recognition:       config.Recognition{Model: "asr", DefaultLanguage: "Auto", MaxUploadMB: 500},
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
		"speech": {Endpoint: second.URL}, "recognition": {Endpoint: second.URL}, "prompt": {Endpoint: second.URL},
	}
	next.Video.DefaultFrames = 65
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
	if loaded.Video.DefaultFrames != 65 || loaded.Engines["video"].Endpoint != second.URL {
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

func TestPromptEnhancementCallsLiteRTForT2V(t *testing.T) {
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
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	defer worker.Close()

	cfg := config.Config{
		DataDir:     t.TempDir(),
		Engines:     map[string]config.Engine{"recognition": {Endpoint: worker.URL}},
		Recognition: config.Recognition{Model: "test-asr", DefaultLanguage: "Auto", MaxUploadMB: 1},
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
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".txt"))
			if err != nil || string(got) != "인식 결과\n" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}
