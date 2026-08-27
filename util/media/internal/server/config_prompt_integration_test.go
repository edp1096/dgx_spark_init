package server

import (
	"bytes"
	"encoding/json"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
)

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
