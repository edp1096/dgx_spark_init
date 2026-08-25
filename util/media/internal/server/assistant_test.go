package server

import (
	"encoding/json"
	"image"
	"image/color"
	"image/png"
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

func TestAssistantChatAppliesOnlyAllowedControls(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request["model"] != "huihui-gemma4-12b" {
			t.Fatalf("model=%v", request["model"])
		}
		content := `{"reply":"이미지 설정을 준비했습니다.","actions":[{"type":"set_image","prompt":"a red fox","width":1024,"height":1024,"seed":7},{"type":"shell","prompt":"rm -rf /"}],"confirmation":"image"}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": content}}}})
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{
		DataDir:           dataDir,
		Engines:           map[string]config.Engine{"prompt": {Endpoint: engine.URL}},
		PromptEnhancement: config.PromptEnhancement{Model: "huihui-gemma4-12b"},
	}
	handler := New(cfg, store, nil).Handler()
	req := httptest.NewRequest(http.MethodPost, "/api/assistant/chat", strings.NewReader(`{"messages":[{"role":"user","content":"붉은 여우를 만들어줘"}],"state":{"tab":"image"}}`))
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var result assistantChatResponse
	if err := json.NewDecoder(res.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if result.Confirmation != "image" || len(result.Actions) != 1 || result.Actions[0].Type != "set_image" || result.Actions[0].Prompt != "a red fox" {
		t.Fatalf("result=%#v", result)
	}
}

func TestAssistantChatAttachesNumberedContactSheetForVisualQuestion(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if len(request.Messages) != 2 {
			t.Fatalf("messages=%#v", request.Messages)
		}
		content, ok := request.Messages[1].Content.([]any)
		if !ok || len(content) != 2 {
			t.Fatalf("multimodal content=%#v", request.Messages[1].Content)
		}
		imagePart, ok := content[0].(map[string]any)
		if !ok || imagePart["type"] != "image_url" {
			t.Fatalf("image part=%#v", content[0])
		}
		imageURL := imagePart["image_url"].(map[string]any)["url"].(string)
		if !strings.HasPrefix(imageURL, "data:image/jpeg;base64,") {
			t.Fatalf("image URL prefix=%q", imageURL[:min(len(imageURL), 40)])
		}
		response := `{"reply":"연락처 시트를 직접 확인한 결과 #1에 파란 이미지가 있습니다.","actions":[]}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": response}}}})
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	canvas := image.NewRGBA(image.Rect(0, 0, 80, 60))
	for y := 0; y < 60; y++ {
		for x := 0; x < 80; x++ {
			canvas.Set(x, y, color.RGBA{B: 220, A: 255})
		}
	}
	outputName := "vision-source.png"
	file, err := os.Create(filepath.Join(store.OutputDir(), outputName))
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(file, canvas); err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	job := jobs.Job{ID: "vision-source", Kind: "image", Status: "completed", Prompt: "blue square", OutputURL: "/api/outputs/" + outputName, CreatedAt: time.Now()}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{
		DataDir:           dataDir,
		Engines:           map[string]config.Engine{"prompt": {Endpoint: engine.URL}},
		PromptEnhancement: config.PromptEnhancement{Model: "huihui-gemma4-12b"},
	}
	handler := New(cfg, store, nil).Handler()
	body := `{"messages":[{"role":"user","content":"어떤 이미지가 파란색으로 보이냐?"}],"state":{"tab":"image","recent_images":[{"index":1,"job_id":"vision-source","status":"completed","prompt":"blue square"}]}}`
	req := httptest.NewRequest(http.MethodPost, "/api/assistant/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var result assistantChatResponse
	if err := json.NewDecoder(res.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if !result.VisionUsed || !strings.Contains(result.Reply, "#1") {
		t.Fatalf("result=%#v", result)
	}
}

func TestAssistantChatAttachesVideoConditioningSheetForPromptAdvice(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Role    string `json:"role"`
				Content any    `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if len(request.Messages) != 2 {
			t.Fatalf("messages=%#v", request.Messages)
		}
		parts, ok := request.Messages[1].Content.([]any)
		if !ok || len(parts) != 2 {
			t.Fatalf("multimodal content=%#v", request.Messages[1].Content)
		}
		imagePart := parts[0].(map[string]any)
		imageURL := imagePart["image_url"].(map[string]any)["url"].(string)
		if !strings.HasPrefix(imageURL, "data:image/jpeg;base64,") {
			t.Fatalf("image URL=%q", imageURL)
		}
		text := parts[1].(map[string]any)["text"].(string)
		if !strings.Contains(text, "START·KEYFRAME·END") {
			t.Fatalf("instruction=%q", text)
		}
		response := `{"reply":"두 장면 사이를 잇는 카메라 이동을 적용했습니다.","actions":[{"type":"set_video","prompt":"The subject turns smoothly as the camera arcs forward, preserving identity and scene continuity before settling into the final composition.","width":768,"height":512,"fps":24,"duration":5}]}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": response}}}})
	}))
	defer engine.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{
		DataDir:           dataDir,
		Engines:           map[string]config.Engine{"prompt": {Endpoint: engine.URL}},
		PromptEnhancement: config.PromptEnhancement{Model: "huihui-gemma4-12b"},
	}
	handler := New(cfg, store, nil).Handler()
	body := `{"messages":[{"role":"user","content":"시작과 마지막 이미지에 어떤 프롬프트가 좋을까?"}],"state":{"tab":"video","video":{"has_start_image":true,"has_end_image":true}},"visual_context":{"kind":"video_conditioning","image_url":"data:image/jpeg;base64,/9j/2Q==","labels":["START 0초","END 5초"]}}`
	req := httptest.NewRequest(http.MethodPost, "/api/assistant/chat", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var result assistantChatResponse
	if err := json.NewDecoder(res.Body).Decode(&result); err != nil {
		t.Fatal(err)
	}
	if !result.VisionUsed || len(result.Actions) != 1 || result.Actions[0].Type != "set_video" || !strings.Contains(result.Actions[0].Prompt, "camera") {
		t.Fatalf("result=%#v", result)
	}
	if result.Actions[0].Width != 0 || result.Actions[0].Height != 0 || result.Actions[0].FPS != 0 || result.Actions[0].Duration != 0 || !strings.Contains(result.Reply, result.Actions[0].Prompt) {
		t.Fatalf("prompt advice normalization failed: %#v", result)
	}
}

func TestAssistantChatRejectsMissingMessages(t *testing.T) {
	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: dataDir}, store, nil).Handler()
	req := httptest.NewRequest(http.MethodPost, "/api/assistant/chat", strings.NewReader(`{"messages":[]}`))
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
}

func TestNormalizeAssistantOutpaintOverridesWrongModule(t *testing.T) {
	result := assistantChatResponse{
		Reply:        "윤곽 편집을 준비했습니다.",
		Actions:      []assistantAction{{Type: "set_module", Module: "nk2e", Enabled: boolPointer(true)}},
		Confirmation: "image",
	}
	request := assistantChatRequest{
		Messages: []assistantChatMessage{{Role: "user", Content: "이미지15를 좌우 64px 늘리고 싶다."}},
		State: map[string]any{"recent_images": []any{
			map[string]any{"index": float64(15), "job_id": "image-15", "status": "completed"},
		}},
	}
	result = normalizeAssistantOutpaint(result, request)
	if len(result.Actions) != 1 {
		t.Fatalf("actions=%#v", result.Actions)
	}
	action := result.Actions[0]
	if action.Type != "set_outpaint" || action.ImageIndex != 15 || action.OutpaintLeft != 64 || action.OutpaintRight != 64 || action.OutpaintTop != 0 || action.OutpaintBottom != 0 {
		t.Fatalf("action=%#v", action)
	}
	if result.Confirmation != "image" || !strings.Contains(result.Reply, "프롬프트 없이") {
		t.Fatalf("result=%#v", result)
	}
}

func boolPointer(value bool) *bool { return &value }
