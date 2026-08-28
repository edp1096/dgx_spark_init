package server

import (
	"bytes"
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
	"strings"
	"testing"
)

func TestCharacterDescriptionUsesVisualInputAndReturnsCanonicalProfile(t *testing.T) {
	promptEngine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if !bytes.Contains(body, []byte("data:image/jpeg;base64,")) || !bytes.Contains(body, []byte("forensic visual character describer")) {
			t.Fatalf("character request did not contain a visual reference and system contract: %s", body)
		}
		content := `{"name_ko":"모모","name_en":"Momo","description_ko":"짧은 검은 단발과 녹색 눈, 남색 코트","canonical_prompt_en":"Momo is a young Korean woman with an oval face, vivid green irises, a short blunt black bob, and a fitted navy wool coat with silver buttons.","observations":{"face":"타원형 얼굴과 녹색 눈","hair":"짧은 검은 단발"}}`
		_ = json.NewEncoder(w).Encode(map[string]any{"choices": []map[string]any{{"message": map[string]string{"content": content}}}})
	}))
	defer promptEngine.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"prompt": {Endpoint: promptEngine.URL}}, PromptEnhancement: config.PromptEnhancement{Model: "gemma-vision"}}
	store, _ := jobs.New(cfg.DataDir)
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("name", "주인공")
	_ = form.WriteField("locked_traits", `["face","hair"]`)
	part, _ := form.CreateFormFile("images", "character.png")
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for y := 0; y < 4; y++ {
		for x := 0; x < 4; x++ {
			img.Set(x, y, color.RGBA{R: 80, G: 160, B: 220, A: 255})
		}
	}
	_ = png.Encode(part, img)
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/character-description", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	New(cfg, store, nil).Handler().ServeHTTP(res, req)
	if res.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}
	var result imageCharacterDescription
	_ = json.Unmarshal(res.Body.Bytes(), &result)
	if result.NameKO != "주인공" || !strings.Contains(result.CanonicalPromptEN, "green irises") || result.Observations["hair"] == "" {
		t.Fatalf("unexpected description: %#v", result)
	}
}
