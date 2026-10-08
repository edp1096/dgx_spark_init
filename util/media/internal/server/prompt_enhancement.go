package server

import (
	"encoding/base64"
	"encoding/json"
	"io"
	mediaprompt "mediaapp/internal/prompt"
	"mime"
	"net/http"
	"path/filepath"
	"strings"
)

func (s *Server) enhancePrompt(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(40 << 20); err != nil {
		http.Error(w, "invalid or oversized form", http.StatusBadRequest)
		return
	}
	original := strings.TrimSpace(r.FormValue("prompt"))
	if original == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}
	mode := strings.ToLower(strings.TrimSpace(r.FormValue("mode")))
	if mode == "" {
		mode = "t2v"
	}
	if mode != "t2v" && mode != "t2v_wildcard" && mode != "i2v" && mode != "t2i" && mode != "edit" && mode != "edit_control" && mode != "control" && mode != "paint" {
		http.Error(w, "mode must be t2i, edit, edit_control, control, paint, t2v, t2v_wildcard or i2v", http.StatusBadRequest)
		return
	}
	if mode == "i2v" && !cfg.PromptEnhancement.VisionEnabled {
		http.Error(w, "I2V prompt enhancement requires a vision-enabled model bundle", http.StatusConflict)
		return
	}

	visionRequested := mode == "i2v" && cfg.PromptEnhancement.VisionEnabled
	userContent := any("User Raw Input Prompt: " + original)
	imageUsed := false
	if visionRequested {
		if file, header, err := r.FormFile("image"); err == nil {
			defer file.Close()
			data, readErr := io.ReadAll(io.LimitReader(file, (32<<20)+1))
			if readErr != nil || len(data) > 32<<20 {
				http.Error(w, "reference image is invalid or too large", http.StatusBadRequest)
				return
			}
			contentType := header.Header.Get("Content-Type")
			if contentType == "" {
				contentType = mime.TypeByExtension(strings.ToLower(filepath.Ext(header.Filename)))
			}
			if contentType == "" {
				contentType = http.DetectContentType(data)
			}
			userContent = []map[string]any{
				{"type": "image_url", "image_url": map[string]string{"url": "data:" + contentType + ";base64," + base64.StdEncoding.EncodeToString(data)}},
				{"type": "text", "text": "User Raw Input Prompt: " + original},
			}
			imageUsed = true
		} else {
			http.Error(w, "reference image is required for I2V prompt enhancement", http.StatusBadRequest)
			return
		}
	}

	systemPrompt := mediaprompt.System(mode, imageUsed)
	if mode == "edit" || mode == "edit_control" {
		preset := strings.TrimSpace(r.FormValue("identity_preset"))
		validPresets := map[string]bool{"": true, "restage": true, "sheet": true, "faceSwap": true, "headSwap": true, "personSwap": true, "tryon": true, "replace": true}
		if !validPresets[preset] {
			http.Error(w, "unsupported identity preset", http.StatusBadRequest)
			return
		}
		preserved := []string{}
		if raw := strings.TrimSpace(r.FormValue("identity_preserve_items")); raw != "" {
			if err := json.Unmarshal([]byte(raw), &preserved); err != nil {
				http.Error(w, "invalid identity preservation selection", http.StatusBadRequest)
				return
			}
			allowed := map[string]bool{"identity": true, "face": true, "hair": true, "body": true, "clothing": true, "pose": true, "background": true, "lighting": true, "composition": true, "untouched": true}
			for _, item := range preserved {
				if !allowed[item] {
					http.Error(w, "invalid identity preservation item", http.StatusBadRequest)
					return
				}
			}
		}
		systemPrompt += mediaprompt.EditModuleContext(preset, preserved)
	}
	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]any{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": userContent},
		},
		"max_completion_tokens": cfg.PromptEnhancement.MaxTokens,
		"temperature":           0,
		"top_k":                 1,
		"seed":                  42,
		"reasoning_effort":      "none",
	}
	data, err := s.chatWithPromptEngine(payload)
	if err != nil {
		http.Error(w, "prompt enhancer: "+err.Error(), http.StatusBadGateway)
		return
	}
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
		http.Error(w, "prompt enhancer returned an invalid response", http.StatusBadGateway)
		return
	}
	enhanced := cleanEnhancedPrompt(response.Choices[0].Message.Content)
	if enhanced == "" || strings.EqualFold(enhanced, "IMAGE_NOT_AVAILABLE") {
		http.Error(w, "prompt enhancer returned no usable prompt", http.StatusBadGateway)
		return
	}
	fallback := false
	if !enhancementPreservesEditContract(mode, original, enhanced) {
		enhanced = original
		fallback = true
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"original_prompt": original,
		"enhanced_prompt": enhanced,
		"mode":            mode,
		"image_used":      imageUsed,
		"fallback":        fallback,
	})
}

func enhancementPreservesEditContract(mode, original, enhanced string) bool {
	if mode != "edit" && mode != "edit_control" {
		return true
	}
	source := strings.ToLower(original)
	result := strings.ToLower(enhanced)
	requireAny := func(trigger string, words ...string) bool {
		if !strings.Contains(source, trigger) {
			return true
		}
		for _, word := range words {
			if strings.Contains(result, word) {
				return true
			}
		}
		return false
	}
	if !requireAny("supporting reference", "supporting reference", "reference image", "reference outfit") ||
		!requireAny("depth", "depth") ||
		!requireAny("pose", "pose", "posture", "body orientation") ||
		!requireAny("clothing", "clothing", "outfit", "garment") {
		return false
	}
	if strings.Contains(source, "do not preserve") || strings.Contains(source, "do not retain") || strings.Contains(source, "do not restore") || strings.Contains(source, "may change") {
		for _, contradiction := range []string{
			"preserve original pose", "preserve the original pose", "preserving original pose", "preserving the original pose", "retain original pose", "retain the original pose", "maintain original pose", "maintain the original pose", "original pose remains", "original pose unchanged",
			"preserve original clothing", "preserve the original clothing", "preserving original clothing", "preserving the original clothing", "retain original clothing", "retain the original clothing", "maintain original clothing", "maintain the original clothing", "original clothing remains", "original clothing unchanged",
			"preserve original outfit", "preserve the original outfit", "preserving original outfit", "preserving the original outfit", "retain original outfit", "retain the original outfit", "maintain original outfit", "maintain the original outfit", "original outfit remains", "original outfit unchanged",
		} {
			if strings.Contains(result, contradiction) {
				return false
			}
		}
	}
	return true
}
