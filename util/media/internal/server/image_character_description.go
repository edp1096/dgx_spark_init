package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

const imageCharacterDescriptionPrompt = `You are a forensic visual character describer for a recurring-character image generation system.
Return JSON only. Do not use markdown fences or commentary.

All supplied images are reference views or detail crops of one character. Describe the character from visible pixels, not from stereotypes or an invented backstory. Do not describe the current pose, facial expression, camera, crop, background, lighting, image quality, or art style as permanent identity.

Inspect systematically and record every visible repeatable feature, including when applicable:
- apparent subject type, apparent gender presentation and age range
- height impression, body build, proportions, silhouette, skin or surface material and tone
- face or head shape, forehead, cheeks, jaw, chin, ears, eyebrows, eyelashes, eye shape, iris color, pupils, nose, lips, teeth, facial hair, scars, freckles, moles and other marks
- makeup: foundation, eye makeup, eyeliner, eyeshadow, blush, lip treatment and distinctive placement
- hair: color, length, texture, density, parting, bangs, layers, tied sections, braids, ornaments and exact style
- neck, shoulders, torso, arms, hands, fingers, fingernail length, shape and color
- upper garments, layers, neckline, sleeves, cuffs, fasteners, seams, straps, suspenders, patterns, materials, colors and fit
- lower garments, belt, stockings, tights, socks, leg details and visible layering
- footwear, heel or sole form, straps, laces, materials, colors, exposed feet and toenails
- glasses, earrings, necklaces, bracelets, watches, rings, piercings, bags, stable carried items and every other accessory
- left/right asymmetry and distinctive identifiers
- for robots or non-human subjects: exact head and faceplate geometry, eye count and placement, antennae, panels, chassis segmentation, limb and joint design, wheel count and placement, materials, colors, markings and scale

Use "not visible" or "uncertain" rather than guessing. If references conflict, report the conflict and prefer details repeated across views. The Korean description and observations inspect all visible categories for review. The user message contains LOCKED TRAITS. canonical_prompt_en must contain only the selected locked categories plus the minimum subject type needed for grammatical coherence; omit unselected clothing, footwear, accessories or mechanical details even when visible. It is a dense natural English visual specification, approximately 100-450 words, suitable for verbatim reuse in every scene prompt. Do not use phrases such as same as before, reference image, character sheet, image one, or previous scene.

Return exactly this shape:
{"name_ko":"...","name_en":"...","description_ko":"...","canonical_prompt_en":"...","observations":{"basic":"...","face":"...","makeup":"...","hair":"...","body":"...","hands_and_nails":"...","upper_clothing":"...","lower_clothing":"...","hosiery":"...","footwear_and_feet":"...","accessories":"...","asymmetry_and_identifiers":"...","mechanical_geometry":"...","uncertain_or_not_visible":"...","other_visible_details":"..."}}`

type imageCharacterDescription struct {
	NameKO            string            `json:"name_ko"`
	NameEN            string            `json:"name_en"`
	DescriptionKO     string            `json:"description_ko"`
	CanonicalPromptEN string            `json:"canonical_prompt_en"`
	Observations      map[string]string `json:"observations"`
}

func (s *Server) describeImageSequenceCharacter(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(128 << 20); err != nil {
		http.Error(w, "invalid or oversized character reference form", http.StatusBadRequest)
		return
	}
	tempParent := filepath.Join(s.dataDir, "temp")
	if err := os.MkdirAll(tempParent, 0o755); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	root, err := os.MkdirTemp(tempParent, "character-description-")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer os.RemoveAll(root)
	paths, err := saveUploads(r, "images", root, 6)
	if err == nil {
		paths, err = s.appendReusedImageInputs(r, "reuse_images", root, 6, paths)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if len(paths) == 0 {
		http.Error(w, "at least one character reference image is required", http.StatusBadRequest)
		return
	}

	lockedTraits, err := parseImageCharacterLockedTraits(r.FormValue("locked_traits"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	content := make([]map[string]any, 0, len(paths)*2+2)
	requestedName := strings.TrimSpace(r.FormValue("name"))
	content = append(content, map[string]any{"type": "text", "text": "Requested character name: " + valueOr(requestedName, "choose a short neutral name from visible evidence")})
	content = append(content, map[string]any{"type": "text", "text": "LOCKED TRAITS for canonical_prompt_en: " + strings.Join(lockedTraits, ", ") + ". Analyze every category in observations, but include only these categories in canonical_prompt_en."})
	for index, path := range paths {
		dataURL, encodeErr := visualReferenceDataURL(path)
		if encodeErr != nil {
			http.Error(w, "character reference image is invalid: "+encodeErr.Error(), http.StatusBadRequest)
			return
		}
		content = append(content,
			map[string]any{"type": "text", "text": fmt.Sprintf("Reference view or detail crop %d:", index+1)},
			map[string]any{"type": "image_url", "image_url": map[string]string{"url": dataURL}},
		)
	}
	cfg := s.config()
	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]any{
			{"role": "system", "content": imageCharacterDescriptionPrompt},
			{"role": "user", "content": content},
		},
		"max_completion_tokens": 2200,
		"temperature":           0,
		"top_k":                 1,
		"seed":                  42,
		"reasoning_effort":      "none",
	}
	data, err := s.chatWithPromptEngine(payload)
	if err != nil {
		http.Error(w, "character description: "+err.Error(), http.StatusBadGateway)
		return
	}
	contentText, err := openAIMessageContent(data)
	if err != nil {
		http.Error(w, "character description returned an invalid response", http.StatusBadGateway)
		return
	}
	var result imageCharacterDescription
	if err := json.Unmarshal([]byte(extractJSONObject(contentText)), &result); err != nil {
		http.Error(w, "character description returned invalid JSON", http.StatusBadGateway)
		return
	}
	result.NameKO = strings.TrimSpace(result.NameKO)
	result.NameEN = strings.TrimSpace(result.NameEN)
	result.DescriptionKO = strings.TrimSpace(result.DescriptionKO)
	result.CanonicalPromptEN = strings.TrimSpace(result.CanonicalPromptEN)
	if requestedName != "" {
		result.NameKO = requestedName
	}
	if result.NameKO == "" || result.NameEN == "" || result.DescriptionKO == "" || result.CanonicalPromptEN == "" {
		http.Error(w, "character description returned incomplete identity data", http.StatusBadGateway)
		return
	}
	if len([]rune(result.CanonicalPromptEN)) > 10000 || len([]rune(result.DescriptionKO)) > 10000 {
		http.Error(w, "character description is too long", http.StatusBadGateway)
		return
	}
	if result.Observations == nil {
		result.Observations = map[string]string{}
	}
	writeJSON(w, http.StatusOK, result)
}

func parseImageCharacterLockedTraits(raw string) ([]string, error) {
	allowed := map[string]bool{"face": true, "hair": true, "body": true, "outfit": true, "accessories": true, "mechanical": true}
	values := []string{"face", "hair", "body", "outfit", "mechanical"}
	if strings.TrimSpace(raw) != "" {
		if err := json.Unmarshal([]byte(raw), &values); err != nil {
			return nil, fmt.Errorf("invalid locked character traits")
		}
	}
	result := make([]string, 0, len(values))
	seen := map[string]bool{}
	for _, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if !allowed[value] {
			return nil, fmt.Errorf("unsupported locked character trait")
		}
		if !seen[value] {
			result = append(result, value)
			seen[value] = true
		}
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("select at least one locked character trait")
	}
	return result, nil
}

func visualReferenceDataURL(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	decoded, _, err := image.Decode(file)
	_ = file.Close()
	if err != nil {
		return "", err
	}
	var output bytes.Buffer
	if err := jpeg.Encode(&output, decoded, &jpeg.Options{Quality: 92}); err != nil {
		return "", err
	}
	return "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(output.Bytes()), nil
}
