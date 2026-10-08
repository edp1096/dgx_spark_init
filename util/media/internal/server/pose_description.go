package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"os"
	"strings"

	_ "golang.org/x/image/webp"
)

// describePoseReference converts arbitrary uploaded image formats to JPEG and
// asks the local multimodal prompt model for the semantic pose description that
// Krea Depth Control needs. Flat illustrations otherwise tend to lose their
// limb arrangement during the realistic-reference preparation pass.
func (s *Server) describeVisualReference(path, instruction string, maxTokens int) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	decoded, _, err := image.Decode(file)
	_ = file.Close()
	if err != nil {
		return "", fmt.Errorf("decode pose reference: %w", err)
	}
	var jpegImage bytes.Buffer
	if err := jpeg.Encode(&jpegImage, decoded, &jpeg.Options{Quality: 90}); err != nil {
		return "", fmt.Errorf("encode pose reference: %w", err)
	}
	dataURL := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(jpegImage.Bytes())
	cfg := s.config()
	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]any{{
			"role": "user",
			"content": []map[string]any{
				{"type": "text", "text": instruction},
				{"type": "image_url", "image_url": map[string]string{"url": dataURL}},
			},
		}},
		"temperature":           0,
		"max_completion_tokens": maxTokens,
		"reasoning_effort":      "none",
	}
	data, err := s.chatWithPromptEngine(payload)
	if err != nil {
		return "", err
	}
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &completion); err != nil || len(completion.Choices) == 0 {
		return "", fmt.Errorf("pose description model returned an invalid response")
	}
	description := strings.TrimSpace(completion.Choices[0].Message.Content)
	if description == "" {
		return "", fmt.Errorf("pose description model returned an empty response")
	}
	return description, nil
}

func (s *Server) describePoseReference(path string) (string, error) {
	return s.describeVisualReference(
		path,
		"Describe only the exact human body geometry: body orientation, every visible limb and joint position, camera view, and framing. Ignore clothing, colors, identity, background, and objects. Do not use any named pose, exercise, dance, sport, yoga/asana label, metaphor, or symbolic pose name. Describe the joint positions literally in one concise English sentence suitable for structural image control.",
		120,
	)
}

func (s *Server) describeOutfitReference(path string) (string, error) {
	return s.describeVisualReference(
		path,
		"Name the single complete garment using only its main color, main material, and exact garment type. Return one 3-to-6-word English noun phrase beginning with a or an, such as a red leather jacket or a black lace bodysuit. Omit patterns, straps, fasteners, coverage, styling, people, body, pose, and background.",
		40,
	)
}

func (s *Server) describeSubjectPronoun(path string) (string, error) {
	return s.describeVisualReference(
		path,
		"Return exactly one lowercase English subject pronoun for the main visible subject: she, he, they, or it. Use they when a human pronoun is uncertain and it for a non-human subject. Return no other words or punctuation.",
		10,
	)
}

// composeIdentityEditPrompt lets the same local Gemma model merge a user's
// extra edit into the concrete module instruction. Krea Identity Edit is highly
// sensitive to instruction shape: appending a third imperative line made the
// original shirt reappear, while one compact change sentence plus the fixed pose
// sentence retained the complete replacement outfit.
func (s *Server) composeIdentityEditPrompt(pronoun, verb, outfit, userInstruction string, hasPose bool) (string, error) {
	cfg := s.config()
	base := pronoun + " " + verb + " now wearing " + outfit
	pose := ""
	if hasPose {
		pose = pronoun + " now holds the same pose."
	}
	system := `Merge the supplied REQUIRED MODULE CLAUSE and EXTRA USER CHANGE into one short, direct English sentence for Krea 2 Identity Edit. Keep the required module clause verbatim at the beginning, in lowercase. Attach the extra change grammatically to that same sentence, normally with "and", without adding preservation language, source-image descriptions, clothing not named in the clause, explanations, headings, or extra sentences. If a REQUIRED FINAL LINE is supplied, output it verbatim as the second and final line. Output only the resulting one or two lines.`
	user := "REQUIRED MODULE CLAUSE: " + base + "\nEXTRA USER CHANGE: " + strings.TrimSpace(userInstruction)
	if pose != "" {
		user += "\nREQUIRED FINAL LINE: " + pose
	}
	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]any{
			{"role": "system", "content": system},
			{"role": "user", "content": user},
		},
		"max_completion_tokens": 160,
		"temperature":           0,
		"top_k":                 1,
		"seed":                  42,
		"reasoning_effort":      "none",
	}
	data, err := s.chatWithPromptEngine(payload)
	if err != nil {
		return "", err
	}
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
		return "", fmt.Errorf("identity prompt composer returned an invalid response")
	}
	composed := cleanEnhancedPrompt(response.Choices[0].Message.Content)
	if !strings.HasPrefix(strings.ToLower(composed), strings.ToLower(base)) {
		return "", fmt.Errorf("identity prompt composer omitted the module clause")
	}
	if pose != "" && !strings.HasSuffix(strings.TrimSpace(composed), pose) {
		return "", fmt.Errorf("identity prompt composer omitted the pose line")
	}
	return composed, nil
}
