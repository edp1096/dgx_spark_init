package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

const imageSequencePlannerPrompt = `You are a storyboard and illustration-set prompt compiler for Krea 2 image generation.
Return JSON only. Do not use markdown fences or commentary.

Input contains either 2-12 Korean scene descriptions, or one Korean story outline and a requested scene count. It may also contain shared_prompt, a user-edited Korean continuity bible that is authoritative, and locked_characters made from user-approved visual analysis.
Output exactly this shape:
{"shared_prompt":"...","world_prompt_en":"...","characters":[{"id":"character_1","name_ko":"...","name_en":"...","description_ko":"...","prompt_en":"..."}],"scenes":[{"scene_title":"...","original_prompt":"...","character_ids":["character_1"],"scene_prompt_en":"...","strategy":"storyboard","change_summary":"..."}]}

Rules:
- For scene descriptions, produce exactly one output scene per input, in the same order.
- For a story outline, divide it into exactly scene_count visually distinct moments with a clear beginning, development, and ending.
- shared_prompt is a detailed but readable Korean continuity bible covering stable cast appearance, wardrobe, recurring props, world, period, and visual style. If the input includes shared_prompt, preserve its meaning exactly and treat it as authoritative.
- Create one characters entry for every recurring visible character. Use short stable ids. name_en is a short natural English proper name or role name with no underscore. description_ko is for user review. prompt_en is the canonical English visual identity block and should introduce name_en naturally.
- locked_characters are authoritative. Copy every locked character field byte-for-byte into characters. Never shorten, translate, paraphrase, correct, or replace a locked prompt_en. Use its id in each scene where that character is visibly present.
- Each character prompt_en must use concrete, repeatable visual anchors: apparent age, height or scale, body proportions, material or skin tone, face or head shape, hair, eyes, distinctive marks, stable clothing, accessories, and mechanical geometry when relevant.
- Keep character prompt_en free of temporary pose, action, expression, camera, scene lighting, and location. Use identical wording once; the server will inject it verbatim into every scene where that character appears.
- world_prompt_en contains only stable English world and style facts that should apply to every image. Do not put scene-specific time, weather, location, camera, or action in it.
- character_ids lists only the characters visibly present in that scene and must use ids from characters.
- scene_prompt_en is an English scene-only prompt. Refer to present characters by name_en exactly and describe action, pose, important object contact, environment, lighting, camera distance, viewpoint, composition, and mood. Never use internal character ids as names. Do not repeat or contradict their visual identity blocks.
- Never write “same as before”, “the same character”, “previously described”, pronoun-only identity references, edit commands, preservation policies, previous/next-frame references, or negative instructions.
- Locations, time of day, action, composition, camera distance, and viewpoint may change naturally between scenes unless the user explicitly fixes them.
- Do not promise temporal interpolation or near-identical frames. These are independent storyboard illustrations, not animation keyframes.
- scene_title and change_summary are short Korean labels.
- For supplied scene descriptions, original_prompt must reproduce the matching input exactly. For a story outline, original_prompt must be a concise Korean description of that scene.
- strategy is always storyboard.`

type imageSequencePlanRequest struct {
	Prompts          []string                 `json:"prompts"`
	Outline          string                   `json:"outline"`
	SceneCount       int                      `json:"scene_count"`
	SharedPrompt     string                   `json:"shared_prompt,omitempty"`
	LockedCharacters []imageSequenceCharacter `json:"locked_characters,omitempty"`
}

type imageSequenceCharacter struct {
	ID            string `json:"id"`
	NameKO        string `json:"name_ko"`
	NameEN        string `json:"name_en"`
	DescriptionKO string `json:"description_ko"`
	PromptEN      string `json:"prompt_en"`
}

type imageSequencePlannedScene struct {
	SceneTitle     string   `json:"scene_title"`
	OriginalPrompt string   `json:"original_prompt"`
	EnhancedPrompt string   `json:"enhanced_prompt"`
	CharacterIDs   []string `json:"character_ids"`
	ScenePromptEN  string   `json:"scene_prompt_en"`
	Strategy       string   `json:"strategy"`
	ChangeSummary  string   `json:"change_summary"`
}

type imageSequencePromptPlan struct {
	SharedPrompt    string                      `json:"shared_prompt"`
	CanonicalPrompt string                      `json:"canonical_prompt_en"`
	WorldPromptEN   string                      `json:"world_prompt_en"`
	Characters      []imageSequenceCharacter    `json:"characters"`
	Scenes          []imageSequencePlannedScene `json:"scenes"`
}

func (s *Server) planImageSequence(w http.ResponseWriter, r *http.Request) {
	var request imageSequencePlanRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 256<<10)).Decode(&request); err != nil {
		http.Error(w, "invalid sequence plan request", http.StatusBadRequest)
		return
	}
	prompts, outline, sharedPrompt, sceneCount, err := validateSequencePlannerRequest(request)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	lockedCharacters, err := validateLockedSequenceCharacters(request.LockedCharacters)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	input, _ := json.Marshal(imageSequencePlanRequest{Prompts: prompts, Outline: outline, SceneCount: sceneCount, SharedPrompt: sharedPrompt, LockedCharacters: lockedCharacters})
	cfg := s.config()
	maxTokens := cfg.PromptEnhancement.MaxTokens
	if maxTokens < 3200 {
		maxTokens = 3200
	}
	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]any{
			{"role": "system", "content": imageSequencePlannerPrompt},
			{"role": "user", "content": string(input)},
		},
		"max_completion_tokens": maxTokens,
		"temperature":           0,
		"top_k":                 1,
		"seed":                  42,
		"reasoning_effort":      "none",
	}
	data, err := s.chatWithPromptEngine(payload)
	if err != nil {
		http.Error(w, "sequence planner: "+err.Error(), http.StatusBadGateway)
		return
	}
	content, err := openAIMessageContent(data)
	if err != nil {
		http.Error(w, "sequence planner returned an invalid response", http.StatusBadGateway)
		return
	}
	var plan imageSequencePromptPlan
	if err := json.Unmarshal([]byte(extractJSONObject(content)), &plan); err != nil {
		http.Error(w, "sequence planner returned invalid JSON", http.StatusBadGateway)
		return
	}
	reconcileLockedSequenceCharacters(&plan, lockedCharacters)
	if err := validateSequencePromptPlan(&plan, prompts, sharedPrompt, sceneCount); err != nil {
		http.Error(w, "sequence planner: "+err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, plan)
}

func validateLockedSequenceCharacters(values []imageSequenceCharacter) ([]imageSequenceCharacter, error) {
	if len(values) > 4 {
		return nil, fmt.Errorf("at most four locked characters are supported")
	}
	result := make([]imageSequenceCharacter, len(values))
	seen := make(map[string]bool, len(values))
	totalPromptLength := 0
	for index, value := range values {
		value.ID = strings.ToLower(strings.TrimSpace(value.ID))
		value.NameKO = strings.TrimSpace(value.NameKO)
		value.NameEN = strings.TrimSpace(value.NameEN)
		value.DescriptionKO = strings.TrimSpace(value.DescriptionKO)
		value.PromptEN = strings.TrimSpace(value.PromptEN)
		if value.ID == "" || value.NameKO == "" || value.NameEN == "" || value.DescriptionKO == "" || value.PromptEN == "" {
			return nil, fmt.Errorf("every locked character requires names and approved descriptions")
		}
		for _, char := range value.ID {
			if !((char >= 'a' && char <= 'z') || (char >= '0' && char <= '9') || char == '_') {
				return nil, fmt.Errorf("locked character id contains unsupported characters")
			}
		}
		if seen[value.ID] {
			return nil, fmt.Errorf("duplicate locked character id")
		}
		if len([]rune(value.DescriptionKO)) > 10000 || len([]rune(value.PromptEN)) > 10000 {
			return nil, fmt.Errorf("locked character description is too long")
		}
		totalPromptLength += len([]rune(value.PromptEN))
		seen[value.ID] = true
		result[index] = value
	}
	if totalPromptLength > 30000 {
		return nil, fmt.Errorf("locked character descriptions are too long")
	}
	return result, nil
}

// The model chooses scene membership, but it cannot rewrite user-approved identity text.
func reconcileLockedSequenceCharacters(plan *imageSequencePromptPlan, locked []imageSequenceCharacter) {
	if len(locked) == 0 {
		return
	}
	lockedByID := make(map[string]imageSequenceCharacter, len(locked))
	for _, character := range locked {
		lockedByID[character.ID] = character
	}
	seen := make(map[string]bool, len(locked))
	for index := range plan.Characters {
		id := strings.ToLower(strings.TrimSpace(plan.Characters[index].ID))
		if character, exists := lockedByID[id]; exists {
			plan.Characters[index] = character
			seen[id] = true
		}
	}
	for _, character := range locked {
		if !seen[character.ID] {
			plan.Characters = append(plan.Characters, character)
		}
	}
	// Scene membership belongs to the planner. A locked profile is copied
	// verbatim only into scenes whose character_ids explicitly include it; forcing
	// every registered character into every scene causes unwanted extra people.
}

func validateSequencePlannerInput(values []string) ([]string, error) {
	if len(values) < 2 || len(values) > 12 {
		return nil, fmt.Errorf("storyboard planning requires 2 to 12 scenes")
	}
	result := make([]string, len(values))
	for index, value := range values {
		result[index] = strings.TrimSpace(value)
		if result[index] == "" {
			return nil, fmt.Errorf("every sequence scene requires a prompt")
		}
		if len([]rune(result[index])) > 4000 {
			return nil, fmt.Errorf("sequence scene prompt is too long")
		}
	}
	return result, nil
}

func validateSequencePlannerRequest(request imageSequencePlanRequest) ([]string, string, string, int, error) {
	outline := strings.TrimSpace(request.Outline)
	sharedPrompt := strings.TrimSpace(request.SharedPrompt)
	if len([]rune(sharedPrompt)) > 12000 {
		return nil, "", "", 0, fmt.Errorf("shared continuity prompt is too long")
	}
	if outline != "" {
		if len([]rune(outline)) > 12000 {
			return nil, "", "", 0, fmt.Errorf("story outline is too long")
		}
		if request.SceneCount < 2 || request.SceneCount > 12 {
			return nil, "", "", 0, fmt.Errorf("storyboard planning requires 2 to 12 scenes")
		}
		return nil, outline, sharedPrompt, request.SceneCount, nil
	}
	prompts, err := validateSequencePlannerInput(request.Prompts)
	if err != nil {
		return nil, "", "", 0, err
	}
	return prompts, "", sharedPrompt, len(prompts), nil
}

func normalizeSequenceStrategies(values []string, count int, allowAuto bool) ([]string, error) {
	if len(values) == 0 {
		values = make([]string, count)
	}
	if len(values) != count {
		return nil, fmt.Errorf("invalid sequence strategies")
	}
	valid := map[string]bool{"minor": true, "major": true, "partial": true}
	if allowAuto {
		valid["auto"] = true
	}
	result := make([]string, count)
	for index, value := range values {
		value = strings.ToLower(strings.TrimSpace(value))
		if value == "" {
			value = "auto"
		}
		if !valid[value] {
			return nil, fmt.Errorf("unsupported sequence strategy")
		}
		result[index] = value
	}
	return result, nil
}

func openAIMessageContent(data []byte) (string, error) {
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
		return "", fmt.Errorf("invalid OpenAI response")
	}
	content := strings.TrimSpace(response.Choices[0].Message.Content)
	if content == "" {
		return "", fmt.Errorf("empty OpenAI response")
	}
	return content, nil
}

func extractJSONObject(value string) string {
	value = strings.TrimSpace(value)
	start, end := strings.Index(value, "{"), strings.LastIndex(value, "}")
	if start >= 0 && end >= start {
		return value[start : end+1]
	}
	return value
}

func validateSequencePromptPlan(plan *imageSequencePromptPlan, prompts []string, sharedPrompt string, sceneCount int) error {
	if len(plan.Scenes) != sceneCount {
		return fmt.Errorf("returned the wrong number of scenes")
	}
	plan.SharedPrompt = strings.TrimSpace(plan.SharedPrompt)
	if sharedPrompt != "" {
		plan.SharedPrompt = sharedPrompt
	}
	if plan.SharedPrompt == "" {
		return fmt.Errorf("returned an empty continuity bible")
	}
	plan.WorldPromptEN = strings.TrimSpace(plan.WorldPromptEN)
	characters := make(map[string]string, len(plan.Characters))
	canonicalParts := make([]string, 0, len(plan.Characters)+1)
	if plan.WorldPromptEN != "" {
		canonicalParts = append(canonicalParts, plan.WorldPromptEN)
	}
	for index := range plan.Characters {
		character := &plan.Characters[index]
		character.ID = strings.ToLower(strings.TrimSpace(character.ID))
		character.NameKO = strings.TrimSpace(character.NameKO)
		character.NameEN = strings.TrimSpace(character.NameEN)
		character.DescriptionKO = strings.TrimSpace(character.DescriptionKO)
		character.PromptEN = strings.TrimSpace(character.PromptEN)
		if character.ID == "" || character.NameKO == "" || character.NameEN == "" || character.DescriptionKO == "" || character.PromptEN == "" {
			return fmt.Errorf("returned an incomplete character identity")
		}
		if _, exists := characters[character.ID]; exists {
			return fmt.Errorf("returned duplicate character ids")
		}
		characters[character.ID] = character.PromptEN
		canonicalParts = append(canonicalParts, character.PromptEN)
	}
	if sharedPrompt == "" && len(plan.Characters) > 0 {
		lines := make([]string, 0, len(plan.Characters)+1)
		lines = append(lines, plan.SharedPrompt, "등장 캐릭터:")
		for _, character := range plan.Characters {
			lines = append(lines, "- "+character.NameKO+": "+character.DescriptionKO)
		}
		plan.SharedPrompt = strings.Join(lines, "\n")
	}
	plan.CanonicalPrompt = strings.Join(canonicalParts, "\n\n")
	if plan.CanonicalPrompt == "" {
		return fmt.Errorf("returned an empty canonical prompt")
	}
	for index := range plan.Scenes {
		scene := &plan.Scenes[index]
		if len(prompts) > 0 {
			scene.OriginalPrompt = prompts[index]
		} else {
			scene.OriginalPrompt = strings.TrimSpace(scene.OriginalPrompt)
		}
		scene.SceneTitle = strings.TrimSpace(scene.SceneTitle)
		scene.ScenePromptEN = strings.TrimSpace(scene.ScenePromptEN)
		scene.ChangeSummary = strings.TrimSpace(scene.ChangeSummary)
		scene.Strategy = "storyboard"
		if scene.OriginalPrompt == "" || scene.ScenePromptEN == "" {
			return fmt.Errorf("returned an empty enhanced prompt")
		}
		parts := make([]string, 0, len(scene.CharacterIDs)+2)
		if plan.WorldPromptEN != "" {
			parts = append(parts, plan.WorldPromptEN)
		}
		seen := make(map[string]bool, len(scene.CharacterIDs))
		for characterIndex, id := range scene.CharacterIDs {
			id = strings.ToLower(strings.TrimSpace(id))
			scene.CharacterIDs[characterIndex] = id
			prompt, exists := characters[id]
			if !exists {
				return fmt.Errorf("scene references an unknown character")
			}
			if !seen[id] {
				parts = append(parts, prompt)
				seen[id] = true
			}
		}
		parts = append(parts, scene.ScenePromptEN)
		scene.EnhancedPrompt = strings.Join(parts, "\n\n")
		if len([]rune(scene.EnhancedPrompt)) > 8000 {
			return fmt.Errorf("compiled scene prompt is too long")
		}
		if scene.SceneTitle == "" {
			scene.SceneTitle = fmt.Sprintf("장면 %d", index+1)
		}
	}
	return nil
}
