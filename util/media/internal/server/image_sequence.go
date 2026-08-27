package server

import (
	"encoding/json"
	"fmt"
	"strings"
)

func parseImageSequencePrompts(raw string) ([]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	var prompts []string
	if err := json.Unmarshal([]byte(raw), &prompts); err != nil {
		return nil, fmt.Errorf("invalid sequence prompts")
	}
	if len(prompts) < 2 || len(prompts) > 6 {
		return nil, fmt.Errorf("sequence generation requires 2 to 6 scenes")
	}
	for index := range prompts {
		prompts[index] = strings.TrimSpace(prompts[index])
		if prompts[index] == "" {
			return nil, fmt.Errorf("every sequence scene requires a prompt")
		}
		if len([]rune(prompts[index])) > 4000 {
			return nil, fmt.Errorf("sequence scene prompt is too long")
		}
	}
	return prompts, nil
}

func parseImageSequenceRegions(raw string, promptCount int) ([]string, error) {
	if promptCount == 0 {
		return nil, nil
	}
	regions := make([]string, promptCount)
	for index := range regions {
		regions[index] = "all"
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return regions, nil
	}
	var provided []string
	if err := json.Unmarshal([]byte(raw), &provided); err != nil || len(provided) != promptCount {
		return nil, fmt.Errorf("invalid sequence regions")
	}
	valid := map[string]bool{"all": true, "left": true, "right": true, "upper": true, "lower": true, "left-arm": true, "right-arm": true, "custom": true}
	for index, region := range provided {
		region = strings.ToLower(strings.TrimSpace(region))
		if !valid[region] {
			return nil, fmt.Errorf("unsupported sequence region")
		}
		regions[index] = region
	}
	regions[0] = "all"
	return regions, nil
}

func sequenceEditPrompt(change string) string {
	return "Change: " + strings.TrimSpace(change) +
		"\nPose replacement rule: Treat every requested pose or body-part movement as a replacement, never an addition. Redraw each moved body part only in its new position and remove it from its previous position. Keep the anatomically correct number of limbs, with no duplicate arms, hands, legs, heads, or ghost body parts." +
		"\nFace continuity rule: Preserve the exact head and facial construction, including faceplate or screen type, eye count, eye shape, eye color, eye spacing, mouth design, and distinguishing details. If an expression change is explicitly requested, alter only that expression and never redesign the head or face." +
		"\nPreserve: the same character identity, face, hair, clothing details unless explicitly changed, visual style, lighting continuity, and scene elements that do not conflict with the requested movement. Do not preserve the previous pose where it conflicts with the new pose."
}
