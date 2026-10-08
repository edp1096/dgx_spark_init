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
	if len(prompts) < 2 || len(prompts) > 12 {
		return nil, fmt.Errorf("storyboard generation requires 2 to 12 scenes")
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

func parseImageSequenceEnhancedPrompts(raw string, prompts []string) ([]string, error) {
	if len(prompts) == 0 {
		return nil, nil
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return append([]string(nil), prompts...), nil
	}
	var enhanced []string
	if err := json.Unmarshal([]byte(raw), &enhanced); err != nil || len(enhanced) != len(prompts) {
		return nil, fmt.Errorf("invalid sequence enhanced prompts")
	}
	for index := range enhanced {
		enhanced[index] = strings.TrimSpace(enhanced[index])
		if enhanced[index] == "" {
			return nil, fmt.Errorf("every sequence scene requires an enhanced prompt")
		}
		if len([]rune(enhanced[index])) > 8000 {
			return nil, fmt.Errorf("sequence enhanced prompt is too long")
		}
	}
	return enhanced, nil
}

func parseImageSequenceStrategies(raw string, promptCount int) ([]string, error) {
	if promptCount == 0 {
		return nil, nil
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		result := make([]string, promptCount)
		for index := range result {
			if index == 0 {
				result[index] = "major"
			} else {
				result[index] = "minor"
			}
		}
		return result, nil
	}
	var strategies []string
	if err := json.Unmarshal([]byte(raw), &strategies); err != nil {
		return nil, fmt.Errorf("invalid sequence strategies")
	}
	result, err := normalizeSequenceStrategies(strategies, promptCount, false)
	if err != nil {
		return nil, err
	}
	result[0] = "major"
	return result, nil
}

func sequenceEditPrompt(change string) string {
	return "Change: " + strings.TrimSpace(change) +
		"\nPreserve: the same subject identity and stable scene details. Render the requested final frame with an anatomically correct subject. A requested movement replaces the previous pose; do not duplicate limbs or retain ghost body parts."
}
