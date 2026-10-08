package config

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
)

type PromptBlock struct {
	ID       string `json:"id" yaml:"id"`
	Kind     string `json:"kind" yaml:"kind"`
	Category string `json:"category" yaml:"category"`
	Group    string `json:"group" yaml:"group"`
	Name     string `json:"name" yaml:"name"`
	Prompt   string `json:"prompt" yaml:"prompt"`
}
type PromptSelection struct {
	PersonaID    string   `json:"persona_id" yaml:"persona_id"`
	ConditionIDs []string `json:"condition_ids" yaml:"condition_ids"`
	Extra        string   `json:"extra" yaml:"extra"`
}
type PromptCombination struct {
	ID              string `json:"id" yaml:"id"`
	Name            string `json:"name" yaml:"name"`
	PromptSelection `yaml:",inline"`
}
type PromptComposer struct {
	CharacterName        string              `json:"character_name" yaml:"character_name"`
	CharacterDescription string              `json:"character_description" yaml:"character_description"`
	CombinationID        string              `json:"combination_id" yaml:"combination_id"`
	Enabled              bool                `json:"enabled" yaml:"enabled"`
	Blocks               []PromptBlock       `json:"blocks" yaml:"blocks"`
	Combinations         []PromptCombination `json:"combinations" yaml:"combinations"`
	PromptSelection      `yaml:",inline"`
}
type promptCategory struct {
	ID    string `json:"id"`
	Label string `json:"label"`
}

var composerDefaults = func() struct {
	Categories []promptCategory `json:"categories"`
	Blocks     []PromptBlock    `json:"blocks"`
} {
	var result struct {
		Categories []promptCategory `json:"categories"`
		Blocks     []PromptBlock    `json:"blocks"`
	}
	data, err := assets.ReadFile("assets/prompt_composer.defaults.json")
	if err != nil {
		panic(err)
	}
	if err = json.Unmarshal(data, &result); err != nil {
		panic(err)
	}
	return result
}()

func (m *ModelConfig) normalizePromptComposer() {
	if m.PromptComposer == nil {
		c := &PromptComposer{Blocks: append([]PromptBlock(nil), composerDefaults.Blocks...), Combinations: []PromptCombination{}, PromptSelection: PromptSelection{ConditionIDs: []string{}, Extra: m.SystemPrompt}}
		for _, preset := range m.SystemPromptPresets {
			if strings.TrimSpace(preset.Prompt) == "" {
				continue
			}
			sum := sha256.Sum256([]byte(preset.Name))
			id := fmt.Sprintf("preset-%x", sum[:8])
			block := PromptBlock{ID: id, Kind: "persona", Name: preset.Name, Prompt: preset.Prompt}
			if preset.Name == "한글 전용" {
				block.Kind = "condition"
				block.Category = "language"
				block.ID = "hangul"
				id = block.ID
			}
			replaced := false
			for i := range c.Blocks {
				if c.Blocks[i].ID == block.ID {
					c.Blocks[i] = block
					replaced = true
					break
				}
			}
			if !replaced {
				c.Blocks = append(c.Blocks, block)
			}
			if preset.Name == m.SystemPromptPreset && preset.Prompt == m.SystemPrompt {
				c.Extra = ""
				if block.Kind == "persona" {
					c.PersonaID = id
				} else {
					c.ConditionIDs = []string{id}
				}
			}
		}
		m.PromptComposer = c
	}
	c := m.PromptComposer
	if c.Blocks == nil {
		c.Blocks = []PromptBlock{}
	}
	if c.Combinations == nil {
		c.Combinations = []PromptCombination{}
	}
	if c.ConditionIDs == nil {
		c.ConditionIDs = []string{}
	}
	if c.Enabled {
		if prompt, err := c.Render(); err == nil {
			m.SystemPrompt = prompt
			m.SystemPromptPreset = ""
		}
	}
}

func (c *PromptComposer) validateSelection(s PromptSelection) error {
	blocks := map[string]PromptBlock{}
	for _, b := range c.Blocks {
		blocks[b.ID] = b
	}
	if s.PersonaID != "" {
		if b, ok := blocks[s.PersonaID]; !ok || b.Kind != "persona" {
			return fmt.Errorf("persona not found: %s", s.PersonaID)
		}
	}
	seen, groups := map[string]bool{}, map[string]string{}
	for _, id := range s.ConditionIDs {
		b, ok := blocks[id]
		if !ok || b.Kind != "condition" {
			return fmt.Errorf("condition not found: %s", id)
		}
		if seen[id] {
			return fmt.Errorf("duplicate selected condition: %s", id)
		}
		seen[id] = true
		if b.Group != "" {
			if other := groups[b.Group]; other != "" {
				return fmt.Errorf("conflicting conditions: %s / %s", other, b.Name)
			}
			groups[b.Group] = b.Name
		}
	}
	if len(s.Extra) > 65536 {
		return fmt.Errorf("additional prompt is too long")
	}
	return nil
}
func (c *PromptComposer) Validate() error {
	if len(c.CharacterName) > 240 || len(c.CharacterDescription) > 4096 {
		return fmt.Errorf("character name or description is too long")
	}
	if len(c.Blocks) > 300 || len(c.Combinations) > 200 {
		return fmt.Errorf("too many prompt blocks or combinations")
	}
	categories := map[string]bool{}
	for _, v := range composerDefaults.Categories {
		categories[v.ID] = true
	}
	seen := map[string]bool{}
	for _, b := range c.Blocks {
		if strings.TrimSpace(b.ID) == "" || seen[b.ID] {
			return fmt.Errorf("invalid or duplicate prompt block ID: %s", b.ID)
		}
		seen[b.ID] = true
		if strings.TrimSpace(b.Name) == "" || strings.TrimSpace(b.Prompt) == "" || len(b.Prompt) > 65536 {
			return fmt.Errorf("invalid prompt block: %s", b.ID)
		}
		if b.Kind != "persona" && b.Kind != "condition" {
			return fmt.Errorf("invalid prompt block kind: %s", b.Kind)
		}
		if b.Kind == "condition" && !categories[b.Category] {
			return fmt.Errorf("invalid prompt category: %s", b.Category)
		}
	}
	if err := c.validateSelection(c.PromptSelection); err != nil {
		return err
	}
	seen = map[string]bool{}
	for _, s := range c.Combinations {
		if s.ID == "" || seen[s.ID] || strings.TrimSpace(s.Name) == "" {
			return fmt.Errorf("invalid prompt combination: %s", s.ID)
		}
		seen[s.ID] = true
		if err := c.validateSelection(s.PromptSelection); err != nil {
			return err
		}
	}
	if c.CombinationID != "" && !seen[c.CombinationID] {
		return fmt.Errorf("prompt combination not found: %s", c.CombinationID)
	}
	return nil
}
func (c *PromptComposer) Render() (string, error) {
	if err := c.Validate(); err != nil {
		return "", err
	}
	parts := []string{}
	identity := []string{}
	if name := strings.TrimSpace(c.CharacterName); name != "" {
		identity = append(identity, "대화에서 사용하는 너의 이름은 "+name+"이다.")
	}
	if description := strings.TrimSpace(c.CharacterDescription); description != "" {
		identity = append(identity, description)
	}
	if len(identity) > 0 {
		parts = append(parts, "[AI 캐릭터]\n"+strings.Join(identity, "\n"))
	}
	selected := map[string]bool{}
	for _, id := range c.ConditionIDs {
		selected[id] = true
	}
	if c.PersonaID != "" {
		for _, b := range c.Blocks {
			if b.ID == c.PersonaID {
				parts = append(parts, "[페르소나]\n"+strings.TrimSpace(b.Prompt))
			}
		}
		if len(selected) > 0 {
			parts = append(parts, "말투·형식이 충돌하면 페르소나의 기본 표현보다 아래 추가 조건을 따른다.")
		}
	}
	for _, category := range composerDefaults.Categories {
		lines := []string{}
		for _, b := range c.Blocks {
			if b.Kind == "condition" && b.Category == category.ID && selected[b.ID] {
				lines = append(lines, strings.TrimSpace(b.Prompt))
			}
		}
		if len(lines) > 0 {
			parts = append(parts, "["+category.Label+"]\n"+strings.Join(lines, "\n"))
		}
	}
	if extra := strings.TrimSpace(c.Extra); extra != "" {
		parts = append(parts, "[직접 추가]\n"+extra)
	}
	return strings.Join(parts, "\n\n"), nil
}
