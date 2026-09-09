package config

import (
	"encoding/json"
	"gopkg.in/yaml.v3"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func composerFixture(t *testing.T) (*PromptComposer, string) {
	t.Helper()
	data, err := os.ReadFile("testdata/prompt-composer.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Composer PromptComposer `json:"composer"`
		Expected string         `json:"expected"`
	}
	if err = json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	return &f.Composer, f.Expected
}
func TestPromptComposerSharedRenderingAndSaveReload(t *testing.T) {
	c, want := composerFixture(t)
	got, err := c.Render()
	if err != nil || got != want {
		t.Fatalf("%q %v", got, err)
	}
	cfg, _, err := Load(filepath.Join(t.TempDir(), "defaults.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg.Model.PromptComposer = c
	cfg.Model.SystemPrompt = "stale text"
	cfg.Normalize()
	if cfg.Model.SystemPrompt != want {
		t.Fatal("resolved prompt did not replace stale text")
	}
	data, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var restored Config
	if err = yaml.Unmarshal(data, &restored); err != nil {
		t.Fatal(err)
	}
	restored.Normalize()
	if restored.Model.SystemPrompt != want || restored.Model.PromptComposer.PersonaID != "p" {
		t.Fatal("composition lost on YAML round trip")
	}
	if err = restored.Validate(); err != nil {
		t.Fatal(err)
	}
}
func TestPromptComposerPreservesExistingDirectContent(t *testing.T) {
	m := ModelConfig{SystemPrompt: "modified custom text", SystemPromptPreset: "mine", SystemPromptPresets: []PromptPreset{{Name: "mine", Prompt: "original preset"}}}
	m.normalizePromptComposer()
	if m.PromptComposer.Enabled || m.SystemPrompt != "modified custom text" || m.PromptComposer.Extra != m.SystemPrompt || m.SystemPromptPresets[0].Prompt != "original preset" {
		t.Fatal("direct content changed")
	}
	m.PromptComposer.Enabled = true
	m.normalizePromptComposer()
	if !strings.Contains(m.SystemPrompt, "modified custom text") {
		t.Fatal("custom text missing after enabling")
	}
	m.PromptComposer.Blocks = []PromptBlock{}
	m.PromptComposer.Enabled = false
	m.normalizePromptComposer()
	if len(m.PromptComposer.Blocks) != 0 {
		t.Fatal("deleted blocks reappeared")
	}
}
func TestPromptComposerRejectsConflictsAndStaleCombinations(t *testing.T) {
	c, _ := composerFixture(t)
	c.Blocks = append(c.Blocks, PromptBlock{ID: "long", Kind: "condition", Category: "length", Group: "length", Name: "자세히", Prompt: "자세하게 답한다."})
	c.ConditionIDs = append(c.ConditionIDs, "long")
	if c.Validate() == nil {
		t.Fatal("conflicting choices accepted")
	}
	c.ConditionIDs = []string{"first", "first"}
	if c.Validate() == nil {
		t.Fatal("duplicate choices accepted")
	}
	c.ConditionIDs = []string{"unknown"}
	if c.Validate() == nil {
		t.Fatal("unknown choice accepted")
	}
	c.ConditionIDs = nil
	c.Combinations = []PromptCombination{{ID: "saved", Name: "저장", PromptSelection: PromptSelection{PersonaID: "missing"}}}
	if c.Validate() == nil {
		t.Fatal("invalid saved combination accepted")
	}
}
func TestPromptComposerNormalizesFileBeforeUse(t *testing.T) {
	c, want := composerFixture(t)
	cfg, _, err := Load(filepath.Join(t.TempDir(), "defaults.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	cfg.Model.PromptComposer = c
	data, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Model.SystemPrompt != want {
		t.Fatal("file prompt was not resolved")
	}
}
