package config

import (
	"encoding/json"
	"sparktalk/internal/modelidentity"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestQwen38FNEXL3IdentityMigrationPreservesSelectionAndCustomSettings(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		c := &cat.Components[i]
		if c.ID == modelidentity.Qwen38FNEXL3 {
			c.ID = modelidentity.LegacyQwenRuntime
			c.Model = modelidentity.LegacyQwenModel
			c.Container = modelidentity.LegacyQwenContainer
			c.ComposeAsset = modelidentity.LegacyQwenCompose
			c.Name = "내 EXL3"
			c.Endpoint = "http://127.0.0.1:18002"
		}
	}
	for i := range cat.Bundles {
		b := &cat.Bundles[i]
		if b.ID == modelidentity.Qwen38FNEXL3 {
			b.ID = modelidentity.LegacyQwenRuntime
			b.ModelID = modelidentity.LegacyQwenModel
			b.ModelType = modelidentity.LegacyQwenType
			for j, id := range b.Components {
				if id == modelidentity.Qwen38FNEXL3 {
					b.Components[j] = modelidentity.LegacyQwenRuntime
				}
			}
			model, container := modelidentity.LegacyQwenModel, modelidentity.LegacyQwenContainer
			b.Bindings[modelidentity.LegacyQwenRuntime] = orchestrator.Deployment{Model: &model, Container: &container}
		}
	}
	imported, err := orchestrator.ValidateCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := imported.Bundle(modelidentity.Qwen38FNEXL3); !ok {
		t.Fatal("legacy catalog import did not migrate")
	}
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", BuiltinRevision: 28, Bundle: "flash-next", ActiveBundle: modelidentity.LegacyQwenRuntime, Catalog: &cat}, Model: ModelConfig{DefaultModel: modelidentity.LegacyQwenModel, ModelType: modelidentity.LegacyQwenType, ReasoningEffort: "medium"}}
	cfg.Normalize()
	if cfg.Runtime.ActiveBundle != modelidentity.Qwen38FNEXL3 || cfg.Runtime.Bundle != "flash-next" || cfg.Runtime.AutoStart || cfg.Model.DefaultModel != modelidentity.Qwen38FNEXL3 || cfg.Model.ModelType != modelidentity.Qwen38FNEXL3 || cfg.Model.ReasoningEffort != "medium" {
		t.Fatal("selection or preferences changed")
	}
	c, _ := cfg.Runtime.Catalog.ResolveComponent(modelidentity.Qwen38FNEXL3, modelidentity.Qwen38FNEXL3)
	if c.Name != "내 EXL3" || c.Container != "sparktalk-qwen38fn_exl3" || c.ComposeAsset != "compose.qwen38fn_exl3.yaml" {
		t.Fatal(c)
	}
	for _, id := range []string{"qwim-mmh3", "nemotron-asr", "qwen3-tts"} {
		x, _ := cfg.Runtime.Catalog.ResolveComponent(modelidentity.Qwen38FNEXL3, id)
		if !x.KeepResident {
			t.Fatal("residency lost")
		}
	}
	before, _ := json.Marshal(cfg)
	cfg.Normalize()
	after, _ := json.Marshal(cfg)
	if string(before) != string(after) {
		t.Fatal("migration is not idempotent")
	}
}
