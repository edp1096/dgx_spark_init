package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"gopkg.in/yaml.v3"
	"sparktalk/internal/orchestrator"
)

func TestRetiredRuntimesMigrateToFlashNextWithoutAutostart(t *testing.T) {
	for _, id := range []string{"flash-next-exl3", "qwen27", "qwen27-gguf", "qwen27-exl3"} {
		for _, renamed := range []bool{false, true} {
			name := id
			if renamed {
				name += "-copy"
			}
			t.Run(name, func(t *testing.T) {
				catalog, err := orchestrator.LoadCatalog()
				if err != nil {
					t.Fatal(err)
				}
				component := orchestrator.Component{ID: name, ComposeAsset: "compose." + id + ".yaml"}
				catalog.Components = append(catalog.Components, component)
				catalog.Bundles = append(catalog.Bundles, orchestrator.Bundle{ID: name, Components: []string{name, "extra-ssh"}})
				cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Catalog: &catalog, Bundle: name, ActiveBundle: name, AutoStart: true}, Model: ModelConfig{ReasoningEffort: "medium"}}
				cfg.Normalize()
				if cfg.Runtime.Bundle != "flash-next" || cfg.Runtime.ActiveBundle != "flash-next" || cfg.Runtime.AutoStart {
					t.Fatal("retired selection must fall back without automatically starting another model")
				}
				if _, ok := cfg.Runtime.Catalog.Component(name); ok {
					t.Fatal("retired component survived")
				}
				if _, ok := cfg.Runtime.Catalog.Bundle(name); ok {
					t.Fatal("retired set survived")
				}
				if _, ok := cfg.Runtime.Catalog.Component("extra-ssh"); !ok {
					t.Fatal("shared service lost")
				}
				if cfg.Model.DefaultModel != "qwen3.8-flash-next" || cfg.Model.ModelType != "qwen3.8" || cfg.Context.WindowTokens != 65536 || cfg.Model.ReasoningEffort != "medium" {
					t.Fatal("fallback profile or preference incorrect")
				}
				before, _ := json.Marshal(cfg)
				cfg.Normalize()
				after, _ := json.Marshal(cfg)
				if string(before) != string(after) {
					t.Fatal("migration not idempotent")
				}
			})
		}
	}
}

func TestEXL3RetirementPreservesSurvivingSelections(t *testing.T) {
	for _, choice := range []struct {
		selected, active, wantSelected, wantActive string
		autoStart                                  bool
	}{
		{"qwen27-exl3", "flash-next", "flash-next", "flash-next", false},
		{"gemma", "qwen27-exl3", "gemma", "gemma", false},
		{"gemma", "flash-next", "gemma", "flash-next", true},
	} {
		catalog, _ := orchestrator.LoadCatalog()
		catalog.Components = append(catalog.Components, orchestrator.Component{ID: "qwen27-exl3", ComposeAsset: "compose.qwen27-exl3.yaml"})
		catalog.Bundles = append(catalog.Bundles, orchestrator.Bundle{ID: "qwen27-exl3", Components: []string{"qwen27-exl3"}})
		cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog, Bundle: choice.selected, ActiveBundle: choice.active, AutoStart: true}}
		cfg.Normalize()
		if cfg.Runtime.Bundle != choice.wantSelected || cfg.Runtime.ActiveBundle != choice.wantActive || cfg.Runtime.AutoStart != choice.autoStart {
			t.Fatalf("selection changed: %+v", choice)
		}
	}
}

func TestRetiredOnlySavedCatalogLoadsWithValidReplacement(t *testing.T) {
	catalog := orchestrator.Catalog{
		Components: []orchestrator.Component{{ID: "qwen27-exl3", ComposeAsset: "compose.qwen27-exl3.yaml"}},
		Bundles:    []orchestrator.Bundle{{ID: "qwen27-exl3", Components: []string{"qwen27-exl3"}}},
	}
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Catalog: &catalog, Bundle: "qwen27-exl3", ActiveBundle: "qwen27-exl3", AutoStart: true}}
	raw, err := yaml.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "sparktalk.yaml")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Runtime.AutoStart || loaded.Runtime.Bundle != "flash-next" {
		t.Fatal("unsafe replacement")
	}
	if _, err := orchestrator.ValidateCatalog(*loaded.Runtime.Catalog); err != nil {
		t.Fatal(err)
	}
	if err := Save(path, loaded); err != nil {
		t.Fatal(err)
	}
	persisted, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := persisted.Runtime.Catalog.Component("qwen27-exl3"); ok {
		t.Fatal("retired model returned after saving")
	}
}
