package config

import (
	"encoding/json"
	"path/filepath"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestRetiredDeepSeekDeploymentCannotReturnAfterReload(t *testing.T) {
	for _, legacy := range []struct{ id, asset, controller, modelType string }{
		{"ds4fve", "", "dspark-cluster", "deepseek-v4"},
		{"ds4fve_exl3", "compose.ds4fve_exl3.yaml", "compose", "deepseek-v4-exl3"},
		{"copied-cluster", "", "dspark-cluster", "deepseek-v4"},
		{"copied-native", "compose.ds4fve_exl3.yaml", "compose", "deepseek-v4-exl3"},
	} {
		t.Run(legacy.id, func(t *testing.T) {
			catalog, err := orchestrator.LoadCatalog()
			if err != nil {
				t.Fatal(err)
			}
			before, _ := catalog.Component("ds41")
			old, _ := json.Marshal(before)
			catalog.Components = append(catalog.Components, orchestrator.Component{ID: legacy.id, Controller: legacy.controller, ComposeAsset: legacy.asset})
			catalog.Bundles = append(catalog.Bundles, orchestrator.Bundle{ID: legacy.id, Name: "Retired deployment", ModelType: legacy.modelType, Components: []string{legacy.id, "extra-ssh"}})
			cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Catalog: &catalog, Bundle: "ds41", ActiveBundle: legacy.id, AutoStart: true, BuiltinRevision: 46}}
			cfg.Normalize()
			for _, id := range []string{"ds4fve", "ds4fve_exl3", legacy.id} {
				if _, ok := cfg.Runtime.Catalog.Component(id); ok {
					t.Fatalf("retired component %s returned", id)
				}
				if _, ok := cfg.Runtime.Catalog.Bundle(id); ok {
					t.Fatalf("retired bundle %s returned", id)
				}
			}
			if cfg.Runtime.ActiveBundle != "ds41" || cfg.Runtime.Bundle != "ds41" || cfg.Runtime.AutoStart {
				t.Fatal("must select surviving DeepSeek TP2 without starting it")
			}
			after, _ := cfg.Runtime.Catalog.Component("ds41")
			now, _ := json.Marshal(after)
			if string(old) != string(now) {
				t.Fatal("surviving deployment changed")
			}
			if _, ok := cfg.Runtime.Catalog.Component("extra-ssh"); !ok {
				t.Fatal("shared service removed")
			}
			path := filepath.Join(t.TempDir(), "talk.yaml")
			if err := Save(path, cfg); err != nil {
				t.Fatal(err)
			}
			loaded, _, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if loaded.Runtime.ActiveBundle != "ds41" || loaded.Runtime.AutoStart || loaded.Model.DefaultModel != "DeepSeek-V4.1-Flash-TF" {
				t.Fatal("retirement did not persist", loaded.Runtime.ActiveBundle, loaded.Model.DefaultModel)
			}
		})
	}
}
