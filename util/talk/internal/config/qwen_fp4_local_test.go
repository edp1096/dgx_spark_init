package config

import (
	"path/filepath"
	"testing"

	"sparktalk/internal/orchestrator"
)

func TestFP4LocalMigrationPreservesSelectionAndCustomDeployment(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sparktalk.yaml")
	cfg, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	const id = "flash-next-fp4-local"
	var components []orchestrator.Component
	for _, x := range cfg.Runtime.Catalog.Components {
		if x.ID != id {
			components = append(components, x)
		}
	}
	var bundles []orchestrator.Bundle
	for _, b := range cfg.Runtime.Catalog.Bundles {
		if b.ID != id {
			bundles = append(bundles, b)
		}
	}
	cfg.Runtime.Catalog.Components, cfg.Runtime.Catalog.Bundles = components, bundles
	cfg.Runtime.Bundle, cfg.Runtime.ActiveBundle = "ds41", "qwen38fn_exl3"
	cfg.Runtime.BuiltinRevision = 42
	cfg.Normalize()
	if cfg.Runtime.Bundle != "ds41" || cfg.Runtime.ActiveBundle != "qwen38fn_exl3" || cfg.Runtime.BuiltinRevision != 52 {
		t.Fatal("migration changed selected/default set")
	}
	if _, ok := cfg.Runtime.Catalog.Bundle(id); !ok {
		t.Fatal("new set missing")
	}
	for i := range cfg.Runtime.Catalog.Components {
		x := &cfg.Runtime.Catalog.Components[i]
		if x.ID == id {
			x.Name, x.Endpoint, x.Port = "My FP4 model", "http://127.0.0.1:19004", 19004
		}
	}
	cfg.Runtime.BuiltinRevision = 42
	cfg.Normalize()
	x, _ := cfg.Runtime.Catalog.Component(id)
	if x.Name != "My FP4 model" || x.Endpoint != "http://127.0.0.1:19004" || x.Port != 19004 {
		t.Fatal("custom deployment overwritten")
	}
	if err := Save(path, cfg); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	loaded.SelectManagedBundle(id)
	if loaded.Model.DefaultModel != orchestrator.QwenFP4Local || loaded.Model.ModelType != "qwen3.8-tensorfold" || loaded.Context.WindowTokens != 1048576 || loaded.Model.ReasoningEffort != "medium" || loaded.Runtime.Bundle != "ds41" {
		t.Fatalf("selected profile is incorrect: %+v", loaded.Model)
	}
}
