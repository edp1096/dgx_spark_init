package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestRetiredFlashNextEXL3RemovedFromSavedCatalog(t *testing.T) {
	catalog, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	catalog.Components = append(catalog.Components, orchestrator.Component{ID: "old-fn-copy", ComposeAsset: "compose.flash-next-exl3.yaml"})
	catalog.Bundles = append(catalog.Bundles, orchestrator.Bundle{ID: "old-fn-set", Components: []string{"old-fn-copy", "extra-ssh"}})
	cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog, Bundle: "old-fn-set", ActiveBundle: "old-fn-set", AutoStart: true}}
	cfg.Normalize()
	if cfg.Runtime.Bundle != "flash-next" || cfg.Runtime.ActiveBundle != "flash-next" || cfg.Runtime.AutoStart {
		t.Fatal("retired selection must fall back without automatically starting another model")
	}
	for _, b := range cfg.Runtime.Catalog.Bundles {
		if b.ID == "old-fn-set" || b.ID == "flash-next-exl3" {
			t.Fatal("retired set survived")
		}
	}
	for _, c := range cfg.Runtime.Catalog.Components {
		if c.ID == "old-fn-copy" {
			t.Fatal("retired service survived")
		}
	}
	if _, ok := cfg.Runtime.Catalog.Component("qwen27-exl3"); !ok {
		t.Fatal("Qwen 27B EXL3 was removed")
	}
	if _, ok := cfg.Runtime.Catalog.Component("extra-ssh"); !ok {
		t.Fatal("shared extra was removed")
	}
}

func TestRetiredQwen27NVFP4SelectsEXL3WithoutAutostart(t *testing.T) {
	catalog, _ := orchestrator.LoadCatalog()
	catalog.Components = append(catalog.Components, orchestrator.Component{ID: "qwen27", ComposeAsset: "compose.qwen27.yaml"})
	catalog.Bundles = append(catalog.Bundles, orchestrator.Bundle{ID: "qwen27", Components: []string{"qwen27", "extra-ssh"}})
	cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog, Bundle: "qwen27", ActiveBundle: "qwen27", AutoStart: true}}
	cfg.Normalize()
	if cfg.Runtime.Bundle != "qwen27-exl3" || cfg.Runtime.ActiveBundle != "qwen27-exl3" || cfg.Runtime.AutoStart {
		t.Fatal("retired NVFP4 must select EXL3 without starting it")
	}
	if _, ok := cfg.Runtime.Catalog.Component("qwen27"); ok {
		t.Fatal("NVFP4 survived")
	}
	if _, ok := cfg.Runtime.Catalog.Component("qwen27-exl3"); !ok {
		t.Fatal("EXL3 missing")
	}
}
