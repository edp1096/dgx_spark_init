package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestQwenImageMigrationPreservesBindingsAndCustomProfiles(t *testing.T) {
	for _, custom := range []bool{false, true} {
		cat, _ := orchestrator.LoadCatalog()
		for i := range cat.Components {
			x := &cat.Components[i]
			if x.ID == "flux2" {
				x.ComposeAsset = "compose.flux2.yaml"
				x.Container = "flux2-klein-nvfp4-api"
				x.Model = "flux2-klein-4b-nvfp4"
				x.Endpoint = "http://custom-host:8691"
				x.Host = "worker"
				x.MemoryGiB = 20
				if custom {
					x.Model = "custom-checkpoint"
				}
			}
		}
		for i := range cat.Bundles {
			cat.Bundles[i].WorkloadSwap = false
			// Recreate the pre-residency catalog used by this migration fixture.
			if binding, exists := cat.Bundles[i].Bindings["flux2"]; exists {
				binding.KeepResident = nil
				cat.Bundles[i].Bindings["flux2"] = binding
			}
		}
		cfg := Config{Image: ImageConfig{Model: "flux2-klein-4b-nvfp4", Mode: "paint"}, Runtime: RuntimeConfig{Catalog: &cat}}
		cfg.migrateQwenImage21()
		validated, err := orchestrator.ValidateCatalog(*cfg.Runtime.Catalog)
		if err != nil {
			t.Fatal(err)
		}
		x, _ := validated.Component("flux2")
		if x.Endpoint != "http://custom-host:8691" || x.Host != "worker" || x.MemoryGiB != 20 {
			t.Fatal("bindings or custom budget changed", x)
		}
		if custom {
			if x.ComposeAsset != "compose.flux2.yaml" || cfg.Image.Mode != "paint" {
				t.Fatal("custom recipe changed")
			}
		} else if x.ComposeAsset != "compose.qwen-image21.yaml" || cfg.Image.Mode != "qwen-image21" {
			t.Fatal("builtin recipe did not migrate")
		}
	}
}

func TestQwenImageMigrationPreservesExternalImageConfiguration(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "flux2" {
			cat.Components[i].ComposeAsset = "compose.flux2.yaml"
			cat.Components[i].Model = "flux2-klein-4b-nvfp4"
		}
	}
	before := ImageConfig{Enabled: true, Endpoint: "http://custom-image-service:8188", Model: "flux2-klein-4b-nvfp4", Mode: "paint"}
	cfg := Config{Runtime: RuntimeConfig{Mode: "external", Catalog: &cat}, Image: before}
	cfg.migrateQwenImage21()
	if cfg.Image != before {
		t.Fatal("external image configuration changed", cfg.Image)
	}
}

func TestFormerQWIMNamesUpgradeWithoutChangingModelOrBudgets(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		x := &cat.Components[i]
		if x.ID == "flux2" {
			x.Name = "QWIM 2.1"
			x.ComposeAsset = "compose.qwim21.yaml"
			x.Container = "sparktalk-qwim21"
			x.MemoryGiB = 20
			x.WorkspaceMemoryGiB = 12
		}
	}
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "flash-next" {
			cat.Bundles[i].Description = "QAD + QWIM 2.1"
		}
	}
	cfg := Config{Image: ImageConfig{Mode: "qwim21"}, Runtime: RuntimeConfig{Mode: "managed", BuiltinRevision: 24, Catalog: &cat, Bundle: "flash-next"}}
	cfg.Normalize()
	x, _ := cfg.Runtime.Catalog.Component("flux2")
	if x.Name != "Qwen Image 2.1" || x.ComposeAsset != "compose.qwen-image21.yaml" || x.Container != "sparktalk-qwen-image21" || cfg.Image.Mode != "qwen-image21" {
		t.Fatalf("incomplete naming upgrade: %+v mode=%s", x, cfg.Image.Mode)
	}
	if x.MemoryGiB != 20 || x.WorkspaceMemoryGiB != 12 || x.Model != "qwen-image-2.1-uc-nvfp4" {
		t.Fatal("naming changed model or budget", x)
	}
	b, _ := cfg.Runtime.Catalog.Bundle("flash-next")
	if b.Description != "QAD + Qwen Image 2.1" {
		t.Fatal("old description remains", b.Description)
	}
}
