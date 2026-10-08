package config

import (
	"encoding/json"
	"testing"

	"sparktalk/internal/orchestrator"
)

func TestQwen38FNEXL3MigrationPreservesQADSelectionAndUserEdits(t *testing.T) {
	catalog, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	id := "qwen38fn_exl3"
	for i, item := range catalog.Bundles {
		if item.ID == id {
			catalog.Bundles = append(catalog.Bundles[:i], catalog.Bundles[i+1:]...)
			break
		}
	}
	for i := range catalog.Components {
		if catalog.Components[i].ID == id {
			catalog.Components[i].Name = "내 EXL3"
		}
	}
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", BuiltinRevision: 26, Bundle: "flash-next", ActiveBundle: "flash-next", AutoStart: false, Catalog: &catalog}}
	cfg.Normalize()
	if cfg.Runtime.Bundle != "flash-next" || cfg.Runtime.ActiveBundle != "flash-next" || cfg.Runtime.AutoStart {
		t.Fatal("migration changed the active/default selection or auto-start")
	}
	component, _ := cfg.Runtime.Catalog.Component(id)
	if component.Name != "내 EXL3" {
		t.Fatal("user edit replaced")
	}
	bundle, ok := cfg.Runtime.Catalog.Bundle(id)
	if !ok || bundle.ContextTokens != 1048576 || bundle.ModelType != "qwen38fn_exl3" || !bundle.WorkloadSwap {
		t.Fatalf("native profile missing: %+v", bundle)
	}
	if _, retired := cfg.Runtime.Catalog.Bundle("flash-next-exl3"); retired {
		t.Fatal("retired legacy EXL3 returned")
	}
	before, _ := json.Marshal(cfg)
	cfg.Normalize()
	after, _ := json.Marshal(cfg)
	if string(before) != string(after) {
		t.Fatal("migration is not idempotent")
	}
	// An explicit later removal must survive normalization.
	for i, item := range cfg.Runtime.Catalog.Bundles {
		if item.ID == id {
			cfg.Runtime.Catalog.Bundles = append(cfg.Runtime.Catalog.Bundles[:i], cfg.Runtime.Catalog.Bundles[i+1:]...)
			break
		}
	}
	cfg.Normalize()
	if _, ok := cfg.Runtime.Catalog.Bundle(id); ok {
		t.Fatal("user-removed native set reappeared")
	}
}

func TestQwen38FNEXL3SwitchWiresItsProfileAndSharedServices(t *testing.T) {
	catalog, _ := orchestrator.LoadCatalog()
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Catalog: &catalog, Bundle: "flash-next", ActiveBundle: "qwen38fn_exl3"}, Model: ModelConfig{ReasoningEffort: "medium"}}
	cfg.Normalize()
	if cfg.Model.DefaultModel != "qwen38fn_exl3" || cfg.Model.Endpoint != "http://127.0.0.1:18002" || cfg.Model.ModelType != "qwen38fn_exl3" || cfg.Context.WindowTokens != 1048576 || cfg.Model.ReasoningEffort != "medium" {
		t.Fatalf("native profile wiring incorrect: %+v", cfg.Model)
	}
	if cfg.Image.Endpoint != "http://127.0.0.1:8730" || cfg.Image.Mode != "basic" || cfg.ASR.Endpoint != "http://127.0.0.1:8693" || cfg.TTS.Endpoint != "http://127.0.0.1:8692" || cfg.Extra.SSHEndpoint != "http://127.0.0.1:8699" {
		t.Fatal("shared service wiring incorrect")
	}
	if cfg.Runtime.Bundle != "flash-next" {
		t.Fatal("live switch changed startup default")
	}
}

func TestQwen38FNEXL3ResidencyMigrationPreservesOtherProfilesAndOverrides(t *testing.T) {
	catalog := legacyQwen38FNEXL3Catalog(t)
	for i := range catalog.Bundles {
		if catalog.Bundles[i].ID == "qwen38fn_exl3" {
			for _, id := range []string{"flux2", "nemotron-asr", "qwen3-tts"} {
				binding := catalog.Bundles[i].Bindings[id]
				binding.KeepResident = nil
				if id == "flux2" {
					binding.MemoryGiB = nil
				}
				catalog.Bundles[i].Bindings[id] = binding
			}
		}
	}
	qad, _ := catalog.Bundle("flash-next")
	beforeQAD, _ := json.Marshal(qad)
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", BuiltinRevision: 27, Bundle: "flash-next", ActiveBundle: "qwen38fn_exl3", Catalog: &catalog}}
	cfg.keepQwen38FNEXL3AuxiliariesResident()
	for _, id := range []string{"flux2", "nemotron-asr", "qwen3-tts"} {
		x, ok := cfg.Runtime.Catalog.ResolveComponent("qwen38fn_exl3", id)
		if !ok || !x.KeepResident || !x.StartAfterLLM {
			t.Fatalf("missing resident auxiliary: %+v", x)
		}
		if id == "flux2" && x.MemoryGiB != 24 {
			t.Fatal("resident image needs a separate full-cache budget")
		}
	}
	qad, _ = cfg.Runtime.Catalog.Bundle("flash-next")
	afterQAD, _ := json.Marshal(qad)
	if string(beforeQAD) != string(afterQAD) {
		t.Fatal("EXL3 migration changed QAD")
	}
	before, _ := json.Marshal(cfg)
	cfg.keepQwen38FNEXL3AuxiliariesResident()
	after, _ := json.Marshal(cfg)
	if string(before) != string(after) {
		t.Fatal("residency migration is not idempotent")
	}
	for i := range cfg.Runtime.Catalog.Bundles {
		if cfg.Runtime.Catalog.Bundles[i].ID == "qwen38fn_exl3" {
			keep, budget := false, 31.0
			binding := cfg.Runtime.Catalog.Bundles[i].Bindings["flux2"]
			binding.KeepResident, binding.MemoryGiB = &keep, &budget
			cfg.Runtime.Catalog.Bundles[i].Bindings["flux2"] = binding
		}
	}
	cfg.Runtime.BuiltinRevision = 27
	cfg.keepQwen38FNEXL3AuxiliariesResident()
	x, _ := cfg.Runtime.Catalog.ResolveComponent("qwen38fn_exl3", "flux2")
	if x.KeepResident || x.MemoryGiB != 31 {
		t.Fatal("explicit user residency/budget override replaced")
	}
}

func TestNewQwen38FNEXL3SetPreservesRemoteAuxiliaryDeployment(t *testing.T) {
	catalog, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for i := range catalog.Bundles {
		catalog.Bundles[i].WorkloadSwap = false
	}
	for i, bundle := range catalog.Bundles {
		if bundle.ID == "qwen38fn_exl3" {
			catalog.Bundles = append(catalog.Bundles[:i], catalog.Bundles[i+1:]...)
			break
		}
	}
	for i := range catalog.Components {
		if catalog.Components[i].ID == "qwim-mmh3" {
			catalog.Components[i].Host = "worker"
			catalog.Components[i].KeepResident = false
			catalog.Components[i].MemoryGiB = 31
		}
	}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog}}
	cfg.addQwen38FNEXL3Set()
	cfg.keepQwen38FNEXL3AuxiliariesResident()
	validated, err := orchestrator.ValidateCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Runtime.Catalog = &validated
	x, ok := cfg.Runtime.Catalog.ResolveComponent("qwen38fn_exl3", "qwim-mmh3")
	if !ok || x.KeepResident || x.Host != "worker" || x.MemoryGiB != 31 {
		t.Fatalf("remote image deployment replaced: %+v", x)
	}
	if _, err := orchestrator.ValidateCatalog(*cfg.Runtime.Catalog); err != nil {
		t.Fatal(err)
	}
}
