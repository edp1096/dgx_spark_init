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
	if cfg.Runtime.Bundle != "qwen27-exl3" || cfg.Runtime.ActiveBundle != "qwen27-exl3" || cfg.Runtime.AutoStart {
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

func TestSavedGGUFCatalogMigratesToEXL3(t *testing.T) {
	catalog, _ := orchestrator.LoadCatalog()
	catalog.Components[0] = orchestrator.Component{ID: "qwen27-gguf", ComposeAsset: "compose.qwen27-gguf.yaml"}
	catalog.Bundles[0] = orchestrator.Bundle{ID: "qwen27-gguf", Components: []string{"qwen27-gguf", "extra-ssh"}}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog, Bundle: "qwen27-gguf", ActiveBundle: "qwen27-gguf", AutoStart: true}}
	cfg.Normalize()
	if cfg.Runtime.Bundle != "qwen27-exl3" || cfg.Runtime.ActiveBundle != "qwen27-exl3" || cfg.Runtime.AutoStart || cfg.Context.WindowTokens != 131072 {
		t.Fatalf("migration failed: %+v", cfg.Runtime)
	}
	if _, ok := cfg.Runtime.Catalog.Component("qwen27-gguf"); ok {
		t.Fatal("retired model survived")
	}
	if _, ok := cfg.Runtime.Catalog.Component("extra-ssh"); !ok {
		t.Fatal("shared extra lost")
	}
	count := len(cfg.Runtime.Catalog.Components)
	cfg.Normalize()
	if len(cfg.Runtime.Catalog.Components) != count {
		t.Fatal("migration is not idempotent")
	}
	for i := range cfg.Runtime.Catalog.Components {
		if cfg.Runtime.Catalog.Components[i].ID == "qwen27-exl3" {
			cfg.Runtime.Catalog.Components[i].RuntimeOptions = map[string]string{"MAX_MODEL_LEN": "262144"}
		}
	}
	cfg.Normalize()
	if cfg.Context.WindowTokens != 262144 {
		t.Fatalf("context override ignored: %d", cfg.Context.WindowTokens)
	}
	c, _ := cfg.Runtime.Catalog.ResolveComponent("qwen27-exl3", "qwen27-exl3")
	if c.MemoryGiB < 27 {
		t.Fatal("larger KV not included in memory budget")
	}
}

func TestGGUFContextChoiceSurvivesEXL3Migration(t *testing.T) {
	catalog, _ := orchestrator.LoadCatalog()
	for i, c := range catalog.Components {
		if c.ID == "qwen27-exl3" {
			catalog.Components = append(catalog.Components[:i], catalog.Components[i+1:]...)
			break
		}
	}
	for i, b := range catalog.Bundles {
		if b.ID == "qwen27-exl3" {
			catalog.Bundles = append(catalog.Bundles[:i], catalog.Bundles[i+1:]...)
			break
		}
	}
	options := map[string]string{"MAX_MODEL_LEN": "65536", "MTP_TOKENS": "3"}
	catalog.Components = append(catalog.Components, orchestrator.Component{ID: "qwen27-gguf", ComposeAsset: "compose.qwen27-gguf.yaml"})
	catalog.Bundles = append(catalog.Bundles, orchestrator.Bundle{ID: "qwen27-gguf", Components: []string{"qwen27-gguf"}, Bindings: map[string]orchestrator.Deployment{"qwen27-gguf": {RuntimeOptions: &options}}})
	cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog, Bundle: "qwen27-gguf", ActiveBundle: "qwen27-gguf"}, Model: ModelConfig{ReasoningEffort: "medium"}}
	cfg.Normalize()
	if cfg.Context.WindowTokens != 65536 || cfg.Model.ReasoningEffort != "medium" {
		t.Fatalf("preferences lost: context=%d effort=%s", cfg.Context.WindowTokens, cfg.Model.ReasoningEffort)
	}
	c, _ := cfg.Runtime.Catalog.ResolveComponent("qwen27-exl3", "qwen27-exl3")
	if c.RuntimeOptions["MTP_TOKENS"] != "" {
		t.Fatal("GGUF-only runtime option survived")
	}
}
