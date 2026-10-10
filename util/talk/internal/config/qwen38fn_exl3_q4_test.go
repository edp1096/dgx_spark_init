package config

import (
	"encoding/json"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestEXL3Q4MigrationAndProfile(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i, b := range cat.Bundles {
		if b.ID == "qwen38fn_exl3_q4" {
			cat.Bundles = append(cat.Bundles[:i], cat.Bundles[i+1:]...)
			break
		}
	}
	for i, x := range cat.Components {
		if x.ID == "qwen38fn_exl3_q4" {
			cat.Components = append(cat.Components[:i], cat.Components[i+1:]...)
			break
		}
	}
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "qwen38fn_exl3" {
			cat.Bundles[i].Name = "Qwen 3.8 Flash-Next EXL3"
			cat.Bundles[i].ContextTokens = 524288
		}
	}
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", BuiltinRevision: 39, Bundle: "flash-next-radixark", ActiveBundle: "flash-next-radixark", Catalog: &cat}}
	cfg.Normalize()
	if cfg.Runtime.Bundle != "flash-next-radixark" || cfg.Runtime.ActiveBundle != "flash-next-radixark" || cfg.Runtime.AutoStart {
		t.Fatal("selection changed")
	}
	pos := map[string]int{}
	for i, b := range cfg.Runtime.Catalog.Bundles {
		pos[b.ID] = i
	}
	if pos["qwen38fn_exl3_q4"] != pos["qwen38fn_exl3"]+1 || pos["flash-next-radixark"] != pos["qwen38fn_exl3_q4"]+1 {
		t.Fatal("selector order", pos)
	}
	old, _ := cfg.Runtime.Catalog.Bundle("qwen38fn_exl3")
	if old.Name != "Qwen 3.8 Flash-Next EXL3 3bit" || old.ContextTokens != 524288 {
		t.Fatal("3bit settings changed", old)
	}
	before, _ := json.Marshal(cfg)
	cfg.Normalize()
	after, _ := json.Marshal(cfg)
	if string(before) != string(after) {
		t.Fatal("not idempotent")
	}
	cfg.ASR.Enabled = true
	cfg.Runtime.ActiveBundle = "qwen38fn_exl3_q4"
	cfg.Model.ReasoningEffort = "medium"
	cfg.Normalize()
	if cfg.Model.DefaultModel != "qwen38fn_exl3_q4" || cfg.Model.Endpoint != "http://127.0.0.1:18004" || cfg.Context.WindowTokens != 1048576 || cfg.Model.ModelType != "qwen38fn_exl3" || cfg.Model.ReasoningEffort != "medium" {
		t.Fatal("Q4 profile", cfg.Model)
	}
	if cfg.Image.Enabled || cfg.TTS.Enabled || !cfg.ASR.Enabled {
		t.Fatal("Q4 auxiliary wiring")
	}
	for _, id := range []string{"nemotron-asr", "extra-embedding"} {
		x, ok := cfg.Runtime.Catalog.ResolveComponent("qwen38fn_exl3_q4", id)
		if !ok || !x.KeepResident || !x.StartAfterLLM {
			t.Fatal("resident auxiliary", x)
		}
	}
	if cfg.Runtime.Bundle != "flash-next-radixark" {
		t.Fatal("switch altered startup default")
	}
	for i, b := range cfg.Runtime.Catalog.Bundles {
		if b.ID == "qwen38fn_exl3_q4" {
			cfg.Runtime.Catalog.Bundles = append(cfg.Runtime.Catalog.Bundles[:i], cfg.Runtime.Catalog.Bundles[i+1:]...)
			break
		}
	}
	cfg.Runtime.ActiveBundle = "flash-next-radixark"
	cfg.Normalize()
	if _, ok := cfg.Runtime.Catalog.Bundle("qwen38fn_exl3_q4"); ok {
		t.Fatal("user removal undone")
	}
}
