package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestDisplayNamesPreserveCustomNamesBindingsAndRuntimeIdentity(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		switch cat.Components[i].ID {
		case "qwen38fn_exl3":
			cat.Components[i].Name = "Huihui Qwen3.8 Native EXL3"
		case "nemotron-asr":
			cat.Components[i].Name = "Nemotron ASR Q5_K"
		case "gemma26":
			cat.Components[i].Name = "내 문서 모델"
		}
	}
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "qwen38fn_exl3" {
			cat.Bundles[i].Name = "Qwen3.8 Flash-Next EXL3"
			name := "Nemotron ASR Q5_K"
			binding := cat.Bundles[i].Bindings["nemotron-asr"]
			binding.Name = &name
			cat.Bundles[i].Bindings["nemotron-asr"] = binding
		}
	}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &cat}}
	cfg.normalizeRuntimeDisplayNames()
	for _, c := range cfg.Runtime.Catalog.Components {
		switch c.ID {
		case "qwen38fn_exl3":
			if c.Name != "Qwen 3.8 Flash-Next EXL3 3bit" || c.Model != "qwen38fn_exl3" {
				t.Fatal(c)
			}
		case "nemotron-asr":
			if c.Name != "Nemotron 3.5 ASR" {
				t.Fatal(c)
			}
		case "gemma26":
			if c.Name != "내 문서 모델" {
				t.Fatal("custom name overwritten")
			}
		}
	}
	for _, b := range cfg.Runtime.Catalog.Bundles {
		if b.ID == "qwen38fn_exl3" && (*b.Bindings["nemotron-asr"].Name != "Nemotron 3.5 ASR" || !*b.Bindings["nemotron-asr"].KeepResident) {
			t.Fatal("name migration changed residency or missed bindings")
		}
	}
}
