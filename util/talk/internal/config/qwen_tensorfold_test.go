package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestQwenTensorFoldUpgradePreservesDeploymentAndSelection(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		x := &cat.Components[i]
		switch x.ID {
		case "flash-next-fp4-local":
			x.Container, x.ProgressKind, x.StartupMemoryGiB = "vllm-qwen38-fp4-local", "b12x", 116
			x.Name, x.Endpoint, x.Port, x.Host = "My FP4", "http://custom:19004", 19004, "worker"
		case "qwen38fn_exl3":
			x.ProgressKind, x.StartupMemoryGiB = "exl3", 100
			x.MemoryGiB, x.WorkspaceMemoryGiB = 96, 6
		}
	}
	for i := range cat.Bundles {
		b := &cat.Bundles[i]
		if b.ID == "flash-next-fp4-local" {
			b.ModelType = "qwen3.8-vllm"
		}
		if b.ID == "qwen38fn_exl3" {
			b.ModelType = "qwen38fn_exl3"
		}
	}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &cat, BuiltinRevision: 44, Bundle: "ds41", ActiveBundle: "flash-next-fp4-local"}, Model: ModelConfig{ReasoningEffort: "none"}}
	cfg.Normalize()
	x, _ := cfg.Runtime.Catalog.Component("flash-next-fp4-local")
	if x.Container != "sparktalk-qwen38-fp4-local" || x.ProgressKind != "tensorfold" || x.StartupMemoryGiB != 103 || x.Name != "My FP4" || x.Endpoint != "http://custom:19004" || x.Port != 19004 || x.Host != "worker" {
		t.Fatal("built-in engine upgrade lost user deployment", x)
	}
	if cfg.Runtime.Bundle != "ds41" || cfg.Runtime.ActiveBundle != "flash-next-fp4-local" || cfg.Model.ModelType != "qwen3.8-tensorfold" || cfg.Model.ReasoningEffort != "none" {
		t.Fatal("selection or thinking changed")
	}
	exl3, _ := cfg.Runtime.Catalog.Component("qwen38fn_exl3")
	if exl3.ProgressKind != "tensorfold" || exl3.MemoryGiB != 96 || exl3.StartupMemoryGiB != 100 || exl3.WorkspaceMemoryGiB != 6 {
		t.Fatal("custom memory reservation changed", exl3)
	}
	_, q4 := cfg.Runtime.Catalog.Bundle("qwen38fn_exl3_q4")
	if q4 {
		t.Fatal("retired 4bit deployment returned")
	}
	cfg.Normalize()
	if cfg.Runtime.BuiltinRevision != 51 {
		t.Fatal("migration did not settle")
	}
}

func TestQwenTensorFoldUpgradePreservesCustomEngine(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "flash-next-fp4-local" {
			cat.Components[i].Container = "custom-engine"
			cat.Components[i].ProgressKind = "vllm"
		}
	}
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "flash-next-fp4-local" {
			cat.Bundles[i].ModelType = "qwen3.8-vllm"
		}
	}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &cat, BuiltinRevision: 44}}
	cfg.Normalize()
	x, _ := cfg.Runtime.Catalog.Component("flash-next-fp4-local")
	b, _ := cfg.Runtime.Catalog.Bundle(x.ID)
	if x.Container != "custom-engine" || x.ProgressKind != "vllm" || b.ModelType != "qwen3.8-vllm" {
		t.Fatal("custom engine was replaced")
	}
}

func TestQwenEXL3RetuningRefreshesOnlyStockDescription(t *testing.T) {
	const previous = "TensorFold · EXL3 3bit HQ h6 ng6 · Q8 KV 1M·YaRN 4·MTP 3 · 이미지·ASR·TTS·Extra"
	for _, description := range []string{previous, "My EXL3 profile"} {
		cat, _ := orchestrator.LoadCatalog()
		for i := range cat.Bundles {
			if cat.Bundles[i].ID == "qwen38fn_exl3" {
				cat.Bundles[i].Description = description
			}
		}
		cfg := Config{Runtime: RuntimeConfig{Catalog: &cat, BuiltinRevision: 51, Bundle: "ds41", ActiveBundle: "qwen38fn_exl3"}}
		cfg.Normalize()
		var actual orchestrator.Bundle
		for _, bundle := range cfg.Runtime.Catalog.Bundles {
			if bundle.ID == "qwen38fn_exl3" {
				actual = bundle
			}
		}
		if description == previous {
			if actual.Description == previous {
				t.Fatal("stale MTP description retained")
			}
		} else if actual.Description != description {
			t.Fatal("custom description changed")
		}
		if cfg.Runtime.Bundle != "ds41" || cfg.Runtime.ActiveBundle != "qwen38fn_exl3" {
			t.Fatal("selection changed")
		}
	}
}
