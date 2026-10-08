package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestGemmaSGLangMigrationPreservesSelection(t *testing.T) {
	cat, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	c := Config{}
	c.Runtime.Catalog = &cat
	c.Runtime.BuiltinRevision = 9
	c.Runtime.Mode = "managed"
	c.Runtime.Bundle = "gemma26"
	const model = "coolthor/Huihui-gemma-4-26B-A4B-it-abliterated-FP8-Dynamic"
	c.Model.DefaultModel = model
	c.Model.ModelType = "gemma4-vllm"
	for i := range cat.Components {
		if cat.Components[i].ID == "gemma26" {
			cat.Components[i].Container = "vllm-gemma26"
			cat.Components[i].Name = "내 모델"
		}
	}
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "gemma26" {
			cat.Bundles[i].ModelType = "gemma4-vllm"
		}
	}
	c.Normalize()
	x, _ := c.Runtime.Catalog.Component("gemma26")
	b, _ := c.Runtime.Catalog.Bundle("gemma26")
	if x.Container != "sglang-gemma26" || x.Name != "내 모델" || b.ModelType != "gemma4" || c.Model.ModelType != "gemma4" || c.Model.DefaultModel != "edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4" {
		t.Fatalf("migration mismatch: %+v / %+v", x, c.Model)
	}
}
