package config

import "testing"

func TestHuihuiModelMigrationPreservesCustomSettings(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 3
	c.Model.DefaultModel = "qwen3.8-flash-next"
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "flash-next" {
			x.Model = "qwen3.8-flash-next"
			x.Name = "My Qwen"
			x.Endpoint = "http://custom:8000"
		}
		if x.ID == "flash-next-tp2" {
			x.Model = "my/custom-model"
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		x := &c.Runtime.Catalog.Bundles[i]
		if x.ID == "flash-next" {
			x.ModelID = "qwen3.8-flash-next"
			x.ContextTokens = 32768
		}
	}
	c.Normalize()
	c.Normalize()
	x, _ := c.Runtime.Catalog.Component("flash-next")
	if x.Model != "local-inference-lab/Qwen3.8-Flash-Next-NVFP4" || x.Name != "My Qwen" || x.Endpoint != "http://custom:8000" {
		t.Fatalf("migration changed custom settings: %+v", x)
	}
	y, _ := c.Runtime.Catalog.Component("flash-next-tp2")
	if y.Model != "my/custom-model" {
		t.Fatal("custom model overwritten")
	}
	b, _ := c.Runtime.Catalog.Bundle("flash-next")
	if b.ModelID != x.Model || b.ContextTokens != 32768 {
		t.Fatal("bundle migration incorrect")
	}
	if c.Model.DefaultModel != x.Model {
		t.Fatal("active model was not migrated")
	}
}
