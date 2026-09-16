package config

import (
	"reflect"
	"testing"

	"sparktalk/internal/orchestrator"
)

func TestGemmaCheckpointMigrationPreservesUserConfiguration(t *testing.T) {
	catalog, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	c := Config{}
	catalog.Hosts["worker"] = catalog.Hosts["local"]
	c.Runtime.Catalog = &catalog
	c.Runtime.BuiltinRevision = 8
	c.Runtime.Mode = "managed"
	c.Runtime.Bundle = "ornith35"
	c.Model.DefaultModel = "edp1096/Huihui-Ornith-1.5-35B-A3B-NVFP4"
	const old = "sakamakismile/Huihui-gemma-4-26B-A4B-it-qat-abliterated-MTP-NVFP4"
	var order []string
	for i := range catalog.Components {
		x := &catalog.Components[i]
		if x.ID == "gemma26" {
			x.Model = old
			x.Name = "내 젬마"
			x.Host = "worker"
			x.RuntimeOptions = map[string]string{"DRAFT_VOCAB": "ko64k", "MTP_TOKENS": "0"}
		}
	}
	for i := range catalog.Bundles {
		x := &catalog.Bundles[i]
		order = append(order, x.ID)
		if x.ID == "gemma26" {
			x.ModelID = old
			x.Name = "짧은 이름"
		}
	}
	c.Normalize()
	c.Normalize()
	catalog = *c.Runtime.Catalog
	x, _ := catalog.Component("gemma26")
	b, _ := catalog.Bundle("gemma26")
	if x.Model == old || x.Model != b.ModelID {
		t.Fatal("model identities were not migrated together")
	}
	if x.Name != "내 젬마" || x.Host != "worker" || x.RuntimeOptions["DRAFT_VOCAB"] != "ko64k" || x.RuntimeOptions["MTP_TOKENS"] != "0" || b.Name != "짧은 이름" {
		t.Fatal("user configuration changed")
	}
	var after []string
	for _, b := range catalog.Bundles {
		after = append(after, b.ID)
	}
	if !reflect.DeepEqual(order, after) || c.Runtime.Bundle != "ornith35" || c.Model.DefaultModel != "edp1096/Huihui-Ornith-1.5-35B-A3B-NVFP4" {
		t.Fatalf("order/selection changed: before=%v after=%v bundle=%q model=%q", order, after, c.Runtime.Bundle, c.Model.DefaultModel)
	}
}

func TestGemmaCheckpointMigrationLeavesCustomCheckpoint(t *testing.T) {
	catalog, _ := orchestrator.LoadCatalog()
	c := Config{}
	catalog.Hosts["worker"] = catalog.Hosts["local"]
	c.Runtime.Catalog = &catalog
	c.Runtime.BuiltinRevision = 8
	for i := range catalog.Components {
		if catalog.Components[i].ID == "gemma26" {
			catalog.Components[i].Model = "custom/gemma"
		}
	}
	for i := range catalog.Bundles {
		if catalog.Bundles[i].ID == "gemma26" {
			catalog.Bundles[i].ModelID = "custom/gemma"
		}
	}
	c.Normalize()
	catalog = *c.Runtime.Catalog
	x, _ := catalog.Component("gemma26")
	b, _ := catalog.Bundle("gemma26")
	if x.Model != "custom/gemma" || b.ModelID != "custom/gemma" {
		t.Fatal("custom checkpoint overwritten")
	}
}
