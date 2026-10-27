package config

import (
	"reflect"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestMiniMaxPDMDUpgradePreservesCustomDeployment(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	x, _ := cat.Component("qwim-mmh3")
	x.Name, x.MemoryGiB, x.StartupMemoryGiB = "Qwen Image 2.1 + MiniMax H3", 28, 18
	x.Host, x.Endpoint = "worker", "http://custom-host:8730"
	custom := x
	custom.ID, custom.Container, custom.MemoryGiB = "custom-h3", "custom-h3", 45
	cat.Components = []orchestrator.Component{x, custom}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &cat, Bundle: "ds41", ActiveBundle: "qwen38fn_exl3"}}
	cfg.upgradeMiniMaxPDMD()
	got := cfg.Runtime.Catalog.Components[0]
	if got.MemoryGiB != 30 || got.StartupMemoryGiB != 20 || got.Host != "worker" || got.Endpoint != x.Endpoint {
		t.Fatalf("incorrect PDMD migration: %+v", got)
	}
	if !reflect.DeepEqual(cfg.Runtime.Catalog.Components[1], custom) {
		t.Fatal("custom component changed")
	}
	if cfg.Runtime.Bundle != "ds41" || cfg.Runtime.ActiveBundle != "qwen38fn_exl3" {
		t.Fatal("model selection changed")
	}
	cfg.Runtime.Catalog.Components[0].MemoryGiB = 42
	cfg.upgradeMiniMaxPDMD()
	if cfg.Runtime.Catalog.Components[0].MemoryGiB != 42 {
		t.Fatal("custom memory budget changed")
	}
}
