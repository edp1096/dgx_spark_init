package config

import (
	"reflect"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestDS41TensorFoldUpgradeKeepsSelectionAndCustomHosts(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		x := &cat.Components[i]
		if x.ID == "ds41" {
			x.Model, x.Container, x.WorkerContainer = "deepseek-v4.1-flash", "ds41-stream-0", "ds41-stream-1"
			x.RuntimeOptions = map[string]string{"DSV41_PRELOAD_COUNT": "224"}
		}
	}
	for i := range cat.Bundles {
		b := &cat.Bundles[i]
		if b.ID != "ds41" {
			continue
		}
		b.ModelID, b.ContextTokens, b.WorkloadSwap = "deepseek-v4.1-flash", 65536, false
		for _, id := range []string{"nemotron-asr", "qwen3-tts"} {
			d := b.Bindings[id]
			host, addr, endpoint := "worker", "192.168.100.60", "http://192.168.100.60:8693"
			if id == "qwen3-tts" {
				endpoint = "http://192.168.100.60:8692"
			}
			d.Host, d.Endpoint, d.BindAddress = &host, &endpoint, &addr
			b.Bindings[id] = d
		}
	}
	host := cat.Hosts["worker"]
	host.User = "custom-user"
	host.IdentityFile = "/custom/key"
	cat.Hosts["worker"] = host
	cfg := Config{Runtime: RuntimeConfig{BuiltinRevision: 43, Catalog: &cat, Bundle: "flash-next-fp4-local", ActiveBundle: "flash-next"}}
	cfg.Normalize()
	x, _ := cfg.Runtime.Catalog.Component("ds41")
	b, _ := cfg.Runtime.Catalog.Bundle("ds41")
	if x.Model != "DeepSeek-V4.1-Flash-TF" || b.ContextTokens != 1048576 || !b.WorkloadSwap || x.RuntimeOptions["DSV41_PRELOAD_COUNT"] != "" {
		t.Fatal("retired streaming recipe not upgraded", x, b)
	}
	if cfg.Runtime.Bundle != "flash-next-fp4-local" || cfg.Runtime.ActiveBundle != "flash-next" || cfg.Runtime.Catalog.Hosts["worker"] != host {
		t.Fatal("user selection/connection changed")
	}
	before := cfg.Public()
	cfg.Normalize()
	if !reflect.DeepEqual(before, cfg.Public()) {
		t.Fatal("upgrade is not idempotent")
	}
}

func TestDS41TensorFoldLeavesCustomRuntimeAlone(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "ds41" {
			cat.Components[i].Model = "my-custom-deepseek"
		}
	}
	cfg := Config{Runtime: RuntimeConfig{BuiltinRevision: 43, Catalog: &cat}}
	cfg.Normalize()
	x, _ := cfg.Runtime.Catalog.Component("ds41")
	if x.Model != "my-custom-deepseek" {
		t.Fatal("custom deployment overwritten")
	}
}

func TestDS41ExplicitOriginalSelectionSurvivesNormalization(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "ds41" {
			cat.Components[i].RuntimeOptions["MODEL_VARIANT"] = "official"
		}
	}
	cfg := Config{Runtime: RuntimeConfig{BuiltinRevision: 44, Catalog: &cat, Bundle: "ds41", ActiveBundle: "ds41"}}
	cfg.Normalize()
	x, _ := cfg.Runtime.Catalog.Component("ds41")
	if x.RuntimeOptions["MODEL_VARIANT"] != "official" || x.ModelPresentation.SelectedVariant != "official" {
		t.Fatal("normalization overwrote the user's explicit original weight choice")
	}
}
