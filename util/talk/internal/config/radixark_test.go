package config

import (
	"path/filepath"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestRetiredRadixArkPreservesSharedTP2AndDoesNotRestart(t *testing.T) {
	cat, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := cat.Bundle("flash-next-radixark"); ok {
		t.Fatal("retired default remains")
	}
	cat.Components = append(cat.Components, orchestrator.Component{ID: "flash-next-radixark", ComposeAsset: "compose.flash-next.yaml"})
	cat.Bundles = append(cat.Bundles, orchestrator.Bundle{ID: "flash-next-radixark", Components: []string{"flash-next-radixark"}})
	cfg := Config{Runtime: RuntimeConfig{Catalog: &cat, Mode: "managed", BuiltinRevision: 33, Bundle: "flash-next-radixark", ActiveBundle: "flash-next-tp2", AutoStart: true}}
	cfg.Normalize()
	if cfg.Runtime.AutoStart || cfg.Runtime.Bundle != "flash-next-tp2" || cfg.Runtime.ActiveBundle != "flash-next-tp2" {
		t.Fatal("unsafe fallback")
	}
	if _, ok := cfg.Runtime.Catalog.Bundle("flash-next-radixark"); ok {
		t.Fatal("retired set returned")
	}
	x, ok := cfg.Runtime.Catalog.Component("flash-next-tp2")
	if !ok || x.Model != orchestrator.QwenRadixArk {
		t.Fatal("shared TP2 model changed")
	}
	path := filepath.Join(t.TempDir(), "sparktalk.yaml")
	if err := Save(path, cfg); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := loaded.Runtime.Catalog.Bundle("flash-next-radixark"); ok {
		t.Fatal("retired set returned after reload")
	}
}
