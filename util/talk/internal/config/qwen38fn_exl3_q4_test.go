package config

import (
	"path/filepath"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestEXL3Q4RetirementPersistsWithoutStartingReplacement(t *testing.T) {
	for _, revision := range []int{39, 45} {
		for _, renamed := range []bool{false, true} {
			cat, err := orchestrator.LoadCatalog()
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := cat.Bundle("qwen38fn_exl3_q4"); ok {
				t.Fatal("retired set in defaults")
			}
			if _, ok := cat.Component("qwen38fn_exl3_q4"); ok {
				t.Fatal("retired component in defaults")
			}
			id := "qwen38fn_exl3_q4"
			if renamed {
				id = "old-q4-copy"
			}
			cat.Components = append(cat.Components, orchestrator.Component{ID: id, ComposeAsset: "compose.qwen38fn_exl3_q4.yaml"})
			cat.Bundles = append(cat.Bundles, orchestrator.Bundle{ID: id, Components: []string{id, "extra-ssh"}})
			cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Catalog: &cat, BuiltinRevision: revision, Bundle: id, ActiveBundle: "flash-next-fp4-local", AutoStart: true}}
			cfg.Normalize()
			if cfg.Runtime.AutoStart || cfg.Runtime.Bundle != "flash-next-fp4-local" || cfg.Runtime.ActiveBundle != "flash-next-fp4-local" {
				t.Fatal("unsafe fallback")
			}
			for _, survivor := range []string{"qwen38fn_exl3", "flash-next-fp4-local"} {
				if _, ok := cfg.Runtime.Catalog.Bundle(survivor); !ok {
					t.Fatal("surviving set lost", survivor)
				}
			}
			if _, ok := cfg.Runtime.Catalog.Component("extra-ssh"); !ok {
				t.Fatal("shared service lost")
			}
			path := filepath.Join(t.TempDir(), "sparktalk.yaml")
			if err := Save(path, cfg); err != nil {
				t.Fatal(err)
			}
			loaded, _, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if _, ok := loaded.Runtime.Catalog.Bundle(id); ok {
				t.Fatal("retired set returned after reload")
			}
			if _, ok := loaded.Runtime.Catalog.Component(id); ok {
				t.Fatal("retired component returned after reload")
			}
		}
	}
}
