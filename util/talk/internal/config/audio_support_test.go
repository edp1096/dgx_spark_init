package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestConstrainedSetsExcludeSharedAudioAfterReload(t *testing.T) {
	cat, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for i := range cat.Bundles {
		cat.Bundles[i].DisabledSupport = nil
	}
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Catalog: &cat, BuiltinRevision: 52}}
	cfg.Normalize()
	for _, id := range []string{"flash-next-fp4-local", "flash-next-tp2", "glm53-worker-extra", "ds41"} {
		cfg.Runtime.ActiveBundle = id
		for _, key := range []string{"fish", "music-ace"} {
			if cfg.SupportEnabled(key) || !cfg.SupportDisabled(key) {
				t.Fatalf("%s exposes %s", id, key)
			}
			if _, ok := cfg.Runtime.Catalog.ResolveSupport(id, "extra-"+key); ok {
				t.Fatalf("%s resolves %s", id, key)
			}
		}
	}
	cfg.Runtime.ActiveBundle = "qwen38fn_exl3"
	if !cfg.SupportEnabled("fish") || !cfg.SupportEnabled("music-ace") {
		t.Fatal("unrelated set changed")
	}
	for _, id := range []string{"extra-fish", "extra-music-ace"} {
		if _, ok := cfg.Runtime.Catalog.Component(id); !ok {
			t.Fatal("global service removed", id)
		}
	}
	cfg.Normalize()
	b, _ := cfg.Runtime.Catalog.Bundle("flash-next-fp4-local")
	if len(b.DisabledSupport) != 2 {
		t.Fatal("normalization is not idempotent", b.DisabledSupport)
	}
}
