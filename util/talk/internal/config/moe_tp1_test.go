package config

import (
	"path/filepath"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestOrnithRetirementSurvivesLegacyMigrationAndReload(t *testing.T) {
	for _, id := range []string{"ornith35", "custom-ornith-copy"} {
		t.Run(id, func(t *testing.T) {
			c := Config{}
			c.Normalize()
			c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, orchestrator.Component{ID: id, ComposeAsset: "compose.ornith35.yaml"})
			c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, orchestrator.Bundle{ID: id, Components: []string{id, "extra-embedding", "extra-ssh"}})
			c.Runtime.Bundle = "ds41"
			c.Runtime.ActiveBundle = id
			c.Runtime.AutoStart = true
			c.Runtime.BuiltinRevision = 4
			c.Normalize()
			if c.Runtime.ActiveBundle != "ds41" || c.Runtime.Bundle != "ds41" || c.Runtime.AutoStart {
				t.Fatal("retired selection or automatic launch survived")
			}
			p := filepath.Join(t.TempDir(), "talk.yaml")
			if err := Save(p, c); err != nil {
				t.Fatal(err)
			}
			loaded, _, err := Load(p)
			if err != nil {
				t.Fatal(err)
			}
			for _, x := range []string{id, "ornith35"} {
				if _, ok := loaded.Runtime.Catalog.Component(x); ok {
					t.Fatal("retired component returned")
				}
				if _, ok := loaded.Runtime.Catalog.Bundle(x); ok {
					t.Fatal("retired set returned")
				}
			}
			for _, x := range []string{"flash-next-fp4-local", "qwen38fn_exl3", "extra-embedding", "extra-ssh"} {
				if _, ok := loaded.Runtime.Catalog.Component(x); !ok {
					t.Fatalf("shared/surviving component removed: %s", x)
				}
			}
		})
	}
}
