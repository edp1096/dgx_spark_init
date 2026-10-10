package config

import (
	"reflect"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestEmbeddingMigrationPreservesCoreAndScopesSearch(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 37
	c.Runtime.ActiveBundle = "flash-next"
	c.Embedding.Enabled = true
	before, _ := c.Runtime.Catalog.ResolveComponent("flash-next", "flash-next")
	for i := range c.Runtime.Catalog.Components {
		if c.Runtime.Catalog.Components[i].ID == "extra-embedding" {
			c.Runtime.Catalog.Components[i].KeepResident = false
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		ids := []string{}
		for _, id := range b.Components {
			if id != "extra-embedding" {
				ids = append(ids, id)
			}
		}
		b.Components = ids
	}
	c.Normalize()
	c.Normalize()
	after, _ := c.Runtime.Catalog.ResolveComponent("flash-next", "flash-next")
	if !reflect.DeepEqual(before, after) || c.Runtime.ActiveBundle != "flash-next" || !c.Embedding.Enabled {
		t.Fatal("migration changed core or user preference")
	}
	for _, id := range []string{"qwen38fn_exl3", "flash-next-radixark", "flash-next", "gemma26"} {
		c.Runtime.ActiveBundle = id
		want := id == "qwen38fn_exl3" || id == "flash-next-radixark"
		if c.SemanticSearchEnabled() != want {
			t.Fatal("semantic scope", id)
		}
		x, member := c.Runtime.Catalog.ResolveComponent(id, "extra-embedding")
		if member != want || (member && (!x.KeepResident || !x.StartAfterLLM)) {
			t.Fatal("residency scope", id)
		}
	}
	c.Runtime.Mode = "external"
	if !c.SemanticSearchEnabled() {
		t.Fatal("external service permission lost")
	}
	c.Embedding.Enabled = false
	if c.SemanticSearchEnabled() {
		t.Fatal("disabled preference ignored")
	}
	if _, err := orchestrator.ValidateCatalog(*c.Runtime.Catalog); err != nil {
		t.Fatal(err)
	}
}
