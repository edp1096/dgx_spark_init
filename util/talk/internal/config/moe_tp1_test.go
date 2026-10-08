package config

import "testing"

func TestSingleSparkMoEMigration(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 4
	c.Runtime.Bundle = "flash-next"
	for i := range c.Runtime.Catalog.Components {
		if c.Runtime.Catalog.Components[i].ID == "ornith35" {
			c.Runtime.Catalog.Components[i].Name = "Custom Ornith"
		}
	}
	c.Normalize()
	c.Normalize()
	for _, id := range []string{"ornith35", "gemma26"} {
		b, ok := c.Runtime.Catalog.Bundle(id)
		if !ok || b.ContextTokens != 1048576 {
			t.Fatalf("Missing 1M TP1 bundle: %s", id)
		}
		count := 0
		for _, x := range c.Runtime.Catalog.Components {
			if x.ID == id {
				count++
			}
		}
		if count != 1 {
			t.Fatalf("Duplicate component: %s", id)
		}
	}
	item, _ := c.Runtime.Catalog.Component("ornith35")
	if item.Name != "Custom Ornith" || c.Runtime.Bundle != "flash-next" {
		t.Fatal("Custom settings overwritten")
	}
}
