package config

import "testing"

func TestQwenTP2MigrationPreservesUserSettings(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 2
	cat := c.Runtime.Catalog
	components := cat.Components[:0]
	for _, x := range cat.Components {
		if x.ID != "flash-next-tp2" {
			components = append(components, x)
		}
	}
	cat.Components = components
	bundles := cat.Bundles[:0]
	for _, x := range cat.Bundles {
		if x.ID != "flash-next-tp2" {
			bundles = append(bundles, x)
		}
	}
	cat.Bundles = bundles
	c.Normalize()
	c.Normalize()
	n := 0
	for _, b := range c.Runtime.Catalog.Bundles {
		if b.ID == "flash-next-tp2" {
			n++
			if b.ContextTokens != 1048576 {
				t.Fatal("wrong context")
			}
		}
	}
	for i := range c.Runtime.Catalog.Components {
		if c.Runtime.Catalog.Components[i].ID == "flash-next-tp2" {
			c.Runtime.Catalog.Components[i].Name = "User TP2"
		}
	}
	c.Runtime.BuiltinRevision = 2
	c.Normalize()
	for _, item := range c.Runtime.Catalog.Components {
		if item.ID == "flash-next-tp2" && item.Name != "User TP2" {
			t.Fatal("user component was overwritten")
		}
	}
	if n != 1 {
		t.Fatalf("got %d TP2 bundles", n)
	}
}
