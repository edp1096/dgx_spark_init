package config

import (
	"reflect"
	"testing"
)

func TestExistingQADSetDefersASRAfterUpgrade(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 14
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "nemotron-asr" {
			x.MemoryGiB = 1.3
			x.StartupMemoryGiB = 0
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID == "flash-next" {
			delete(b.Bindings, "nemotron-asr")
		}
	}
	before, _ := c.Runtime.Catalog.Bundle("flash-next-tp2")
	c.Normalize()
	c.Normalize()
	x, _ := c.Runtime.Catalog.ResolveComponent("flash-next", "nemotron-asr")
	if !x.StartAfterLLM || x.StartupMemoryGiB != 3.5 || x.MemoryGiB != 6 {
		t.Fatalf("upgrade incomplete: %+v", x)
	}
	after, _ := c.Runtime.Catalog.Bundle("flash-next-tp2")
	// Shared service estimate changes its aggregate, but bindings/membership stay intact.
	before.MemoryGiB = after.MemoryGiB
	if !reflect.DeepEqual(before, after) {
		t.Fatal("TP2 profile changed")
	}
}
func TestASRBudgetPreservesCustomEstimateAndExplicitOrder(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 14
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "nemotron-asr" {
			x.MemoryGiB = 8
			x.StartupMemoryGiB = 4
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID == "flash-next" {
			d := b.Bindings["nemotron-asr"]
			v := false
			d.StartAfterLLM = &v
			b.Bindings["nemotron-asr"] = d
		}
	}
	c.Normalize()
	x, _ := c.Runtime.Catalog.ResolveComponent("flash-next", "nemotron-asr")
	if x.StartAfterLLM || x.MemoryGiB != 8 || x.StartupMemoryGiB != 4 {
		t.Fatalf("customization changed %+v", x)
	}
}
