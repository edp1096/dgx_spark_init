package config

import (
	"reflect"
	"testing"
)

func TestRadixArkMigrationPreservesQADAndSelection(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 33
	c.Runtime.ActiveBundle = "flash-next"
	before, _ := c.Runtime.Catalog.Bundle("flash-next")
	components := c.Runtime.Catalog.Components[:0]
	for _, x := range c.Runtime.Catalog.Components {
		if x.ID != "flash-next-radixark" {
			components = append(components, x)
		}
	}
	c.Runtime.Catalog.Components = components
	bundles := c.Runtime.Catalog.Bundles[:0]
	for _, b := range c.Runtime.Catalog.Bundles {
		if b.ID != "flash-next-radixark" {
			bundles = append(bundles, b)
		}
	}
	c.Runtime.Catalog.Bundles = bundles
	c.Normalize()
	c.Normalize()
	after, _ := c.Runtime.Catalog.Bundle("flash-next")
	if !reflect.DeepEqual(before, after) || c.Runtime.ActiveBundle != "flash-next" {
		t.Fatal("QAD profile/selection changed")
	}
	b, ok := c.Runtime.Catalog.Bundle("flash-next-radixark")
	if !ok || b.ContextTokens != 1048576 {
		t.Fatal("missing 1M RadixArk")
	}
	count := 0
	for _, b := range c.Runtime.Catalog.Bundles {
		if b.ID == "flash-next-radixark" {
			count++
		}
	}
	if count != 1 {
		t.Fatal("non-idempotent migration")
	}
}

func TestRadixArkWithExtrasSurvivesLegacyTTSMigration(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 31
	c.Normalize()
	c.ApplyManagedBundle("flash-next-radixark")
	b, ok := c.Runtime.Catalog.Bundle("flash-next-radixark")
	if !ok || !reflect.DeepEqual(b.Components, []string{"flash-next-radixark", "nemotron-asr", "extra-media", "extra-ssh", "extra-collector", "extra-documents", "extra-embedding"}) || c.ASR.Enabled || c.TTS.Enabled || c.Image.Enabled || c.Context.WindowTokens != 1048576 {
		t.Fatalf("LLM/Extra profile changed: %+v", b)
	}
}

func TestRadixArkExtraUpgradePreservesBindingsAndLLM(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 34
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID == "flash-next-radixark" {
			b.Components = []string{"flash-next-radixark"}
			b.WorkloadSwap = false
		}
	}
	core, _ := c.Runtime.Catalog.ResolveComponent("flash-next-radixark", "flash-next-radixark")
	c.Normalize()
	c.Normalize()
	after, _ := c.Runtime.Catalog.ResolveComponent("flash-next-radixark", "flash-next-radixark")
	if !reflect.DeepEqual(core, after) {
		t.Fatal("Extra upgrade changed running LLM")
	}
	b, _ := c.Runtime.Catalog.Bundle("flash-next-radixark")
	if !b.WorkloadSwap || len(b.Components) != 7 {
		t.Fatalf("Extra lease/membership missing: %+v", b)
	}
}

func TestRadixArkASRUpgradeKeepsLLMAndUserPreferences(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 35
	c.ASR.Enabled = false
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID == "flash-next-radixark" {
			members := []string{}
			for _, id := range b.Components {
				if id != "nemotron-asr" {
					members = append(members, id)
				}
			}
			b.Components = members
			delete(b.Bindings, "nemotron-asr")
		}
	}
	core, _ := c.Runtime.Catalog.ResolveComponent("flash-next-radixark", "flash-next-radixark")
	c.Normalize()
	c.Normalize()
	after, _ := c.Runtime.Catalog.ResolveComponent("flash-next-radixark", "flash-next-radixark")
	asr, ok := c.Runtime.Catalog.ResolveComponent("flash-next-radixark", "nemotron-asr")
	if !reflect.DeepEqual(core, after) || !ok || !asr.StartAfterLLM || !asr.KeepResident || c.ASR.Enabled {
		t.Fatal("ASR migration changed LLM, user toggle or resident policy")
	}
	c.ASR.Enabled = true
	c.ApplyManagedBundle("flash-next-radixark")
	if !c.ASR.Enabled || c.TTS.Enabled || c.Image.Enabled || c.Context.WindowTokens != 1048576 {
		t.Fatal("wrong live feature profile")
	}
}
