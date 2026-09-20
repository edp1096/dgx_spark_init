package config

import (
	"reflect"
	"testing"
)

func TestQADMigrationIsTP1OnlyAndPreservesCustomization(t *testing.T) {
	const old = "edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4"
	const next = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
	for _, active := range []string{"flash-next", "flash-next-tp2"} {
		t.Run(active, func(t *testing.T) {
			c := Config{}
			c.Normalize()
			c.Runtime.BuiltinRevision = 11
			c.Runtime.ActiveBundle = active
			c.Model.DefaultModel = old
			for i := range c.Runtime.Catalog.Components {
				x := &c.Runtime.Catalog.Components[i]
				if x.ID == "flash-next" {
					x.Model, x.Name, x.Endpoint = old, "My QAD", "http://custom:8000"
					x.RuntimeOptions = map[string]string{"DRAFT_VOCAB": "off"}
				}
			}
			for i := range c.Runtime.Catalog.Bundles {
				x := &c.Runtime.Catalog.Bundles[i]
				if x.ID == "flash-next" {
					x.ModelID, x.ContextTokens = old, 32768
				}
			}
			before, _ := c.Runtime.Catalog.Component("flash-next-tp2")
			c.Normalize()
			c.Normalize()
			x, _ := c.Runtime.Catalog.Component("flash-next")
			if x.Model != next || x.Name != "My QAD" || x.Endpoint != "http://custom:8000" || x.RuntimeOptions["DRAFT_VOCAB"] != "off" {
				t.Fatalf("TP1 customization lost: %+v", x)
			}
			b, _ := c.Runtime.Catalog.Bundle("flash-next")
			if b.ModelID != next || b.ContextTokens != 32768 {
				t.Fatalf("bundle customization lost: %+v", b)
			}
			after, _ := c.Runtime.Catalog.Component("flash-next-tp2")
			if !reflect.DeepEqual(before, after) {
				t.Fatal("TP2 changed")
			}
			want := old
			if active == "flash-next" {
				want = next
			}
			if c.Model.DefaultModel != want {
				t.Fatalf("active %s model = %q, want %q", active, c.Model.DefaultModel, want)
			}
		})
	}
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 11
	for i := range c.Runtime.Catalog.Components {
		if c.Runtime.Catalog.Components[i].ID == "flash-next" {
			c.Runtime.Catalog.Components[i].Model = "user/custom"
		}
	}
	c.Normalize()
	x, _ := c.Runtime.Catalog.Component("flash-next")
	if x.Model != "user/custom" {
		t.Fatal("custom model replaced")
	}
}

func TestQADContext1MMigrationKeepsDeferredServices(t *testing.T) {
	c := Config{}
	c.Normalize()
	c.Runtime.BuiltinRevision = 12
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID == "flash-next" {
			b.ContextTokens = 65536
			b.Components = []string{"flash-next", "nemotron-asr", "magpie-tts", "extra-media", "extra-ssh", "extra-collector"}
		}
	}
	before, _ := c.Runtime.Catalog.Bundle("flash-next-tp2")
	c.Normalize()
	c.Normalize()
	b, _ := c.Runtime.Catalog.Bundle("flash-next")
	if b.ContextTokens != 1048576 {
		t.Fatal(b.ContextTokens)
	}
	found := map[string]bool{}
	for _, id := range b.Components {
		found[id] = true
	}
	if !found["flux2"] || !found["nemotron-asr"] || !found["magpie-tts"] || !found["extra-media"] {
		t.Fatal(b.Components)
	}
	for _, id := range []string{"flux2", "nemotron-asr", "magpie-tts"} {
		p := b.Bindings[id].StartAfterLLM
		if p == nil || !*p {
			t.Fatal(id)
		}
	}
	after, _ := c.Runtime.Catalog.Bundle("flash-next-tp2")
	if !reflect.DeepEqual(before, after) {
		t.Fatal("TP2 changed")
	}
}
