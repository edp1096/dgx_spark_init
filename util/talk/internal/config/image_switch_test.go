package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestImageCapabilityReturnsAfterSwitchingFromCodingSet(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	c := Config{Runtime: RuntimeConfig{Mode: "managed", Catalog: &cat, ActiveBundle: "qwen38fn_exl3", Bundle: "qwen38fn_exl3"}, Image: ImageConfig{Enabled: true}}
	c.Normalize()
	c.SelectManagedBundle("flash-next-fp4-local")
	if c.Image.Enabled {
		t.Fatal("image advertised in a set without an image model")
	}
	c.SelectManagedBundle("qwen38fn_exl3")
	if !c.Image.Enabled || c.Image.Mode != "reference" || c.Image.Model != "qwen-image-2.1-uc-nvfp4" {
		t.Fatalf("image capability not restored: %+v", c.Image)
	}
	c.Image.Enabled = false
	c.SelectManagedBundle("qwen38fn_exl3")
	if c.Image.Enabled {
		t.Fatal("same-set restart overwrote user switch")
	}
	c.Normalize()
	if c.Image.Enabled {
		t.Fatal("normalization overwrote user switch")
	}
}

func TestSetSwitchDefaultsMatchAllMountedFeatures(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	c := Config{Runtime: RuntimeConfig{Mode: "managed", Catalog: &cat, ActiveBundle: "flash-next-fp4-local", Bundle: "flash-next-fp4-local"}}
	c.Normalize()
	for _, id := range []string{"qwen38fn_exl3", "flash-next-tp2", "ds41", "glm53-worker-extra", "flash-next-fp4-local", "qwen38fn_exl3"} {
		c.SelectManagedBundle(id)
		roles := map[string]bool{}
		for _, x := range c.Runtime.Catalog.BundleComponents(id) {
			roles[x.ServiceRole()] = true
		}
		for role, enabled := range map[string]bool{"asr": c.ASR.Enabled, "tts": c.TTS.Enabled, "image": c.Image.Enabled, "embedding": c.Embedding.Enabled} {
			if enabled != roles[role] {
				t.Fatalf("%s %s enabled=%v, mounted=%v", id, role, enabled, roles[role])
			}
		}
		if id != "qwen38fn_exl3" && (c.SupportEnabled("fish") || c.SupportEnabled("music-ace")) {
			t.Fatal("excluded audio re-enabled", id)
		}
	}
	c.ASR.Enabled = false
	c.TTS.Enabled = false
	c.Image.Enabled = false
	c.Embedding.Enabled = false
	c.SelectManagedBundle("qwen38fn_exl3")
	if c.ASR.Enabled || c.TTS.Enabled || c.Image.Enabled || c.Embedding.Enabled {
		t.Fatal("same-set user preferences lost")
	}
}
