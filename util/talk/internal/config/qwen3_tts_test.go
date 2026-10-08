package config

import (
	"reflect"
	"testing"

	"sparktalk/internal/orchestrator"
)

func TestQwenTTSMigrationReplacesLegacyAndPreservesUserState(t *testing.T) {
	cat, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for i := range cat.Components {
		x := &cat.Components[i]
		if x.ID == "qwen3-tts" {
			x.ID, x.Model, x.ComposeAsset = "magpie-tts", "magpietts", "compose.magpie-tts.yaml"
			x.Name, x.Container = "Magpie TTS", "sparktalk-magpie-tts"
			x.MemoryGiB, x.StartupMemoryGiB = 3, 1.5
		}
	}
	for i := range cat.Bundles {
		b := &cat.Bundles[i]
		for j, id := range b.Components {
			if id == "qwen3-tts" {
				b.Components[j] = "magpie-tts"
			}
		}
		if binding, ok := b.Bindings["qwen3-tts"]; ok {
			b.Bindings["magpie-tts"] = binding
			delete(b.Bindings, "qwen3-tts")
		}
		if b.ID == "glm53-worker-extra" {
			members := []string{}
			for _, id := range b.Components {
				if id != "magpie-tts" {
					members = append(members, id)
				}
			}
			b.Components = members
			delete(b.Bindings, "magpie-tts")
		}
		if b.ID == "ds41" {
			host, endpoint, mem := "worker", "http://192.168.100.60:8792", 8.0
			binding := b.Bindings["magpie-tts"]
			binding.Host, binding.Endpoint, binding.MemoryGiB = &host, &endpoint, &mem
			b.Bindings["magpie-tts"] = binding
		}
	}
	cfg := Config{Runtime: RuntimeConfig{BuiltinRevision: 32, Catalog: &cat, Bundle: "flash-next", ActiveBundle: "qwen38fn_exl3"}, TTS: TTSConfig{Enabled: false, AutoPlay: true, Model: "magpietts", Voice: "Aria", SampleRate: 22050}}
	cfg.Normalize()
	if cfg.TTS.Enabled || !cfg.TTS.AutoPlay || cfg.TTS.Voice != "sohee" || cfg.TTS.SampleRate != 24000 || cfg.TTS.Model != "qwen3-tts-0.6b-q8" || cfg.Runtime.ActiveBundle != "qwen38fn_exl3" {
		t.Fatalf("migration lost selected set/toggles or kept old voice: %+v", cfg.TTS)
	}
	for _, x := range cfg.Runtime.Catalog.Components {
		if x.ID == "magpie-tts" || x.ComposeAsset == "compose.magpie-tts.yaml" {
			t.Fatal("retired engine retained", x)
		}
	}
	for _, b := range cfg.Runtime.Catalog.Bundles {
		if b.ID == "flash-next-radixark" {
			continue
		} // Explicit LLM-only set.
		x, ok := cfg.Runtime.Catalog.ResolveComponent(b.ID, "qwen3-tts")
		if !ok || x.MemoryGiB < 4 {
			t.Fatal("set missing Qwen/budget", b.ID, x)
		}
	}
	x, _ := cfg.Runtime.Catalog.ResolveComponent("ds41", "qwen3-tts")
	if x.Host != "worker" || x.Endpoint != "http://192.168.100.60:8792" || x.MemoryGiB != 8 {
		t.Fatal("custom placement/budget lost", x)
	}
	before := cfg.Public()
	cfg.Normalize()
	if !reflect.DeepEqual(before, cfg.Public()) {
		t.Fatal("migration is not idempotent")
	}
}

func TestBuiltinSpeechSetsHaveQwenTTSAndSwitchPreservesToggle(t *testing.T) {
	cfg := Config{}
	cfg.Normalize()
	cfg.TTS.Enabled = true
	for _, b := range cfg.Runtime.Catalog.Bundles {
		if b.ID == "flash-next-radixark" {
			continue
		} // Tested separately as LLM-only.
		cfg.ApplyManagedBundle(b.ID)
		if !cfg.TTS.Enabled || cfg.TTS.Model != "qwen3-tts-0.6b-q8" {
			t.Fatal("TTS unavailable in set", b.ID, cfg.TTS)
		}
	}
	cfg.TTS.Enabled = false
	cfg.ApplyManagedBundle("glm53-worker-extra")
	if cfg.TTS.Enabled {
		t.Fatal("disabled TTS was enabled by a set switch")
	}
}
