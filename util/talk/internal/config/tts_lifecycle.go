package config

import "sparktalk/internal/orchestrator"

// The single-Spark QAD recipe includes on-demand TTS. Keep enable/voice choices
// and custom placements; the builtin TTS default already executes locally.
func (c *Config) migrateManagedTTS() {
	if c.Runtime.Catalog == nil {
		return
	}
	nativePresent := false
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ComposeAsset == "compose.qwen3-tts.yaml" && x.Model == "qwen3-tts-0.6b-q8" {
			nativePresent = nativePresent || (x.ID == "qwen3-tts" && (x.Host == "" || x.Host == "local") && (x.Controller == "" || x.Controller == "compose"))
			if x.MemoryGiB < 4 {
				x.MemoryGiB = 4
			}
			if x.StartupMemoryGiB == 0 {
				x.StartupMemoryGiB = 3
			}
			if x.WorkspaceMemoryGiB == 0 {
				x.WorkspaceMemoryGiB = 1.5
			}
		}
	}
	if !nativePresent {
		return
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "flash-next" || !b.WorkloadSwap {
			continue
		}
		exists := false
		for _, id := range b.Components {
			exists = exists || id == "qwen3-tts"
		}
		if exists {
			continue
		}
		b.Components = append(b.Components, "qwen3-tts")
		if b.Bindings == nil {
			b.Bindings = map[string]orchestrator.Deployment{}
		}
		deferred := true
		b.Bindings["qwen3-tts"] = orchestrator.Deployment{StartAfterLLM: &deferred}
	}
}
