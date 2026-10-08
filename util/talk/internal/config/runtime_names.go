package config

import "sparktalk/internal/orchestrator"

func (c *Config) normalizeRuntimeDisplayNames() {
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Components {
		item := &c.Runtime.Catalog.Components[i]
		item.Name = orchestrator.RuntimeDisplayName(item.ID, item.Name)
	}
	for i := range c.Runtime.Catalog.Bundles {
		item := &c.Runtime.Catalog.Bundles[i]
		item.Name = orchestrator.RuntimeDisplayName(item.ID, item.Name)
		for id, binding := range item.Bindings {
			if binding.Name != nil {
				name := orchestrator.RuntimeDisplayName(id, *binding.Name)
				binding.Name = &name
				item.Bindings[id] = binding
			}
		}
		if item.ID == "qwen38fn_exl3" && item.Description == "1× DGX Spark · Native ExLlamaV3 · 원본 3bit HQ h6 ng6 · Q8 KV · YaRN 1M · MTP 3 · 비전 · Qwen Image 2.1·ASR·TTS" {
			item.Description = "1× DGX Spark · ExLlamaV3 · EXL3 3bit HQ h6 ng6 · Q8 KV · YaRN 1M · MTP 3 · 비전 · Qwen Image 2.1·ASR·TTS"
		}
	}
}
