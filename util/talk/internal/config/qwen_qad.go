package config

import "sparktalk/internal/orchestrator"

// The QAD checkpoint is qualified for TP1. Never migrate the TP2 identity or a
// user-supplied model, endpoint, context limit, binding, or runtime option.
func (c *Config) migrateQwenQADTP1() {
	defaults, err := orchestrator.LoadCatalog()
	if err != nil || c.Runtime.Catalog == nil {
		return
	}
	const previous = "edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4"
	d, _ := defaults.Component("flash-next")
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID != "flash-next" || x.Model != previous {
			continue
		}
		x.Model = d.Model
		if x.Name == "Huihui-RadixArk Qwen3.8 Flash-Next" {
			x.Name = d.Name
		}
		if x.MemoryGiB == 96 {
			x.MemoryGiB = d.MemoryGiB
		}
		if x.StartupTimeoutSeconds == 1200 {
			x.StartupTimeoutSeconds = d.StartupTimeoutSeconds
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		x := &c.Runtime.Catalog.Bundles[i]
		if x.ID == "flash-next" && x.ModelID == previous {
			x.ModelID = d.Model
			if x.Description == "64K 문맥과 Flash-Next를 사용하는 고성능 세트" {
				b, _ := defaults.Bundle("flash-next")
				x.Description = b.Description
			}
		}
	}
	active := c.Runtime.ActiveBundle
	if active == "" {
		active = c.Runtime.Bundle
	}
	if c.Runtime.Mode == "managed" && (active == "flash-next" || active == "") && c.Model.DefaultModel == previous {
		c.Model.DefaultModel = d.Model
	}
}

// Upgrade the built-in TP1 profile without changing other bundles.
func (c *Config) migrateQwenTP1Context1M() {
	if c.Runtime.Catalog == nil {
		return
	}
	defaults, err := orchestrator.LoadCatalog()
	if err != nil {
		return
	}
	d, _ := defaults.Bundle("flash-next")
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "flash-next" || b.ModelID != d.ModelID {
			continue
		}
		if b.ContextTokens == 65536 {
			b.ContextTokens = d.ContextTokens
			b.Description = d.Description
		}
		kept := make([]string, 0, len(b.Components))
		for _, id := range b.Components {
			if id != "flux2" {
				kept = append(kept, id)
			}
		}
		b.Components = kept
		delete(b.Bindings, "flux2")
		if b.Bindings == nil {
			b.Bindings = map[string]orchestrator.Deployment{}
		}
		for _, id := range []string{"nemotron-asr", "magpie-tts"} {
			member := false
			for _, component := range b.Components {
				if component == id {
					member = true
				}
			}
			if !member {
				continue
			}
			binding := b.Bindings[id]
			enabled := true
			binding.StartAfterLLM = &enabled
			b.Bindings[id] = binding
		}
	}
}

// Add deferred FLUX to the built-in QAD set only when its service is available.
func (c *Config) migrateQwenQADFlux() {
	if c.Runtime.Catalog == nil {
		return
	}
	available := false
	for _, component := range c.Runtime.Catalog.Components {
		if component.ID == "flux2" {
			available = true
		}
	}
	if !available {
		return
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "flash-next" || b.ModelID != "local-inference-lab/Qwen3.8-Flash-Next-NVFP4" {
			continue
		}
		found := false
		for _, id := range b.Components {
			if id == "flux2" {
				found = true
			}
		}
		if !found {
			b.Components = append(b.Components, "flux2")
		}
		if b.Bindings == nil {
			b.Bindings = map[string]orchestrator.Deployment{}
		}
		binding := b.Bindings["flux2"]
		enabled := true
		binding.StartAfterLLM = &enabled
		b.Bindings["flux2"] = binding
		if b.Description == "1× DGX Spark · SGLang/B12X · QAD · FP8 KV · 1M 문맥 · ASR/TTS" {
			b.Description = "1× DGX Spark · SGLang/B12X · QAD · FP8 KV · 1M 문맥 · FLUX·ASR/TTS"
		}
	}
}
