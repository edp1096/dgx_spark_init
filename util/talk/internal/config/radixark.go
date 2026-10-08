package config

import "sparktalk/internal/orchestrator"

func (c *Config) addRadixArkSet() {
	if c.Runtime.Catalog == nil {
		return
	}
	if _, ok := c.Runtime.Catalog.Hosts["local"]; !ok {
		return
	}
	defaults, err := orchestrator.LoadCatalog()
	if err != nil {
		return
	}
	const id = "flash-next-radixark"
	known := map[string]bool{}
	for _, x := range c.Runtime.Catalog.Components {
		known[x.ID] = true
	}
	if !known[id] {
		x, ok := defaults.Component(id)
		if !ok {
			return
		}
		c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, x)
		known[id] = true
	}
	for _, b := range c.Runtime.Catalog.Bundles {
		if b.ID == id {
			return
		}
	}
	b, ok := defaults.Bundle(id)
	if !ok {
		return
	}
	var members []string
	for _, member := range b.Components {
		if known[member] {
			members = append(members, member)
		} else {
			delete(b.Bindings, member)
		}
	}
	b.Components = members
	index := len(c.Runtime.Catalog.Bundles)
	for i, existing := range c.Runtime.Catalog.Bundles {
		if existing.ID == "qwen38fn_exl3" {
			index = i + 1
			break
		}
	}
	c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, orchestrator.Bundle{})
	copy(c.Runtime.Catalog.Bundles[index+1:], c.Runtime.Catalog.Bundles[index:])
	c.Runtime.Catalog.Bundles[index] = b
}

// Extra workers are available on demand without loading image or TTS models.
// Preserve saved bindings and omitted global services in custom catalogs.
func (c *Config) enableRadixArkExtras() {
	if c.Runtime.Catalog == nil {
		return
	}
	known := map[string]bool{}
	for _, x := range c.Runtime.Catalog.Components {
		known[x.ID] = x.IsSupport()
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "flash-next-radixark" {
			continue
		}
		seen := map[string]bool{}
		for _, id := range b.Components {
			seen[id] = true
		}
		for _, id := range []string{"extra-media", "extra-ssh", "extra-collector", "extra-documents"} {
			if known[id] && !seen[id] {
				b.Components = append(b.Components, id)
			}
		}
		b.WorkloadSwap = true
		if b.Description == "1× DGX Spark · RadixArk NVFP4 · FP8 KV 1M · MTP 3 · LLM 단독" {
			b.Description = "1× DGX Spark · RadixArk NVFP4 · FP8 KV 1M · MTP 3 · Extra"
		}
	}
}

// Restore transcription as an on-demand workload; do not reload the LLM or
// change global ASR preferences for users who disabled it explicitly.
func (c *Config) enableRadixArkASR() {
	if c.Runtime.Catalog == nil {
		return
	}
	known := false
	for _, x := range c.Runtime.Catalog.Components {
		known = known || x.ID == "nemotron-asr"
	}
	if !known {
		return
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "flash-next-radixark" {
			continue
		}
		present := false
		for _, id := range b.Components {
			present = present || id == "nemotron-asr"
		}
		if !present {
			b.Components = append(b.Components, "nemotron-asr")
		}
		if b.Bindings == nil {
			b.Bindings = map[string]orchestrator.Deployment{}
		}
		if _, exists := b.Bindings["nemotron-asr"]; !exists {
			deferred := true
			b.Bindings["nemotron-asr"] = orchestrator.Deployment{StartAfterLLM: &deferred}
		}
		b.WorkloadSwap = true
		if b.Description == "1× DGX Spark · RadixArk NVFP4 · FP8 KV 1M · MTP 3 · Extra" {
			b.Description = "1× DGX Spark · RadixArk NVFP4 · FP8 KV 1M · MTP 3 · ASR·Extra"
		}
	}
}
