package config

import "sparktalk/internal/orchestrator"

// Preserve the user's global preference across switches, but never call a
// local embedding model that the selected managed set does not contain.
func (c Config) SemanticSearchEnabled() bool {
	if !c.Embedding.Enabled {
		return false
	}
	if c.Runtime.Mode != "managed" {
		return true
	}
	catalog := c.Runtime.Catalog
	if catalog == nil {
		defaults, err := orchestrator.LoadCatalog()
		if err != nil {
			return false
		}
		catalog = &defaults
	}
	bundle := c.Runtime.ActiveBundle
	if bundle == "" {
		bundle = c.Runtime.Bundle
	}
	// Membership is sufficient; do not rely on a validated lookup map here.
	for _, b := range catalog.Bundles {
		if b.ID == bundle {
			for _, id := range b.Components {
				if id == "extra-embedding" {
					return true
				}
			}
		}
	}
	return false
}

func (c *Config) migrateResidentEmbedding() {
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "extra-embedding" && x.ComposeAsset == "compose.extra-embedding.yaml" && x.Controller != "external" && (x.Host == "" || x.Host == "local") {
			x.Name = "EmbeddingGemma 2"
			x.Model = "google/embeddinggemma-2"
			x.KeepResident, x.StartAfterLLM = true, true
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "qwen38fn_exl3" && b.ID != "flash-next-radixark" {
			continue
		}
		found := false
		for _, id := range b.Components {
			found = found || id == "extra-embedding"
		}
		if !found {
			b.Components = append(b.Components, "extra-embedding")
		}
	}
}

func (c *Config) migrateNVFP4ResidentASR() {
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "flash-next-radixark" && x.MemoryGiB == 110 && x.WorkspaceMemoryGiB == 0 {
			x.WorkspaceMemoryGiB = 8
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "flash-next-radixark" {
			continue
		}
		member := false
		for _, id := range b.Components {
			member = member || id == "nemotron-asr"
		}
		if !member {
			continue
		}
		if b.Bindings == nil {
			b.Bindings = map[string]orchestrator.Deployment{}
		}
		binding := b.Bindings["nemotron-asr"]
		if (binding.Host != nil && *binding.Host != "local" && *binding.Host != "") || (binding.Controller != nil && *binding.Controller != "compose") {
			continue
		}
		keep := true
		binding.KeepResident, binding.StartAfterLLM = &keep, &keep
		b.Bindings["nemotron-asr"] = binding
	}
}
