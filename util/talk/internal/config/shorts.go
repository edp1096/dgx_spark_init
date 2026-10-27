package config

import "sparktalk/internal/orchestrator"

// Add optional services without replacing custom deployments or set membership.
func (c *Config) ensureShortsSupport() {
	defaults, err := orchestrator.LoadCatalog()
	if err != nil || c.Runtime.Catalog == nil {
		return
	}
	for _, id := range []string{"extra-shorts", "extra-fish", "extra-music-ace"} {
		found := false
		for i, component := range c.Runtime.Catalog.Components {
			found = found || component.ID == id
			if id == "extra-fish" && component.ID == id && component.ComposeAsset == "compose.extra-fish.yaml" && component.Role == "tool" {
				c.Runtime.Catalog.Components[i].Role = "tts"
			}
		}
		if !found {
			if component, ok := defaults.Component(id); ok {
				c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, component)
			}
		}
	}
}

// Keep the qualified coding/TP2 memory profiles free of shared audio workers.
// Shared services otherwise appear even when they are absent from set members.
func (c *Config) restrictConstrainedAudioSupport() {
	if c.Runtime.Catalog == nil {
		return
	}
	constrained := map[string]bool{}
	for _, x := range c.Runtime.Catalog.Components {
		if x.Controller == "external" || x.Role != "llm" {
			continue
		}
		if x.ComposeAsset == "compose.qwen-fp4-local.yaml" || x.WorkerHost != "" {
			constrained[x.ID] = true
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		restrict := false
		for _, id := range b.Components {
			restrict = restrict || constrained[id]
		}
		if !restrict {
			continue
		}
		for _, id := range []string{"extra-fish", "extra-music-ace"} {
			found := false
			for _, value := range b.DisabledSupport {
				found = found || value == id
			}
			if !found {
				b.DisabledSupport = append(b.DisabledSupport, id)
			}
		}
	}
}
