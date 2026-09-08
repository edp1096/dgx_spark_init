package config

import "sparktalk/internal/orchestrator"

// Retired built-ins must not reappear when a saved catalog is loaded or imported.
func (c *Config) removeRetiredRuntimes() {
	catalog := c.Runtime.Catalog
	if catalog == nil {
		return
	}
	removed := map[string]bool{
		"flash-next-exl3": true, "qwen27": true, "qwen27-gguf": true, "qwen27-exl3": true,
	}
	retiredAssets := map[string]bool{
		"compose.flash-next-exl3.yaml": true, "compose.qwen27.yaml": true,
		"compose.qwen27-gguf.yaml": true, "compose.qwen27-exl3.yaml": true,
	}
	components := make([]orchestrator.Component, 0, len(catalog.Components))
	for _, component := range catalog.Components {
		if removed[component.ID] || retiredAssets[component.ComposeAsset] {
			removed[component.ID] = true
		} else {
			components = append(components, component)
		}
	}
	removedBundles := map[string]bool{
		"flash-next-exl3": true, "qwen27": true, "qwen27-gguf": true, "qwen27-exl3": true,
	}
	bundles := make([]orchestrator.Bundle, 0, len(catalog.Bundles))
	for _, bundle := range catalog.Bundles {
		retired := removedBundles[bundle.ID]
		for _, id := range bundle.Components {
			if removed[id] {
				retired = true
			}
		}
		if retired {
			removedBundles[bundle.ID] = true
		} else {
			for id := range bundle.Bindings {
				if removed[id] {
					delete(bundle.Bindings, id)
				}
			}
			bundles = append(bundles, bundle)
		}
	}
	catalog.Components, catalog.Bundles = components, bundles
	if !removedBundles[c.Runtime.Bundle] && !removedBundles[c.Runtime.ActiveBundle] {
		return
	}

	// Changing a retired selection must never automatically launch its replacement.
	c.Runtime.AutoStart = false
	fallback := ""
	for _, preferred := range []string{c.Runtime.ActiveBundle, c.Runtime.Bundle, "flash-next"} {
		for _, bundle := range bundles {
			if bundle.ID == preferred {
				fallback = preferred
				break
			}
		}
		if fallback != "" {
			break
		}
	}
	if fallback == "" && len(bundles) > 0 {
		fallback = bundles[0].ID
	}
	if fallback == "" {
		// A catalog containing only retired sets needs a valid replacement. Retain
		// existing service definitions and add only the missing Flash-Next model.
		defaults, _ := orchestrator.LoadCatalog()
		model, _ := defaults.Component("flash-next")
		bundle, _ := defaults.Bundle("flash-next")
		available := map[string]bool{}
		for _, component := range catalog.Components {
			available[component.ID] = true
		}
		if !available[model.ID] {
			catalog.Components = append(catalog.Components, model)
			available[model.ID] = true
			if catalog.Hosts == nil {
				catalog.Hosts = map[string]orchestrator.Host{}
			}
			if _, ok := catalog.Hosts[model.Host]; !ok {
				catalog.Hosts[model.Host] = defaults.Hosts[model.Host]
			}
		}
		members := []string{}
		for _, id := range bundle.Components {
			if available[id] {
				members = append(members, id)
			}
		}
		bundle.Components = members
		catalog.Bundles = append(catalog.Bundles, bundle)
		fallback = bundle.ID
	}
	if removedBundles[c.Runtime.Bundle] {
		c.Runtime.Bundle = fallback
	}
	if removedBundles[c.Runtime.ActiveBundle] {
		c.Runtime.ActiveBundle = fallback
	}
}
