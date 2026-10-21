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
		"gemma": true, "gemma26": true, "gemma31": true, "ornith35": true,
		"qwen38fn_exl3_q4": true, "flash-next-radixark": true,
		"ds4fve": true, "ds4fve_exl3": true,
	}
	retiredAssets := map[string]bool{
		"compose.flash-next-exl3.yaml": true, "compose.qwen27.yaml": true,
		"compose.qwen27-gguf.yaml": true, "compose.qwen27-exl3.yaml": true,
		"compose.gemma.yaml": true, "compose.gemma26.yaml": true, "compose.gemma31.yaml": true, "compose.ornith35.yaml": true,
		"compose.qwen38fn_exl3_q4.yaml": true,
		"compose.ds4fve_exl3.yaml":      true,
	}
	components := make([]orchestrator.Component, 0, len(catalog.Components))
	for _, component := range catalog.Components {
		if removed[component.ID] || retiredAssets[component.ComposeAsset] || component.Controller == "dspark-cluster" {
			removed[component.ID] = true
		} else {
			components = append(components, component)
		}
	}
	removedBundles := map[string]bool{
		"flash-next-exl3": true, "qwen27": true, "qwen27-gguf": true, "qwen27-exl3": true,
		"gemma": true, "gemma26": true, "gemma31": true, "ornith35": true,
		"qwen38fn_exl3_q4": true, "flash-next-radixark": true,
		"ds4fve": true, "ds4fve_exl3": true,
	}
	bundles := make([]orchestrator.Bundle, 0, len(catalog.Bundles))
	for _, bundle := range catalog.Bundles {
		retired := removedBundles[bundle.ID] || bundle.ModelType == "deepseek-v4-exl3"
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
	preferences := []string{c.Runtime.ActiveBundle, c.Runtime.Bundle, "flash-next"}
	if removedBundles[c.Runtime.ActiveBundle] && (c.Runtime.ActiveBundle == "ds4fve" || c.Runtime.ActiveBundle == "ds4fve_exl3") {
		preferences = []string{c.Runtime.Bundle, "ds41", "flash-next"}
	}
	for _, preferred := range preferences {
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
		for id := range bundle.Bindings {
			if !available[id] {
				delete(bundle.Bindings, id)
			}
		}
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
