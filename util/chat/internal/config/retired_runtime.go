package config

import "sparktalk/internal/orchestrator"

// Remove the retired built-in service from saved catalogs as well as defaults.
func (c *Config) removeRetiredRuntimes() {
	catalog := c.Runtime.Catalog
	if catalog == nil {
		return
	}
	removed := map[string]bool{"flash-next-exl3": true, "qwen27": true}
	components := make([]orchestrator.Component, 0, len(catalog.Components))
	for _, component := range catalog.Components {
		if removed[component.ID] || component.ComposeAsset == "compose.flash-next-exl3.yaml" || component.ComposeAsset == "compose.qwen27.yaml" {
			removed[component.ID] = true
		} else {
			components = append(components, component)
		}
	}
	removedBundles := map[string]bool{"flash-next-exl3": true, "qwen27": true}
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
			bundles = append(bundles, bundle)
		}
	}
	catalog.Components = components
	catalog.Bundles = bundles
	fallback := "flash-next"
	if len(bundles) > 0 {
		fallback = bundles[0].ID
		for _, b := range bundles {
			if b.ID == "flash-next" {
				fallback = b.ID
			}
		}
	}
	if removedBundles[c.Runtime.Bundle] {
		if c.Runtime.Bundle == "qwen27" {
			c.Runtime.Bundle = "qwen27-exl3"
		} else {
			c.Runtime.Bundle = fallback
		}
	}
	if removedBundles[c.Runtime.ActiveBundle] {
		if c.Runtime.ActiveBundle == "qwen27" {
			c.Runtime.ActiveBundle = "qwen27-exl3"
		} else {
			c.Runtime.ActiveBundle = fallback
		}
		c.Runtime.AutoStart = false
	}
}
