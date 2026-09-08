package config

import (
	"sparktalk/internal/orchestrator"
	"strconv"
)

// Remove the retired built-in service from saved catalogs as well as defaults.
func (c *Config) removeRetiredRuntimes() {
	catalog := c.Runtime.Catalog
	if catalog == nil {
		return
	}
	inheritedContext := ""
	for _, bundle := range catalog.Bundles {
		if bundle.ID != c.Runtime.ActiveBundle && bundle.ID != c.Runtime.Bundle {
			continue
		}
		for _, component := range catalog.Components {
			if component.ComposeAsset != "compose.qwen27-gguf.yaml" {
				continue
			}
			for _, id := range bundle.Components {
				if id != component.ID {
					continue
				}
				resolved := bundle.Bindings[id].Apply(component)
				value := resolved.RuntimeOptions["MAX_MODEL_LEN"]
				if value == "" && bundle.ContextTokens > 0 {
					value = strconv.Itoa(bundle.ContextTokens)
				}
				if n, e := strconv.Atoi(value); e == nil && n >= 32768 && n <= 262144 {
					inheritedContext = value
				}
			}
		}
	}
	removed := map[string]bool{"flash-next-exl3": true, "qwen27": true, "qwen27-gguf": true}
	components := make([]orchestrator.Component, 0, len(catalog.Components))
	for _, component := range catalog.Components {
		if removed[component.ID] || component.ComposeAsset == "compose.flash-next-exl3.yaml" || component.ComposeAsset == "compose.qwen27.yaml" || component.ComposeAsset == "compose.qwen27-gguf.yaml" {
			removed[component.ID] = true
		} else {
			components = append(components, component)
		}
	}
	removedBundles := map[string]bool{"flash-next-exl3": true, "qwen27": true, "qwen27-gguf": true}
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
	defaults, _ := orchestrator.LoadCatalog()
	hasComponent, hasBundle := false, false
	for _, item := range components {
		if item.ID == "qwen27-exl3" {
			hasComponent = true
		}
	}
	for _, item := range bundles {
		if item.ID == "qwen27-exl3" {
			hasBundle = true
		}
	}
	if !hasComponent {
		item, _ := defaults.Component("qwen27-exl3")
		if inheritedContext != "" {
			item.RuntimeOptions = map[string]string{"MAX_MODEL_LEN": inheritedContext}
		}
		components = append(components, item)
	}
	if !hasBundle {
		item, _ := defaults.Bundle("qwen27-exl3")
		// Respect catalogs that removed optional services.
		available := map[string]bool{}
		for _, component := range components {
			available[component.ID] = true
		}
		members := []string{}
		for _, id := range item.Components {
			if available[id] {
				members = append(members, id)
			}
		}
		item.Components = members
		bundles = append(bundles, item)
	}
	catalog.Components = components
	catalog.Bundles = bundles
	fallback := "qwen27-exl3"
	if len(bundles) > 0 {
		fallback = bundles[0].ID
		for _, b := range bundles {
			if b.ID == "qwen27-exl3" {
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
