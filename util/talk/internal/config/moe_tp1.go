package config

import "sparktalk/internal/orchestrator"

// Add new built-ins without replacing user-edited components or active selection.
func (c *Config) addSingleSparkMoEModels() {
	defaults, err := orchestrator.LoadCatalog()
	if err != nil || c.Runtime.Catalog == nil {
		return
	}
	if _, ok := c.Runtime.Catalog.Hosts["local"]; !ok {
		return
	}
	components := map[string]bool{}
	bundles := map[string]bool{}
	for _, item := range c.Runtime.Catalog.Components {
		components[item.ID] = true
	}
	for _, item := range c.Runtime.Catalog.Bundles {
		bundles[item.ID] = true
	}
	for _, id := range []string{"ornith35", "gemma26"} {
		if !components[id] {
			if item, ok := defaults.Component(id); ok {
				c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, item)
				components[id] = true
			}
		}
		if !bundles[id] {
			if item, ok := defaults.Bundle(id); ok {
				complete := true
				for _, member := range item.Components {
					complete = complete && components[member]
				}
				if complete {
					c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, item)
				}
			}
		}
	}
}
