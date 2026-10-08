package config

import "sparktalk/internal/orchestrator"

func (c *Config) keepQwen38FNEXL3AuxiliariesResident() {
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Bundles {
		bundle := &c.Runtime.Catalog.Bundles[i]
		if bundle.ID != "qwen38fn_exl3" {
			continue
		}
		if bundle.Bindings == nil {
			bundle.Bindings = map[string]orchestrator.Deployment{}
		}
		for _, id := range bundle.Components {
			if id != "flux2" && id != "nemotron-asr" && id != "qwen3-tts" {
				continue
			}
			binding := bundle.Bindings[id]
			var component orchestrator.Component
			for _, base := range c.Runtime.Catalog.Components {
				if base.ID == id {
					component = binding.Apply(base)
					break
				}
			}
			if component.Controller != "compose" || (component.Host != "" && component.Host != "local") {
				continue
			}
			if binding.KeepResident == nil {
				keep := true
				binding.KeepResident = &keep
			}
			if id == "flux2" && *binding.KeepResident && binding.MemoryGiB == nil {
				budget := max(24.0, component.MemoryGiB)
				binding.MemoryGiB = &budget
			}
			bundle.Bindings[id] = binding
		}
	}
}

// Offer the qualified native runtime once, preserving edits and live selection.
func (c *Config) addQwen38FNEXL3Set() {
	if c.Runtime.Catalog == nil {
		return
	}
	if _, local := c.Runtime.Catalog.Hosts["local"]; !local {
		return
	}
	defaults, err := orchestrator.LoadCatalog()
	if err != nil {
		return
	}
	const id = "qwen38fn_exl3"
	components := map[string]bool{}
	for _, item := range c.Runtime.Catalog.Components {
		components[item.ID] = true
	}
	if !components[id] {
		item, ok := defaults.Component(id)
		if !ok {
			return
		}
		c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, item)
		components[id] = true
	}
	for _, item := range c.Runtime.Catalog.Bundles {
		if item.ID == id {
			return
		}
	}
	item, ok := defaults.Bundle(id)
	if !ok {
		return
	}
	// User-removed auxiliary services stay removed from a custom catalog.
	members := []string{}
	for _, member := range item.Components {
		if components[member] {
			members = append(members, member)
			for _, base := range c.Runtime.Catalog.Components {
				if base.ID == member && (base.Role == "image" || item.Bindings[member].KeepResident != nil) && (base.Controller != "compose" || (base.Host != "" && base.Host != "local")) {
					binding := item.Bindings[member]
					binding.KeepResident = nil
					binding.MemoryGiB = nil
					item.Bindings[member] = binding
					item.WorkloadSwap = false
				}
			}
		} else {
			delete(item.Bindings, member)
		}
	}
	item.Components = members
	c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, item)
}
