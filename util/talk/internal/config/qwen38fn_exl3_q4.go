package config

import "sparktalk/internal/orchestrator"

// Offer the qualified Q4 coding set once without changing selections or overrides.
func (c *Config) addQwen38FNEXL3Q4Set() {
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
	const id = "qwen38fn_exl3_q4"
	members := map[string]orchestrator.Component{}
	for _, x := range c.Runtime.Catalog.Components {
		members[x.ID] = x
	}
	if _, ok := members[id]; !ok {
		x, ok := defaults.Component(id)
		if !ok {
			return
		}
		c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, x)
		members[id] = x
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
	kept := []string{}
	for _, member := range b.Components {
		x, ok := members[member]
		if !ok {
			delete(b.Bindings, member)
			continue
		}
		kept = append(kept, member)
		binding := b.Bindings[member]
		x = binding.Apply(x)
		if (x.KeepResident || x.Role == "llm") && (x.Controller != "compose" || (x.Host != "" && x.Host != "local")) {
			binding.KeepResident, binding.MemoryGiB = nil, nil
			b.Bindings[member] = binding
			b.WorkloadSwap = false
		}
	}
	b.Components = kept
	index := len(c.Runtime.Catalog.Bundles)
	for i, x := range c.Runtime.Catalog.Bundles {
		if x.ID == "flash-next-radixark" {
			index = i
			break
		}
	}
	for i, x := range c.Runtime.Catalog.Bundles {
		if x.ID == "qwen38fn_exl3" {
			index = i + 1
			break
		}
	}
	c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, orchestrator.Bundle{})
	copy(c.Runtime.Catalog.Bundles[index+1:], c.Runtime.Catalog.Bundles[index:])
	c.Runtime.Catalog.Bundles[index] = b
}
