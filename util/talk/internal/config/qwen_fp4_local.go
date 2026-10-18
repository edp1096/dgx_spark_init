package config

import "sparktalk/internal/orchestrator"

// Add a separate set without replacing the selected QAD/EXL3/RadixArk profile,
// its startup selection, or user-edited deployments.
func (c *Config) addQwenFP4LocalSet() {
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
	const id = "flash-next-fp4-local"
	known := map[string]bool{}
	for _, component := range c.Runtime.Catalog.Components {
		known[component.ID] = true
	}
	if !known[id] {
		component, ok := defaults.Component(id)
		if !ok {
			return
		}
		c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, component)
		known[id] = true
	}
	for _, bundle := range c.Runtime.Catalog.Bundles {
		if bundle.ID == id {
			return
		}
	}
	bundle, ok := defaults.Bundle(id)
	if !ok {
		return
	}
	var members []string
	for _, member := range bundle.Components {
		if known[member] {
			members = append(members, member)
		} else {
			delete(bundle.Bindings, member)
		}
	}
	bundle.Components = members
	index := len(c.Runtime.Catalog.Bundles)
	for i, existing := range c.Runtime.Catalog.Bundles {
		if existing.ID == "qwen38fn_exl3_q4" {
			index = i + 1
		}
		if existing.ID == "flash-next-radixark" {
			index = i
			break
		}
	}
	c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, orchestrator.Bundle{})
	copy(c.Runtime.Catalog.Bundles[index+1:], c.Runtime.Catalog.Bundles[index:])
	c.Runtime.Catalog.Bundles[index] = bundle
}
