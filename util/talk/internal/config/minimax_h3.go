package config

import "sparktalk/internal/orchestrator"

// Replace the former EXL3 entry in place with its joint image/video runtime.
// Previously selected MMH3 sets keep their deployments and use the canonical ID.
func (c *Config) addMiniMaxH3Set() {
	if c.Runtime.Catalog == nil {
		return
	}
	const canonical = "qwen38fn_exl3"
	const previous = "qwen38fn_exl3-mmh3"
	original, paired := -1, -1
	for i, b := range c.Runtime.Catalog.Bundles {
		if b.ID == canonical {
			original = i
		}
		if b.ID == previous {
			paired = i
		}
	}
	if original < 0 && paired < 0 {
		return
	}
	defaults, err := orchestrator.LoadCatalog()
	if err != nil {
		return
	}
	have := map[string]bool{}
	for _, x := range c.Runtime.Catalog.Components {
		have[x.ID] = true
	}
	if !have["qwim-mmh3"] {
		x, ok := defaults.Component("qwim-mmh3")
		if !ok {
			return
		}
		c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, x)
		have[x.ID] = true
	}
	b, ok := defaults.Bundle(canonical)
	if !ok {
		return
	}
	slot := original
	if paired >= 0 {
		b = c.Runtime.Catalog.Bundles[paired]
		if slot < 0 {
			slot = paired
		}
	} else {
		description := b.Description
		b = c.Runtime.Catalog.Bundles[original]
		members := append([]string(nil), b.Components...)
		joint := false
		for i, member := range members {
			if member == "qwim-mmh3" {
				joint = true
			}
			if member == "flux2" {
				members[i] = "qwim-mmh3"
				joint = true
				b.Description = description
			}
		}
		if !joint {
			members = append(members, "qwim-mmh3")
			b.Description = description
		}
		b.Components = members
		delete(b.Bindings, "flux2")
	}
	b.ID, b.Name = canonical, "Qwen 3.8 Flash-Next EXL3"
	members := []string{}
	for _, id := range b.Components {
		if have[id] {
			members = append(members, id)
		} else {
			delete(b.Bindings, id)
		}
	}
	b.Components = members
	bundles := make([]orchestrator.Bundle, 0, len(c.Runtime.Catalog.Bundles))
	for i, item := range c.Runtime.Catalog.Bundles {
		if i == slot {
			bundles = append(bundles, b)
		}
		if item.ID != canonical && item.ID != previous {
			bundles = append(bundles, item)
		}
	}
	c.Runtime.Catalog.Bundles = bundles
	if c.Runtime.Bundle == previous {
		c.Runtime.Bundle = canonical
	}
	if c.Runtime.ActiveBundle == previous {
		c.Runtime.ActiveBundle = canonical
	}
}
