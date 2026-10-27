package config

import "sparktalk/internal/orchestrator"

// Remove the retired music service from saved catalogs without retiring the
// user's model set or changing its active LLM.
func (c *Config) removeRetiredMusicSupport() {
	catalog := c.Runtime.Catalog
	if catalog == nil {
		return
	}
	removed := map[string]bool{"extra-music-heart": true}
	components := make([]orchestrator.Component, 0, len(catalog.Components))
	for _, x := range catalog.Components {
		if removed[x.ID] || x.ComposeAsset == "compose.extra-music-heart.yaml" {
			removed[x.ID] = true
			continue
		}
		components = append(components, x)
	}
	catalog.Components = components
	for i := range catalog.Bundles {
		b := &catalog.Bundles[i]
		keep := make([]string, 0, len(b.Components))
		for _, id := range b.Components {
			if !removed[id] {
				keep = append(keep, id)
			}
		}
		b.Components = keep
		for id := range b.Bindings {
			if removed[id] {
				delete(b.Bindings, id)
			}
		}
	}
}
