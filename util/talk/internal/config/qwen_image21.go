package config

import (
	"strings"

	"sparktalk/internal/orchestrator"
)

// Keep the image slot ID and existing bundle/host bindings. Only the built-in
// FLUX recipe migrates; custom image endpoints and checkpoints remain explicit.
func (c *Config) migrateQwenImage21() {
	if c.Runtime.Catalog == nil {
		return
	}
	defaults, err := orchestrator.LoadCatalog()
	if err != nil {
		return
	}
	next, _ := defaults.Component("flux2")
	migrated := false
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID != "flux2" || x.ComposeAsset != "compose.flux2.yaml" || x.Model != "flux2-klein-4b-nvfp4" {
			continue
		}
		migrated = true
		x.ComposeAsset, x.Container, x.Model, x.ProgressKind = next.ComposeAsset, next.Container, next.Model, next.ProgressKind
		if x.Name == "FLUX.2 Klein 4B" {
			x.Name = next.Name
		}
		x.MemoryGiB = max(x.MemoryGiB, next.MemoryGiB)
		x.WorkspaceMemoryGiB = max(x.WorkspaceMemoryGiB, next.WorkspaceMemoryGiB)
		if x.StartupMemoryGiB == 0 || x.StartupMemoryGiB == 3.75 {
			x.StartupMemoryGiB = next.StartupMemoryGiB
		}
		if x.StartupTimeoutSeconds == 300 {
			x.StartupTimeoutSeconds = next.StartupTimeoutSeconds
		}
		for j := range c.Runtime.Catalog.Bundles {
			b := &c.Runtime.Catalog.Bundles[j]
			for _, member := range b.Components {
				if member == x.ID {
					b.Description = strings.ReplaceAll(b.Description, "FLUX", "Qwen Image 2.1")
					break
				}
			}
		}
	}
	if migrated && c.Runtime.Mode != "external" && c.Image.Model == "flux2-klein-4b-nvfp4" {
		c.Image.Model = "qwen-image-2.1-uc-nvfp4"
		c.Image.Mode = "qwen-image21"
	}
}

// Upgrade the former abbreviated deployment names while preserving user values.
// The old mode token remains readable so saved settings continue to work.
func (c *Config) normalizeQwenImageNaming() {
	if c.Runtime.Catalog == nil {
		return
	}
	rename := func(value string) string {
		value = strings.ReplaceAll(value, "QWIM 2.1", "Qwen Image 2.1")
		return strings.ReplaceAll(value, "QWIM", "Qwen Image 2.1")
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ComposeAsset == "compose.qwim21.yaml" {
			x.ComposeAsset = "compose.qwen-image21.yaml"
		}
		if x.ComposeAsset == "compose.qwen-image21.yaml" {
			x.Name = rename(x.Name)
			if x.Container == "sparktalk-qwim21" {
				x.Container = "sparktalk-qwen-image21"
			}
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		b.Description = rename(b.Description)
		if d, ok := b.Bindings["flux2"]; ok {
			if d.Name != nil {
				next := rename(*d.Name)
				d.Name = &next
			}
			if d.Container != nil && *d.Container == "sparktalk-qwim21" {
				next := "sparktalk-qwen-image21"
				d.Container = &next
			}
			b.Bindings["flux2"] = d
		}
	}
}
