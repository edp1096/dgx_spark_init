package config

import (
	"strings"

	"sparktalk/internal/orchestrator"
)

// Recognize saved/imported legacy settings even after the builtin revision
// advanced. There is no legacy inference backend in the application.
func (c *Config) hasLegacyTTS() bool {
	if strings.TrimSpace(c.TTS.Model) == "magpietts" {
		return true
	}
	if c.Runtime.Catalog != nil {
		for _, x := range c.Runtime.Catalog.Components {
			if x.ID == "magpie-tts" || x.ComposeAsset == "compose.magpie-tts.yaml" || x.Model == "magpietts" {
				return true
			}
		}
	}
	return false
}

// Catalog imports are migrated before validating recipe existence. This does
// not start a service or modify the application's current configuration.
func MigrateLegacyTTS(catalog *orchestrator.Catalog) {
	c := Config{Runtime: RuntimeConfig{Catalog: catalog}}
	if c.hasLegacyTTS() {
		c.migrateQwenTTS()
	}
}

func (c *Config) migrateQwenTTS() {
	defaults, err := orchestrator.LoadCatalog()
	if err != nil || c.Runtime.Catalog == nil {
		return
	}
	canonical, _ := defaults.Component("qwen3-tts")
	catalog := c.Runtime.Catalog
	ids := map[string]string{"magpie-tts": "qwen3-tts"}
	present := false
	for _, x := range catalog.Components {
		present = present || (x.ID == canonical.ID && x.ComposeAsset != "compose.magpie-tts.yaml" && x.Model != "magpietts")
	}
	components := make([]orchestrator.Component, 0, len(catalog.Components)+1)
	for _, x := range catalog.Components {
		legacy := x.ID == "magpie-tts" || x.ComposeAsset == "compose.magpie-tts.yaml" || x.Model == "magpietts"
		if !legacy {
			components = append(components, x)
			continue
		}
		newID := canonical.ID
		ids[x.ID] = newID
		if present {
			continue
		}
		present = true
		x.ID, x.Model, x.ComposeAsset = newID, canonical.Model, canonical.ComposeAsset
		if x.Container == "sparktalk-magpie-tts" || x.Container == "" {
			x.Container = canonical.Container
		}
		if x.Name == "" || x.Name == "Magpie TTS" {
			x.Name = canonical.Name
		}
		x.MemoryGiB = max(x.MemoryGiB, canonical.MemoryGiB)
		x.StartupMemoryGiB = max(x.StartupMemoryGiB, canonical.StartupMemoryGiB)
		x.WorkspaceMemoryGiB = max(x.WorkspaceMemoryGiB, canonical.WorkspaceMemoryGiB)
		components = append(components, x)
	}
	known := map[string]orchestrator.Bundle{}
	for _, b := range defaults.Bundles {
		known[b.ID] = b
	}
	for _, b := range catalog.Bundles {
		if _, ok := known[b.ID]; ok && !present {
			components = append(components, canonical)
			present = true
		}
	}
	catalog.Components = components
	for i := range catalog.Bundles {
		b := &catalog.Bundles[i]
		members, seen := []string{}, map[string]bool{}
		for _, id := range b.Components {
			if replacement, ok := ids[id]; ok {
				id = replacement
			}
			if !seen[id] {
				members, seen[id] = append(members, id), true
			}
		}
		profile, builtin := known[b.ID]
		profileHasTTS := false
		for _, id := range profile.Components {
			profileHasTTS = profileHasTTS || id == canonical.ID
		}
		if builtin && profileHasTTS && !seen[canonical.ID] {
			members = append(members, canonical.ID)
			if b.Bindings == nil {
				b.Bindings = map[string]orchestrator.Deployment{}
			}
			if binding, ok := profile.Bindings[canonical.ID]; ok {
				// Keep the saved execution topology; don't invent an SSH host.
				host := ""
				if binding.Host != nil {
					host = *binding.Host
				}
				if _, hostExists := catalog.Hosts[host]; host == "" || hostExists {
					b.Bindings[canonical.ID] = binding
				}
			}
		}
		b.Components = members
		for old, replacement := range ids {
			if binding, ok := b.Bindings[old]; ok && old != replacement {
				if _, exists := b.Bindings[replacement]; !exists {
					b.Bindings[replacement] = binding
				}
				delete(b.Bindings, old)
			}
		}
		for _, id := range members {
			if id != canonical.ID && id != ids[id] {
				continue
			}
			if binding, ok := b.Bindings[id]; ok {
				if binding.MemoryGiB != nil && *binding.MemoryGiB < canonical.MemoryGiB {
					memory := canonical.MemoryGiB
					binding.MemoryGiB = &memory
				}
				if binding.StartupMemoryGiB != nil && *binding.StartupMemoryGiB < canonical.StartupMemoryGiB {
					memory := canonical.StartupMemoryGiB
					binding.StartupMemoryGiB = &memory
				}
				b.Bindings[id] = binding
			}
		}
		b.Description = strings.ReplaceAll(b.Description, "Magpie TTS", "Qwen3-TTS Q8")
	}
	if strings.TrimSpace(c.TTS.Model) == "" || strings.TrimSpace(c.TTS.Model) == "magpietts" {
		c.TTS.Model = canonical.Model
	}
	switch strings.ToLower(strings.TrimSpace(c.TTS.Voice)) {
	case "", "sofia", "aria", "john", "jason", "leo":
		c.TTS.Voice = "sohee"
	}
	if c.TTS.SampleRate == 0 || c.TTS.SampleRate == 22050 {
		c.TTS.SampleRate = 24000
	}
	switch strings.ToLower(c.TTS.Language) {
	case "ar-msa", "ar-ae", "ar-sa", "hi-in", "vi-vn":
		c.TTS.Language = "auto"
	}
}
