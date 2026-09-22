package config

import (
	"sparktalk/internal/orchestrator"
	"strings"
)

func (c *Config) applyManagedRuntime() {
	c.ApplyManagedBundle(c.Runtime.ActiveBundle)
}

// ApplyManagedBundle updates only the live model profile. Runtime.Bundle is
// the user's startup default and must not be changed by a live model switch.
func (c *Config) ApplyManagedBundle(bundle string) {
	catalog, _ := orchestrator.LoadCatalog()
	if c.Runtime.Catalog != nil {
		var err error
		catalog, err = orchestrator.ValidateCatalog(*c.Runtime.Catalog)
		if err != nil {
			return
		}
	}
	profile, ok := catalog.Bundle(strings.ToLower(strings.TrimSpace(bundle)))
	if !ok {
		return
	}
	c.Model.DefaultModel, c.Model.ModelType = profile.ModelID, profile.ModelType
	c.Context.WindowTokens = profile.ContextTokens
	present := map[string]bool{}
	for _, id := range profile.Components {
		component, _ := catalog.ResolveComponent(profile.ID, id)
		role := component.ServiceRole()
		present[role] = true
		endpoint := strings.TrimRight(component.Endpoint, "/")
		switch role {
		case "llm":
			c.Model.Endpoint = endpoint
		case "asr":
			c.ASR.Endpoint, c.ASR.Model = endpoint, component.Model
		case "tts":
			c.TTS.Endpoint, c.TTS.Model = endpoint, component.Model
		case "image":
			c.Image.Endpoint, c.Image.Model = endpoint, component.Model
		case "media":
			c.Extra.MediaEndpoint, c.ASR.FFmpegEndpoint = endpoint, endpoint
		case "ssh":
			c.Extra.SSHEndpoint = endpoint
		case "documents":
			c.Extra.DocumentsEndpoint = endpoint
		case "collector":
			c.Extra.CollectorEndpoint = endpoint
		}
	}
	// A set cannot call services it does not contain; preserve user toggles for present members.
	if !present["asr"] {
		c.ASR.Enabled = false
	}
	if !present["tts"] {
		c.TTS.Enabled = false
	}
	if !present["image"] {
		c.Image.Enabled = false
	}
	// Shared support services remain available independently of model membership.
	for _, component := range catalog.SupportComponents(profile.ID) {
		endpoint := strings.TrimRight(component.Endpoint, "/")
		spec, _ := component.SupportSpec()
		switch spec.Key {
		case "media":
			c.Extra.MediaEndpoint, c.ASR.FFmpegEndpoint = endpoint, endpoint
		case "ssh":
			c.Extra.SSHEndpoint = endpoint
		case "collector":
			c.Extra.CollectorEndpoint = endpoint
		case "documents":
			c.Extra.DocumentsEndpoint = endpoint
		}
	}
	c.Model.ReasoningEffort = normalizeReasoningEffort(c.Model.ModelType, c.Model.ReasoningEffort)
}
