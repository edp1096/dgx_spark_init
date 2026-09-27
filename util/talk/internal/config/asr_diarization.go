package config

import "sparktalk/internal/orchestrator"

// The old short-dictation estimate omitted long-media ASR workspaces. Preserve
// explicitly customized estimates; update only the previous built-in value.
func (c *Config) migrateNemotronDiarizationBudget() {
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "nemotron-asr" && x.Model == "nemotron-3.5-asr-streaming-0.6b" && (x.MemoryGiB == 1.3 || x.MemoryGiB == 6) {
			x.MemoryGiB = 6
			if x.StartupMemoryGiB == 0 {
				x.StartupMemoryGiB = 3.5
			}
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "flash-next" {
			continue
		}
		for _, id := range b.Components {
			if id != "nemotron-asr" {
				continue
			}
			if b.Bindings == nil {
				b.Bindings = map[string]orchestrator.Deployment{}
			}
			binding := b.Bindings[id]
			if binding.StartAfterLLM == nil {
				enabled := true
				binding.StartAfterLLM = &enabled
				b.Bindings[id] = binding
			}
		}
	}

}
