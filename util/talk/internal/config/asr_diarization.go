package config

// The old short-dictation estimate omitted long-media ASR workspaces. Preserve
// explicitly customized estimates; update only the previous built-in value.
func (c *Config) migrateNemotronDiarizationBudget() {
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "nemotron-asr" && x.Model == "nemotron-3.5-asr-streaming-0.6b" && x.MemoryGiB == 1.3 {
			x.MemoryGiB = 6
		}
	}
}
