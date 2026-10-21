package config

import "sparktalk/internal/orchestrator"

// Upgrade only the built-in GLM deployment. Preserve IDs, host placement,
// the selected checkpoint variant and network settings; drop retired tuning.
func (c *Config) upgradeGLMFP4Local() {
	if c.Runtime.Catalog == nil {
		return
	}
	normalize := func(options map[string]string) map[string]string {
		if options == nil {
			options = map[string]string{}
		}
		for _, key := range []string{"DFLASH_TOKENS", "KV_CACHE_MEMORY", "GPU_MEMORY_UTILIZATION"} {
			delete(options, key)
		}
		options["MAX_MODEL_LEN"], options["MAX_NUM_SEQS"] = "1048576", "1"
		if options["MTP_TOKENS"] != "1" && options["MTP_TOKENS"] != "5" {
			options["MTP_TOKENS"] = "5"
		}
		if options["MODEL_VARIANT"] == "" {
			options["MODEL_VARIANT"] = "abliterated"
		}
		return options
	}
	upgraded := false
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID != "glm53" || (x.Controller != "glm53-cluster" && x.Controller != "glm53-fp4-cluster") {
			continue
		}
		x.Controller, x.ProgressKind = "glm53-fp4-cluster", "b12x"
		x.Name = orchestrator.RuntimeDisplayName(x.ID, x.Name)
		x.RuntimeOptions = normalize(x.RuntimeOptions)
		if x.MemoryGiB == 0 || x.MemoryGiB == 110 || x.MemoryGiB == 113 {
			x.MemoryGiB = 108
		}
		if x.WorkerMemoryGiB == 0 || x.WorkerMemoryGiB == 113 {
			x.WorkerMemoryGiB = 108
		}
		if x.StartupMemoryGiB == 0 {
			x.StartupMemoryGiB = 115
		}
		upgraded = true
	}
	if !upgraded {
		return
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if binding, ok := b.Bindings["glm53"]; ok && binding.RuntimeOptions != nil {
			options := normalize(*binding.RuntimeOptions)
			binding.RuntimeOptions = &options
			b.Bindings["glm53"] = binding
		}
		if b.ID == "glm53-worker-extra" {
			b.Name = orchestrator.RuntimeDisplayName(b.ID, b.Name)
			b.ContextTokens = 1048576
			b.Description = "2× DGX Spark · FP4-Local · vLLM/B12X · FP8 KV 1M · MTP 5 · TTS·Extra"
		}
	}
}
