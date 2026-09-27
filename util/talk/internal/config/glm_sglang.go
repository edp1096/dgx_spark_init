package config

// Keep the selected checkpoint, context and hosts. The retired vLLM byte-budget
// knobs cannot control SGLang's token pool and must not silently override it.
func (c *Config) migrateGLMSGLang() {
	if c.Runtime.Catalog == nil {
		return
	}
	upgrade := func(options map[string]string) {
		delete(options, "KV_CACHE_MEMORY")
		delete(options, "GPU_MEMORY_UTILIZATION")
		options["MAX_NUM_SEQS"] = "1"
		options["MTP_TOKENS"] = "0"
		options["DFLASH_TOKENS"] = "5"
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID != "glm53" || x.Controller != "glm53-cluster" {
			continue
		}
		x.ProgressKind = "sglang"
		if x.RuntimeOptions == nil {
			x.RuntimeOptions = map[string]string{}
		}
		upgrade(x.RuntimeOptions)
		if x.MemoryGiB == 0 || x.MemoryGiB == 110 {
			x.MemoryGiB = 113
		}
		if x.WorkerMemoryGiB == 0 || x.WorkerMemoryGiB == 108 {
			x.WorkerMemoryGiB = 113
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "glm53-worker-extra" {
			continue
		}
		b.Description = "NVFP4 TP2 · SGLang · DFlash2 · 두 Spark 필요"
		if binding, ok := b.Bindings["glm53"]; ok && binding.RuntimeOptions != nil {
			opts := *binding.RuntimeOptions
			if opts == nil {
				opts = map[string]string{}
				binding.RuntimeOptions = &opts
			}
			upgrade(opts)
			b.Bindings["glm53"] = binding
		}
	}
}

// Only the GLM bundle drops TTS; other bundles and the saved voice are retained.
func (c *Config) removeGLMMagpie() {
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "glm53-worker-extra" {
			continue
		}
		ids := make([]string, 0, len(b.Components))
		for _, id := range b.Components {
			if id != "magpie-tts" {
				ids = append(ids, id)
			}
		}
		b.Components = ids
		delete(b.Bindings, "magpie-tts")
	}
}
