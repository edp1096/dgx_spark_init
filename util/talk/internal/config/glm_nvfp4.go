package config

// GLM's retired EXL3 runtime cannot supply defaults for the NVFP4 checkpoint.
// Keep host assignments and the selected variant, but reset unqualified tuning.
func (c *Config) migrateGLMNVFP4() {
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID != "glm53" || x.Controller != "glm53-cluster" {
			continue
		}
		x.Name = "GLM 5.3 Flash NVFP4"
		if x.RuntimeOptions == nil {
			x.RuntimeOptions = map[string]string{}
		}
		x.RuntimeOptions["MAX_MODEL_LEN"] = "8192"
		x.RuntimeOptions["MAX_NUM_SEQS"] = "1"
		x.RuntimeOptions["MTP_TOKENS"] = "0"
		delete(x.RuntimeOptions, "DFLASH_TOKENS")
		delete(x.RuntimeOptions, "KV_CACHE_MEMORY")
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "glm53-worker-extra" {
			continue
		}
		b.Name = "GLM 5.3 Flash NVFP4"
		b.ContextTokens = 8192
		b.Description = "NVFP4 TP2 · 두 Spark 필요 · Huihui 8K 검증"
		if binding, ok := b.Bindings["glm53"]; ok {
			// Runtime options in a bundle override the component defaults.
			if binding.RuntimeOptions != nil {
				(*binding.RuntimeOptions)["MAX_MODEL_LEN"] = "8192"
				(*binding.RuntimeOptions)["MAX_NUM_SEQS"] = "1"
				(*binding.RuntimeOptions)["MTP_TOKENS"] = "0"
				delete(*binding.RuntimeOptions, "DFLASH_TOKENS")
				delete(*binding.RuntimeOptions, "KV_CACHE_MEMORY")
				b.Bindings["glm53"] = binding
			}
		}
	}
}
