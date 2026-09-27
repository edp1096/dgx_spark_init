package config

// Upgrade only the initial 8K GLM profile. Leave explicitly customized context
// or KV budgets, model selection, host assignments and other models intact.
func (c *Config) migrateGLMLongContext() {
	if c.Runtime.Catalog == nil {
		return
	}
	var options map[string]string
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "glm53" && x.Controller == "glm53-cluster" {
			options = x.RuntimeOptions
		}
	}
	if options["MAX_MODEL_LEN"] != "8192" || options["KV_CACHE_MEMORY"] != "" {
		return
	}
	for _, b := range c.Runtime.Catalog.Bundles {
		if b.ID != "glm53-worker-extra" {
			continue
		}
		if binding, ok := b.Bindings["glm53"]; ok && binding.RuntimeOptions != nil {
			o := *binding.RuntimeOptions
			if o["KV_CACHE_MEMORY"] != "" || (o["MAX_MODEL_LEN"] != "" && o["MAX_MODEL_LEN"] != "8192") {
				return
			}
		}
	}
	options["MAX_MODEL_LEN"] = "1048576"
	options["KV_CACHE_MEMORY"] = "12884901888"
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "glm53-worker-extra" {
			continue
		}
		if binding, ok := b.Bindings["glm53"]; ok && binding.RuntimeOptions != nil {
			o := *binding.RuntimeOptions
			if o["MAX_MODEL_LEN"] == "8192" {
				o["MAX_MODEL_LEN"] = "1048576"
				o["KV_CACHE_MEMORY"] = "12884901888"
			}
		}
		if b.ContextTokens == 8192 {
			b.ContextTokens = 1048576
			if b.Description == "NVFP4 TP2 · 두 Spark 필요 · Huihui 8K 검증" {
				b.Description = "NVFP4 TP2 · 두 Spark 필요 · Huihui 1M 검증"
			}
		}
	}
}
