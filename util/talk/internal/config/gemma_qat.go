package config

import "sparktalk/internal/orchestrator"

// Change only the previous built-in checkpoint identity. User names, ordering,
// deployment bindings, runtime options, and unrelated active models are retained.
func (c *Config) migrateGemmaCheckpoint() {
	const previous = "sakamakismile/Huihui-gemma-4-26B-A4B-it-qat-abliterated-MTP-NVFP4"
	if c.Runtime.Catalog == nil {
		return
	}
	defaults, err := orchestrator.LoadCatalog()
	if err != nil {
		return
	}
	target, ok := defaults.Component("gemma26")
	if !ok {
		return
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "gemma26" && x.Model == previous {
			x.Model = target.Model
			if x.MemoryGiB == 70 {
				x.MemoryGiB = target.MemoryGiB
			}
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		x := &c.Runtime.Catalog.Bundles[i]
		if x.ID == "gemma26" && x.ModelID == previous {
			x.ModelID = target.Model
		}
	}
	if c.Runtime.Mode == "managed" && c.Model.DefaultModel == previous {
		c.Model.DefaultModel = target.Model
	}
}

func (c *Config) migrateGemmaSGLang() {
	if c.Runtime.Catalog == nil {
		return
	}
	const model = "coolthor/Huihui-gemma-4-26B-A4B-it-abliterated-FP8-Dynamic"
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "gemma26" && x.Model == model && x.Container == "vllm-gemma26" {
			x.Container = "sglang-gemma26"
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		x := &c.Runtime.Catalog.Bundles[i]
		if x.ID == "gemma26" && x.ModelID == model && x.ModelType == "gemma4-vllm" {
			x.ModelType = "gemma4"
		}
	}
	if c.Runtime.Mode == "managed" && c.Model.DefaultModel == model && c.Model.ModelType == "gemma4-vllm" {
		c.Model.ModelType = "gemma4"
	}
}

func (c *Config) migrateGemmaOwnNVFP4() {
	const old = "coolthor/Huihui-gemma-4-26B-A4B-it-abliterated-FP8-Dynamic"
	const target = "edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4"
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "gemma26" && (x.Model == old || x.Model == target) {
			x.Model = target
			x.Container = "sglang-gemma26"
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		x := &c.Runtime.Catalog.Bundles[i]
		if x.ID == "gemma26" && (x.ModelID == old || x.ModelID == target) {
			x.ModelID = target
			x.ModelType = "gemma4"
		}
	}
	if c.Runtime.Mode == "managed" && (c.Model.DefaultModel == old || c.Model.DefaultModel == target) {
		c.Model.DefaultModel = target
		c.Model.ModelType = "gemma4"
	}
}
