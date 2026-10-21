package config

import "sparktalk/internal/orchestrator"

// Replace the built-in engines in place. IDs, host placement, bindings and
// user-selected sets remain stable; external/custom deployments are preserved.
func (c *Config) upgradeQwenTensorFold() {
	defaults, err := orchestrator.LoadCatalog()
	if err != nil || c.Runtime.Catalog == nil {
		return
	}
	upgraded := map[string]bool{}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		oldContainer := ""
		oldMemory, oldStartup, oldWorkspace := 0.0, 0.0, 0.0
		switch x.ID {
		case "flash-next-fp4-local":
			oldContainer = "vllm-qwen38-fp4-local"
			oldMemory, oldStartup, oldWorkspace = 103, 116, 3
		case "qwen38fn_exl3":
			oldContainer = "sparktalk-qwen38fn_exl3"
			oldMemory = 85
		default:
			continue
		}
		want, _ := defaults.Component(x.ID)
		if x.Controller != "compose" || x.Container != oldContainer || x.ComposeAsset != want.ComposeAsset || x.Model != want.Model {
			continue
		}
		x.Container, x.ProgressKind = want.Container, want.ProgressKind
		if x.MemoryGiB == 0 || x.MemoryGiB == oldMemory {
			x.MemoryGiB = want.MemoryGiB
		}
		if x.StartupMemoryGiB == 0 || x.StartupMemoryGiB == oldStartup {
			x.StartupMemoryGiB = want.StartupMemoryGiB
		}
		if x.WorkspaceMemoryGiB == 0 || x.WorkspaceMemoryGiB == oldWorkspace {
			x.WorkspaceMemoryGiB = want.WorkspaceMemoryGiB
		}
		upgraded[x.ID] = true
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if !upgraded[b.ID] {
			continue
		}
		want, _ := defaults.Bundle(b.ID)
		stockEXL3 := b.ID == "qwen38fn_exl3" && b.ModelType == want.ModelType && b.Description == "TensorFold · EXL3 3bit HQ h6 ng6 · Q8 KV 1M·YaRN 4·MTP 3 · 이미지·ASR·TTS·Extra"
		if b.ModelID == want.ModelID && (b.ModelType == "qwen3.8-vllm" || b.ModelType == "qwen38fn_exl3" || stockEXL3) {
			b.ModelType, b.Description = want.ModelType, want.Description
		}
	}
}

// Refresh the known built-in text without replaying an older engine migration.
func (c *Config) refreshQwenEXL3TuningDescription() {
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "qwen38fn_exl3" || b.ModelType != "qwen3.8-tensorfold" || b.Description != "TensorFold · EXL3 3bit HQ h6 ng6 · Q8 KV 1M·YaRN 4·MTP 3 · 이미지·ASR·TTS·Extra" {
			continue
		}
		for _, component := range c.Runtime.Catalog.Components {
			if component.ID != b.ID || component.Controller != "compose" || component.Container != "sparktalk-qwen38fn_exl3" || component.ComposeAsset != "compose.qwen38fn_exl3.yaml" || component.Model != b.ModelID {
				continue
			}
			defaults, err := orchestrator.LoadCatalog()
			if err != nil {
				return
			}
			want, ok := defaults.Bundle(b.ID)
			if ok {
				b.Description = want.Description
			}
			break
		}
	}
}
