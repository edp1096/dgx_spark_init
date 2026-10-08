package config

import "sparktalk/internal/orchestrator"

// Update only the old built-in model identity; preserve custom models and settings.
func (c *Config) migrateQwenHuihuiModel() {
	defaults, err := orchestrator.LoadCatalog()
	if err != nil || c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if (x.ID == "flash-next" || x.ID == "flash-next-tp2") && x.Model == "qwen3.8-flash-next" {
			d, _ := defaults.Component(x.ID)
			x.Model = d.Model
			if x.Name == "Flash-Next" || x.Name == "Qwen3.8 Flash-Next" || x.Name == "Qwen3.8 Flash-Next TP2" {
				x.Name = d.Name
			}
		}
	}
	for i := range c.Runtime.Catalog.Bundles {
		x := &c.Runtime.Catalog.Bundles[i]
		if (x.ID == "flash-next" || x.ID == "flash-next-tp2") && x.ModelID == "qwen3.8-flash-next" {
			d, _ := defaults.Bundle(x.ID)
			x.ModelID = d.ModelID
			if x.Name == "Flash-Next" || x.Name == "Qwen3.8 Flash-Next" || x.Name == "Qwen3.8 Flash-Next TP2 · 1M" {
				x.Name = d.Name
			}
		}
	}
	if c.Runtime.Mode == "managed" && c.Model.DefaultModel == "qwen3.8-flash-next" {
		id := c.Runtime.ActiveBundle
		if id == "" {
			id = c.Runtime.Bundle
		}
		if id != "flash-next-tp2" {
			id = "flash-next"
		}
		d, _ := defaults.Component(id)
		c.Model.DefaultModel = d.Model
	}
}
