package config

func (c *Config) normalizeRuntimeDisplayNames() {
	if c.Runtime.Catalog == nil {
		return
	}
	rename := func(id, name string) string {
		switch id {
		case "gemma", "gemma31":
			if name == "Gemma" || name == "Gemma 31B" || name == "Gemma 4 31B" {
				return "Gemma4 31B"
			}
		case "flash-next":
			if name == "Flash-Next" {
				return "Qwen3.8 Flash-Next"
			}
		}
		return name
	}
	for i := range c.Runtime.Catalog.Components {
		item := &c.Runtime.Catalog.Components[i]
		item.Name = rename(item.ID, item.Name)
	}
	for i := range c.Runtime.Catalog.Bundles {
		item := &c.Runtime.Catalog.Bundles[i]
		item.Name = rename(item.ID, item.Name)
	}
}
