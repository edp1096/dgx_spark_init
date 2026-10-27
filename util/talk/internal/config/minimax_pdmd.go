package config

// Account for the resident BF16 adapter without changing custom deployments.
func (c *Config) upgradeMiniMaxPDMD() {
	if c.Runtime.Catalog == nil {
		return
	}
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID != "qwim-mmh3" || x.Controller != "compose" || x.Container != "sparktalk-qwim-mmh3" || x.ComposeAsset != "compose.qwim-mmh3.yaml" {
			continue
		}
		if x.MemoryGiB == 0 || x.MemoryGiB == 28 {
			x.MemoryGiB = 30
		}
		if x.StartupMemoryGiB == 0 || x.StartupMemoryGiB == 18 {
			x.StartupMemoryGiB = 20
		}
		if x.Name == "Qwen Image 2.1 + MiniMax H3" {
			x.Name = "Qwen Image 2.1 + MiniMax H3 PDMD4"
		}
	}
}
