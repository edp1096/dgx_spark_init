package config

func (c Config) SupportDisabled(key string) bool {
	bundle := c.Runtime.ActiveBundle
	if bundle == "" {
		bundle = c.Runtime.Bundle
	}
	return c.Runtime.Mode == "managed" && c.Runtime.Catalog != nil && c.Runtime.Catalog.SupportDisabled(bundle, "extra-"+key)
}

func (c Config) SupportEnabled(key string) bool {
	if c.SupportDisabled(key) {
		return false
	}
	switch key {
	case "shorts", "fish", "music-ace":
		return c.Runtime.Mode == "managed"
	case "embedding":
		return c.SemanticSearchEnabled()
	case "media":
		return c.Tools.MediaImportEnabled
	case "collector":
		return c.Extra.CollectorEnabled
	case "documents":
		return c.Extra.DocumentsEnabled
	case "ssh":
		return c.Extra.SSHEnabled
	}
	return false
}
func (c Config) SupportEndpoint(key string) string {
	switch key {
	case "shorts", "fish", "music-ace":
		id := "extra-" + key
		if c.Runtime.Catalog != nil {
			for _, x := range c.Runtime.Catalog.Components {
				if x.ID == id {
					return x.Endpoint
				}
			}
		}
		if key == "shorts" {
			return "http://127.0.0.1:8725"
		}
		if key == "music-ace" {
			return "http://127.0.0.1:8727"
		}
		return "http://127.0.0.1:8726"
	case "embedding":
		return c.Embedding.Endpoint
	case "media":
		if c.Extra.MediaEndpoint != "" {
			return c.Extra.MediaEndpoint
		}
		return c.ASR.FFmpegEndpoint
	case "collector":
		return c.Extra.CollectorEndpoint
	case "documents":
		return c.Extra.DocumentsEndpoint
	case "ssh":
		return c.Extra.SSHEndpoint
	}
	return ""
}
