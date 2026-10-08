package config

func (c Config) SupportEnabled(key string) bool {
	switch key {
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
