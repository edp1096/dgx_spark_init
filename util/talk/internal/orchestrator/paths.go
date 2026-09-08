package orchestrator

import (
	"fmt"
	"path/filepath"
)

// Preparation and launch must resolve paths from the same execution host.
func (c *Controller) runtimeHostPaths(component Component) (string, string, error) {
	host := c.host(component.Host)
	c.mu.RLock()
	data, cache := c.dataDir, c.modelCache
	c.mu.RUnlock()
	if host.Address != "" && (host.DataDir == "" || host.ModelCache == "") {
		return "", "", fmt.Errorf("host %s requires data_dir and model_cache", component.Host)
	}
	if host.DataDir != "" {
		data = host.DataDir
	}
	if host.ModelCache != "" {
		cache = host.ModelCache
	}
	if !filepath.IsAbs(data) || !filepath.IsAbs(cache) {
		return "", "", fmt.Errorf("absolute runtime data/cache directories are required")
	}
	return data, cache, nil
}

func runtimePathEnvironment(component Component, data, cache string) []string {
	env := []string{"env", "SPARKTALK_DATA_DIR=" + data, "SPARKTALK_HF_CACHE=" + cache}
	switch component.ComposeAsset {
	case "compose.nemotron-asr.yaml":
		env = append(env, "SPARKTALK_NEMO_MODEL_DIR="+filepath.Join(filepath.Dir(cache), "nemo-speech"))
	case "compose.magpie-tts.yaml":
		env = append(env, "SPARKTALK_MAGPIE_MODEL_DIR="+filepath.Join(filepath.Dir(cache), "nemo-speech", "magpie-v2607"))
	}
	return env
}
