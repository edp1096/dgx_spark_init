package config

import "sparktalk/internal/orchestrator"

// Upgrade only the retired built-in deployment. Preserve custom controllers,
// host connection details, endpoints and the selected/active model set.
func (c *Config) upgradeDS41TensorFold() {
	defaults, err := orchestrator.LoadCatalog()
	if err != nil || c.Runtime.Catalog == nil {
		return
	}
	want, _ := defaults.Component("ds41")
	upgraded := false
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID != "ds41" || x.Controller != "ds41-cluster" || x.Model != "deepseek-v4.1-flash" || x.Container != "ds41-stream-0" || x.WorkerContainer != "ds41-stream-1" {
			continue
		}
		x.Model, x.Container, x.WorkerContainer = want.Model, want.Container, want.WorkerContainer
		x.MemoryGiB, x.WorkerMemoryGiB = max(x.MemoryGiB, want.MemoryGiB), max(x.WorkerMemoryGiB, want.WorkerMemoryGiB)
		options := map[string]string{}
		for key, value := range x.RuntimeOptions {
			if key != "DSV41_PRELOAD_COUNT" {
				options[key] = value
			}
		}
		options["MODEL_VARIANT"], options["MAX_MODEL_LEN"] = want.RuntimeOptions["MODEL_VARIANT"], "1048576"
		x.RuntimeOptions = options
		upgraded = true
	}
	if !upgraded {
		return
	}
	wantBundle, _ := defaults.Bundle("ds41")
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID != "ds41" || b.ModelID != "deepseek-v4.1-flash" {
			continue
		}
		b.ModelID, b.ContextTokens, b.Description = wantBundle.ModelID, wantBundle.ContextTokens, wantBundle.Description
		// Migrate default speech bindings only. Preserve custom placements and
		// retain eager startup when they cannot use the local workload lease.
		localSpeech := true
		for _, id := range []string{"nemotron-asr", "qwen3-tts"} {
			d, ok := b.Bindings[id]
			if !ok {
				localSpeech = false
				continue
			}
			expected := "http://192.168.100.60:8693"
			if id == "qwen3-tts" {
				expected = "http://192.168.100.60:8692"
			}
			if d.Host != nil && *d.Host == "worker" && d.Endpoint != nil && *d.Endpoint == expected && (d.MemoryGiB == nil || *d.MemoryGiB == 4) {
				b.Bindings[id] = wantBundle.Bindings[id]
			} else if d.Host == nil || *d.Host != "local" {
				localSpeech = false
			}
		}
		b.WorkloadSwap = localSpeech
	}
}
