package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestGLMLongContextMigration(t *testing.T) {
	for _, custom := range []bool{false, true} {
		options := map[string]string{"MAX_MODEL_LEN": "8192", "MODEL_VARIANT": "abliterated", "MTP_TOKENS": "0"}
		if custom {
			options["MAX_MODEL_LEN"] = "262144"
		}
		catalog := orchestrator.Catalog{
			Components: []orchestrator.Component{
				{ID: "glm53", Controller: "glm53-cluster", Host: "local", WorkerHost: "worker", RuntimeOptions: options},
				{ID: "flash-next", RuntimeOptions: map[string]string{"MAX_MODEL_LEN": "1048576", "MTP_TOKENS": "3"}},
			},
			Bundles: []orchestrator.Bundle{{ID: "glm53-worker-extra", ContextTokens: 8192}},
		}
		cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog, ActiveBundle: "flash-next"}}
		cfg.migrateGLMLongContext()
		cfg.migrateGLMLongContext()
		if custom {
			if options["MAX_MODEL_LEN"] != "262144" || options["KV_CACHE_MEMORY"] != "" {
				t.Fatal("custom context changed", options)
			}
		} else if options["MAX_MODEL_LEN"] != "1048576" || options["KV_CACHE_MEMORY"] != "12884901888" || catalog.Bundles[0].ContextTokens != 1048576 {
			t.Fatal("initial profile not upgraded", options, catalog.Bundles[0])
		}
		if options["MODEL_VARIANT"] != "abliterated" || catalog.Components[0].WorkerHost != "worker" || catalog.Components[1].RuntimeOptions["MTP_TOKENS"] != "3" || cfg.Runtime.ActiveBundle != "flash-next" {
			t.Fatal("unrelated selection changed")
		}
	}
}

func TestGLMLongContextPreservesCustomBundleBudget(t *testing.T) {
	options := map[string]string{"KV_CACHE_MEMORY": "8000000000"}
	catalog := orchestrator.Catalog{
		Components: []orchestrator.Component{{ID: "glm53", Controller: "glm53-cluster", RuntimeOptions: map[string]string{"MAX_MODEL_LEN": "8192"}}},
		Bundles:    []orchestrator.Bundle{{ID: "glm53-worker-extra", ContextTokens: 8192, Bindings: map[string]orchestrator.Deployment{"glm53": {RuntimeOptions: &options}}}},
	}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog}}
	cfg.migrateGLMLongContext()
	if options["KV_CACHE_MEMORY"] != "8000000000" || catalog.Bundles[0].ContextTokens != 8192 || catalog.Components[0].RuntimeOptions["MAX_MODEL_LEN"] != "8192" {
		t.Fatal("custom bundle budget changed")
	}
}
