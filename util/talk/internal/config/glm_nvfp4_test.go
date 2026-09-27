package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestGLMNVFP4MigrationPreservesOtherModelsAndSelection(t *testing.T) {
	options := map[string]string{"MAX_MODEL_LEN": "524288", "DFLASH_TOKENS": "7", "MODEL_VARIANT": "abliterated", "KV_CACHE_MEMORY": "14400000000"}
	catalog := orchestrator.Catalog{
		Components: []orchestrator.Component{
			{ID: "glm53", Controller: "glm53-cluster", Host: "local", WorkerHost: "worker", RuntimeOptions: map[string]string{"MODEL_VARIANT": "abliterated", "MAX_MODEL_LEN": "524288"}},
			{ID: "flash-next", RuntimeOptions: map[string]string{"MAX_MODEL_LEN": "1048576", "MTP_TOKENS": "3"}},
		},
		Bundles: []orchestrator.Bundle{{ID: "glm53-worker-extra", Bindings: map[string]orchestrator.Deployment{"glm53": {RuntimeOptions: &options}}}},
	}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &catalog, Bundle: "flash-next", ActiveBundle: "flash-next"}}
	cfg.migrateGLMNVFP4()
	glm := catalog.Components[0]
	if glm.RuntimeOptions["MODEL_VARIANT"] != "abliterated" || glm.RuntimeOptions["MAX_MODEL_LEN"] != "8192" || glm.WorkerHost != "worker" {
		t.Fatalf("migration lost selection or host: %+v", glm)
	}
	if options["MAX_MODEL_LEN"] != "8192" || options["DFLASH_TOKENS"] != "" || options["KV_CACHE_MEMORY"] != "" {
		t.Fatal(options)
	}
	if catalog.Components[1].RuntimeOptions["MAX_MODEL_LEN"] != "1048576" || cfg.Runtime.ActiveBundle != "flash-next" {
		t.Fatal("unrelated QAD state changed")
	}
	cfg.migrateGLMNVFP4()
	if glm.RuntimeOptions["MODEL_VARIANT"] != "abliterated" {
		t.Fatal("migration not idempotent")
	}
}
