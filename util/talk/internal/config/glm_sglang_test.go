package config

import (
	"testing"

	"sparktalk/internal/orchestrator"
)

func TestGLMSGLangMigrationPreservesModelContextAndHosts(t *testing.T) {
	opts := map[string]string{"MAX_MODEL_LEN": "262144", "MODEL_VARIANT": "abliterated", "KV_CACHE_MEMORY": "12884901888", "GPU_MEMORY_UTILIZATION": "0.85", "HEAD_RAIL_IP": "10.200.0.1"}
	bindingOpts := map[string]string{"MAX_MODEL_LEN": "524288", "KV_CACHE_MEMORY": "12884901888"}
	cat := orchestrator.Catalog{
		Components: []orchestrator.Component{
			{ID: "glm53", Controller: "glm53-cluster", Host: "local", WorkerHost: "worker", ProgressKind: "service", MemoryGiB: 110, WorkerMemoryGiB: 108, RuntimeOptions: opts},
			{ID: "flash-next", RuntimeOptions: map[string]string{"MTP_TOKENS": "3"}},
		},
		Bundles: []orchestrator.Bundle{{ID: "glm53-worker-extra", ContextTokens: 524288, Bindings: map[string]orchestrator.Deployment{"glm53": {RuntimeOptions: &bindingOpts}}}},
	}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &cat, ActiveBundle: "flash-next"}}
	cfg.migrateGLMSGLang()
	cfg.migrateGLMSGLang()
	if opts["MAX_MODEL_LEN"] != "262144" || opts["MODEL_VARIANT"] != "abliterated" || opts["HEAD_RAIL_IP"] != "10.200.0.1" || bindingOpts["MAX_MODEL_LEN"] != "524288" {
		t.Fatal("model, context or rail changed", opts, bindingOpts)
	}
	for _, o := range []map[string]string{opts, bindingOpts} {
		if o["DFLASH_TOKENS"] != "5" || o["MTP_TOKENS"] != "0" || o["MAX_NUM_SEQS"] != "1" || o["KV_CACHE_MEMORY"] != "" || o["GPU_MEMORY_UTILIZATION"] != "" {
			t.Fatal("SGLang profile not migrated", o)
		}
	}
	x := cat.Components[0]
	if x.ProgressKind != "sglang" || x.MemoryGiB != 113 || x.WorkerMemoryGiB != 113 || x.WorkerHost != "worker" || cfg.Runtime.ActiveBundle != "flash-next" || cat.Components[1].RuntimeOptions["MTP_TOKENS"] != "3" {
		t.Fatal("component/selection migration mismatch", x)
	}
}

func TestRemoveGLMMagpiePreservesOtherBundle(t *testing.T) {
	c := Config{Runtime: RuntimeConfig{Catalog: &orchestrator.Catalog{Bundles: []orchestrator.Bundle{
		{ID: "glm53-worker-extra", Components: []string{"glm53", "magpie-tts", "extra-media"}, Bindings: map[string]orchestrator.Deployment{"magpie-tts": {}}},
		{ID: "ds41", Components: []string{"ds41", "magpie-tts"}},
	}}}}
	c.removeGLMMagpie()
	c.removeGLMMagpie()
	b := c.Runtime.Catalog.Bundles[0]
	if len(b.Components) != 2 || b.Components[0] != "glm53" || b.Components[1] != "extra-media" {
		t.Fatal(b.Components)
	}
	if _, ok := b.Bindings["magpie-tts"]; ok {
		t.Fatal("GLM TTS binding retained")
	}
	if len(c.Runtime.Catalog.Bundles[1].Components) != 2 {
		t.Fatal("other bundle changed")
	}
}
