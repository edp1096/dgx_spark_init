package config

import (
	"path/filepath"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestGLMFP4UpgradePersistsDeploymentAndSelection(t *testing.T) {
	path := filepath.Join(t.TempDir(), "talk.yaml")
	cfg, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	for i := range cfg.Runtime.Catalog.Components {
		x := &cfg.Runtime.Catalog.Components[i]
		if x.ID == "glm53" {
			x.Controller = "glm53-cluster"
			x.ProgressKind = "sglang"
			x.Name = "GLM 5.3 Flash NVFP4"
			x.Host = "local"
			x.WorkerHost = "worker"
			x.Endpoint = "http://127.0.0.1:19053"
			x.Port = 19053
			x.MemoryGiB = 113
			x.WorkerMemoryGiB = 113
			x.StartupMemoryGiB = 0
			x.RuntimeOptions = map[string]string{"MODEL_VARIANT": "abliterated", "MTP_TOKENS": "0", "DFLASH_TOKENS": "5", "MASTER_PORT": "29599", "HEAD_RAIL_IP": "10.200.0.1"}
		}
	}
	cfg.Runtime.BuiltinRevision = 45
	cfg.Runtime.Bundle = "ds41"
	cfg.Runtime.ActiveBundle = "flash-next-fp4-local"
	cfg.Runtime.AutoStart = false
	cfg.Normalize()
	if err := Save(path, cfg); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	x, _ := loaded.Runtime.Catalog.Component("glm53")
	if x.Controller != "glm53-fp4-cluster" || x.ProgressKind != "b12x" || x.Name != "GLM 5.3 Flash FP4-Local TP2" || x.Port != 19053 || x.Endpoint != "http://127.0.0.1:19053" || x.Host != "local" || x.WorkerHost != "worker" {
		t.Fatalf("deployment not preserved: %+v", x)
	}
	if x.RuntimeOptions["MTP_TOKENS"] != "5" || x.RuntimeOptions["DFLASH_TOKENS"] != "" || x.RuntimeOptions["MAX_MODEL_LEN"] != "1048576" || x.RuntimeOptions["MASTER_PORT"] != "29599" || x.RuntimeOptions["MODEL_VARIANT"] != "abliterated" {
		t.Fatal(x.RuntimeOptions)
	}
	if x.MemoryGiB != 108 || x.WorkerMemoryGiB != 108 || x.StartupMemoryGiB != 115 {
		t.Fatal("incorrect qualified memory plan", x)
	}
	if loaded.Runtime.AutoStart || loaded.Runtime.Bundle != "ds41" || loaded.Runtime.ActiveBundle != "flash-next-fp4-local" || loaded.Runtime.BuiltinRevision != 51 {
		t.Fatal("selection changed")
	}
}

func TestGLMFP4UpgradePreservesCustomAndBundleOverrides(t *testing.T) {
	opts := map[string]string{"MODEL_VARIANT": "official", "MTP_TOKENS": "1", "DFLASH_TOKENS": "5"}
	cat := orchestrator.Catalog{Components: []orchestrator.Component{{ID: "glm53", Name: "My GLM", Controller: "glm53-cluster", MemoryGiB: 109, WorkerMemoryGiB: 109, StartupMemoryGiB: 116}, {ID: "custom-glm", Controller: "glm53-cluster"}}, Bundles: []orchestrator.Bundle{{ID: "custom", Bindings: map[string]orchestrator.Deployment{"glm53": {RuntimeOptions: &opts}}}}}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &cat}}
	cfg.upgradeGLMFP4Local()
	cfg.upgradeGLMFP4Local()
	x := cat.Components[0]
	if x.Name != "My GLM" || x.MemoryGiB != 109 || x.WorkerMemoryGiB != 109 || x.StartupMemoryGiB != 116 || cat.Components[1].Controller != "glm53-cluster" {
		t.Fatal("custom deployment changed")
	}
	if opts["MODEL_VARIANT"] != "official" || opts["MTP_TOKENS"] != "1" || opts["DFLASH_TOKENS"] != "" || opts["MAX_MODEL_LEN"] != "1048576" {
		t.Fatal(opts)
	}
}
