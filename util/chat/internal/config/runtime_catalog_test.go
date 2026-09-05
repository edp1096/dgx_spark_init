package config

import (
	"path/filepath"
	"testing"

	"sparktalk/internal/orchestrator"
)

func TestEditableSetEndpointsSurviveSaveReload(t *testing.T) {
	catalog, _ := orchestrator.LoadCatalog()
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Bundle: "glm53-worker-extra", ActiveBundle: "glm53-worker-extra", Catalog: &catalog}}
	cfg.Normalize()
	if cfg.Model.ModelType != "glm5.3" || cfg.Extra.CollectorEndpoint != "http://192.168.100.60:8695" || cfg.ASR.FFmpegEndpoint != "http://192.168.100.60:8690" {
		t.Fatalf("wrong wiring: %+v %+v", cfg.Model, cfg.Extra)
	}
	if cfg.ASR.Enabled || cfg.TTS.Enabled || cfg.Image.Enabled {
		t.Fatal("absent services enabled")
	}
	for i := range cfg.Runtime.Catalog.Bundles {
		if cfg.Runtime.Catalog.Bundles[i].ID == "glm53-worker-extra" {
			value := cfg.Runtime.Catalog.Bundles[i].Bindings["extra-collector"]
			endpoint := "http://127.0.0.1:18695"
			value.Endpoint = &endpoint
			cfg.Runtime.Catalog.Bundles[i].Bindings["extra-collector"] = value
		}
	}
	cfg.Normalize()
	path := filepath.Join(t.TempDir(), "sparktalk.yaml")
	if err := Save(path, cfg); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if loaded.Extra.CollectorEndpoint != "http://127.0.0.1:18695" {
		t.Fatal("custom endpoint overwritten")
	}
	if loaded.Runtime.Bundle != "glm53-worker-extra" {
		t.Fatal("custom set lost")
	}
}

func TestSwitchingSetsAppliesOwnExtraBinding(t *testing.T) {
	catalog, _ := orchestrator.LoadCatalog()
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Bundle: "qwen27", ActiveBundle: "qwen27", Catalog: &catalog}}
	cfg.Normalize()
	if cfg.Extra.CollectorEndpoint != "http://127.0.0.1:8695" {
		t.Fatal("wrong initial endpoint")
	}
	cfg.Runtime.ActiveBundle = "glm53-worker-extra"
	cfg.Normalize()
	if cfg.Extra.CollectorEndpoint != "http://192.168.100.60:8695" {
		t.Fatal("worker binding ignored")
	}
	cfg.Runtime.ActiveBundle = "qwen27"
	cfg.Normalize()
	if cfg.Extra.CollectorEndpoint != "http://127.0.0.1:8695" {
		t.Fatal("worker binding leaked")
	}
}
