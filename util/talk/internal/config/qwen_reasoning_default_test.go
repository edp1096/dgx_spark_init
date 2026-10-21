package config

import (
	"path/filepath"
	"testing"

	"sparktalk/internal/llm"
)

func TestQwenReasoningDefaultSurvivesBundleSwitchSaveAndReload(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sparktalk.yaml")
	cfg, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Model.ReasoningEffort != "medium" {
		t.Fatal("new installation did not default to medium")
	}
	startup := cfg.Runtime.Bundle
	for _, id := range []string{"qwen38fn_exl3", "flash-next-fp4-local", "flash-next", "flash-next-tp2"} {
		t.Run(id, func(t *testing.T) {
			cfg.SelectManagedBundle("glm53-worker-extra")
			cfg.Model.ReasoningEffort = "max"
			cfg.SelectManagedBundle(id)
			if cfg.Model.ReasoningEffort != "medium" || cfg.Runtime.Bundle != startup {
				t.Fatal("set change inherited another model's effort or changed startup selection")
			}
			if err := Save(path, cfg); err != nil {
				t.Fatal(err)
			}
			reloaded, _, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if reloaded.Runtime.ActiveBundle != id || reloaded.Model.ReasoningEffort != "medium" {
				t.Fatal("saved default did not survive restart")
			}
			for _, explicit := range []string{"none", "low", "medium", "xhigh"} {
				reloaded.Model.ReasoningEffort = explicit
				reloaded.SelectManagedBundle(id)
				if reloaded.Model.ReasoningEffort != explicit {
					t.Fatal("restarting the same set overwrote explicit effort", explicit)
				}
			}
			for _, missing := range []string{"", "max", "unknown"} {
				if normalizeReasoningEffort(cfg.Model.ModelType, missing) != "medium" || llm.NormalizeReasoningEffort(cfg.Model.ModelType, missing) != "medium" {
					t.Fatal("configuration and actual request defaults differ", missing)
				}
			}
		})
	}
}
