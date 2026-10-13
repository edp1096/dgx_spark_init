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
	if cfg.Model.ReasoningEffort != "low" {
		t.Fatal("new installation did not default to low")
	}
	startup := cfg.Runtime.Bundle
	for _, id := range []string{"qwen38fn_exl3", "qwen38fn_exl3_q4", "flash-next-radixark", "flash-next", "flash-next-tp2"} {
		t.Run(id, func(t *testing.T) {
			cfg.SelectManagedBundle("glm53-worker-extra")
			cfg.Model.ReasoningEffort = "max"
			cfg.SelectManagedBundle(id)
			if cfg.Model.ReasoningEffort != "low" || cfg.Runtime.Bundle != startup {
				t.Fatal("set change inherited another model's effort or changed startup selection")
			}
			if err := Save(path, cfg); err != nil {
				t.Fatal(err)
			}
			reloaded, _, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if reloaded.Runtime.ActiveBundle != id || reloaded.Model.ReasoningEffort != "low" {
				t.Fatal("saved default did not survive restart")
			}
			for _, explicit := range []string{"none", "medium", "xhigh"} {
				reloaded.Model.ReasoningEffort = explicit
				reloaded.SelectManagedBundle(id)
				if reloaded.Model.ReasoningEffort != explicit {
					t.Fatal("restarting the same set overwrote explicit effort", explicit)
				}
			}
			for _, missing := range []string{"", "max", "unknown"} {
				if normalizeReasoningEffort(cfg.Model.ModelType, missing) != "low" || llm.NormalizeReasoningEffort(cfg.Model.ModelType, missing) != "low" {
					t.Fatal("configuration and actual request defaults differ", missing)
				}
			}
		})
	}
}
