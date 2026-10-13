package config

import (
	"bytes"
	"fmt"
	"gopkg.in/yaml.v3"
	"os"
	"path/filepath"
	"reflect"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestSavedGemmaRetirementPreservesQwenAndSharedServices(t *testing.T) {
	for _, revision := range []int{8, 40, 41, 42} {
		t.Run(fmt.Sprint(revision), func(t *testing.T) {
			c := Config{Runtime: RuntimeConfig{Mode: "managed", Bundle: "flash-next-radixark", ActiveBundle: "qwen38fn_exl3"}, Model: ModelConfig{ReasoningEffort: "xhigh"}}
			c.Normalize()
			model, context, extra := c.Model, c.Context, c.Extra
			components := append([]orchestrator.Component(nil), c.Runtime.Catalog.Components...)
			bundles := append([]orchestrator.Bundle(nil), c.Runtime.Catalog.Bundles...)
			c.Runtime.BuiltinRevision = revision
			for _, id := range []string{"gemma26", "gemma31"} {
				c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, orchestrator.Component{ID: id, Role: "llm", ComposeAsset: "compose." + id + ".yaml"})
				c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, orchestrator.Bundle{ID: id, Components: []string{id, "extra-ssh", "extra-embedding"}})
			}
			raw, err := yaml.Marshal(c)
			if err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(t.TempDir(), "sparktalk.yaml")
			if err = os.WriteFile(path, raw, 0600); err != nil {
				t.Fatal(err)
			}
			loaded, created, err := Load(path)
			if err != nil || created {
				t.Fatal("saved catalog failed", err)
			}
			if loaded.Runtime.Bundle != c.Runtime.Bundle || loaded.Runtime.ActiveBundle != c.Runtime.ActiveBundle || !reflect.DeepEqual(model, loaded.Model) || !reflect.DeepEqual(context, loaded.Context) || !reflect.DeepEqual(extra, loaded.Extra) {
				t.Fatal("Qwen selection or preferences changed")
			}
			if !reflect.DeepEqual(components, loaded.Runtime.Catalog.Components) || !reflect.DeepEqual(bundles, loaded.Runtime.Catalog.Bundles) {
				t.Fatal("surviving models/services changed")
			}
			unchanged, _ := os.ReadFile(path)
			if !bytes.Equal(raw, unchanged) {
				t.Fatal("read-only migration wrote settings")
			}
			if err = Save(path, loaded); err != nil {
				t.Fatal(err)
			}
			again, _, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(loaded, again) {
				t.Fatal("retirement did not persist idempotently")
			}
		})
	}
}

func TestActualSavedGemmaRetirement(t *testing.T) {
	path := os.Getenv("SPARKTALK_CHECK_CONFIG")
	if path == "" {
		t.Skip("optional read-only production configuration check")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var before Config
	if err = yaml.Unmarshal(raw, &before); err != nil {
		t.Fatal(err)
	}
	after, created, err := Load(path)
	if err != nil || created {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before.Model, after.Model) || !reflect.DeepEqual(before.Context, after.Context) || !reflect.DeepEqual(before.Extra, after.Extra) || before.Runtime.Bundle != after.Runtime.Bundle || before.Runtime.ActiveBundle != after.Runtime.ActiveBundle {
		t.Fatal("active configuration changed during retirement")
	}
	for _, c := range after.Runtime.Catalog.Components {
		if c.ID == "gemma26" || c.ID == "gemma31" {
			t.Fatal("retired Gemma remained")
		}
	}
	if !bytes.Equal(raw, mustReadConfig(t, path)) {
		t.Fatal("read-only check changed production settings")
	}
	t.Logf("preserved active %s; %d bundles, %d components", after.Model.DefaultModel, len(after.Runtime.Catalog.Bundles), len(after.Runtime.Catalog.Components))
}
func mustReadConfig(t *testing.T, path string) []byte {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
