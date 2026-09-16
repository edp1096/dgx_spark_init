package config

import (
	"bytes"
	"os"
	"reflect"
	"testing"

	"gopkg.in/yaml.v3"
)

// Opt-in, read-only verification against the operator's current saved settings.
func TestGemmaCheckpointMigrationSavedConfig(t *testing.T) {
	path := os.Getenv("SPARKTALK_CHECK_CONFIG")
	if path == "" {
		t.Skip("set SPARKTALK_CHECK_CONFIG for read-only saved-config verification")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var before Config
	if err := yaml.Unmarshal(raw, &before); err != nil {
		t.Fatal(err)
	}
	after, created, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if created {
		t.Fatal("existing config unexpectedly replaced")
	}
	expectedModel := before.Model.DefaultModel
	if before.Runtime.Mode == "managed" && (expectedModel == "sakamakismile/Huihui-gemma-4-26B-A4B-it-qat-abliterated-MTP-NVFP4" || expectedModel == "coolthor/Huihui-gemma-4-26B-A4B-it-abliterated-FP8-Dynamic") {
		expectedModel = "edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4"
	}
	if before.Runtime.Bundle != after.Runtime.Bundle || before.Runtime.ActiveBundle != after.Runtime.ActiveBundle || expectedModel != after.Model.DefaultModel {
		t.Fatal("active selection changed")
	}
	if before.ASR.Enabled != after.ASR.Enabled || before.TTS.Enabled != after.TTS.Enabled || before.TTS.AutoPlay != after.TTS.AutoPlay {
		t.Fatal("voice preferences changed")
	}
	if len(before.Runtime.Catalog.Bundles) != len(after.Runtime.Catalog.Bundles) {
		t.Fatal("bundle count changed")
	}
	for i, b := range before.Runtime.Catalog.Bundles {
		x := after.Runtime.Catalog.Bundles[i]
		if b.ID != x.ID || b.Name != x.Name {
			t.Fatal("saved bundle names/order changed")
		}
	}
	for _, b := range before.Runtime.Catalog.Components {
		x, ok := after.Runtime.Catalog.Component(b.ID)
		if !ok || b.Name != x.Name || b.Host != x.Host || !reflect.DeepEqual(b.RuntimeOptions, x.RuntimeOptions) {
			t.Fatalf("saved component configuration changed: %s", b.ID)
		}
	}
	c, ok := after.Runtime.Catalog.Component("gemma26")
	if !ok || c.Model != "edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4" {
		t.Fatal("Gemma model identity not migrated")
	}
	unchanged, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, unchanged) {
		t.Fatal("read-only verification modified config")
	}
}
