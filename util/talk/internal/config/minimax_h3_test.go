package config

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"sparktalk/internal/orchestrator"
)

func legacyQwen38FNEXL3Catalog(t *testing.T) orchestrator.Catalog {
	t.Helper()
	catalog, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for i := range catalog.Bundles {
		b := &catalog.Bundles[i]
		if b.ID != "qwen38fn_exl3" {
			continue
		}
		for j, id := range b.Components {
			if id == "qwim-mmh3" {
				b.Components[j] = "flux2"
			}
		}
		keep, budget := true, 24.0
		b.Bindings["flux2"] = orchestrator.Deployment{KeepResident: &keep, MemoryGiB: &budget, StartAfterLLM: &keep}
	}
	return catalog
}

func TestMiniMaxH3ReplacesLegacyEXL3InPlace(t *testing.T) {
	catalog := legacyQwen38FNEXL3Catalog(t)
	originalIndex := -1
	for i, b := range catalog.Bundles {
		if b.ID == "qwen38fn_exl3" {
			originalIndex = i
		}
	}
	components := []orchestrator.Component{}
	for _, x := range catalog.Components {
		if x.ID != "qwim-mmh3" {
			components = append(components, x)
		}
	}
	catalog.Components = components
	c := Config{Runtime: RuntimeConfig{Mode: "managed", BuiltinRevision: 29, Bundle: "qwen38fn_exl3", ActiveBundle: "qwen38fn_exl3", Catalog: &catalog}}
	c.Normalize()
	b := c.Runtime.Catalog.Bundles[originalIndex]
	if b.ID != "qwen38fn_exl3" || b.Name != "Qwen 3.8 Flash-Next EXL3 3bit" {
		t.Fatalf("EXL3 entry moved or renamed incorrectly: %+v", b)
	}
	if _, ok := c.Runtime.Catalog.Bundle("qwen38fn_exl3-mmh3"); ok {
		t.Fatal("duplicate MMH3 set remains")
	}
	if _, ok := c.Runtime.Catalog.ResolveComponent(b.ID, "flux2"); ok {
		t.Fatal("former EXL3 image engine remains in set")
	}
	x, ok := c.Runtime.Catalog.ResolveComponent(b.ID, "qwim-mmh3")
	if !ok || !x.KeepResident || x.MemoryGiB != 28 || c.Image.Endpoint != x.Endpoint || c.Image.Mode != "basic" {
		t.Fatalf("joint runtime not applied: %+v", x)
	}
	if c.ASR.Endpoint != "http://127.0.0.1:8693" {
		t.Fatal("ASR endpoint changed")
	}
}

func TestMiniMaxH3ConsolidatesSelectedSetAndPreservesDeployments(t *testing.T) {
	for _, selected := range []string{"qwen38fn_exl3", "qwen38fn_exl3-mmh3"} {
		t.Run(selected, func(t *testing.T) {
			catalog := legacyQwen38FNEXL3Catalog(t)
			defaults, _ := orchestrator.LoadCatalog()
			paired, _ := defaults.Bundle("qwen38fn_exl3")
			paired.ID, paired.Name, paired.Description, paired.ContextTokens = "qwen38fn_exl3-mmh3", "Qwen 3.8 Flash-Next EXL3 + MMH3", "내 영상 세트 설정", 262144
			endpoint, health := "http://127.0.0.1:8731", "http://127.0.0.1:8731/health"
			paired.Bindings["qwim-mmh3"] = orchestrator.Deployment{Endpoint: &endpoint, HealthURL: &health}
			catalog.Bundles = append(catalog.Bundles, paired)
			unrelated := map[string]string{}
			originalIndex := -1
			for i, b := range catalog.Bundles {
				if b.ID == "qwen38fn_exl3" {
					originalIndex = i
				} else if b.ID != paired.ID {
					raw, _ := json.Marshal(b)
					unrelated[b.ID] = string(raw)
				}
			}
			c := Config{Runtime: RuntimeConfig{Mode: "managed", BuiltinRevision: 31, Bundle: selected, ActiveBundle: selected, Catalog: &catalog}}
			c.Normalize()
			if c.Runtime.Bundle != "qwen38fn_exl3" || c.Runtime.ActiveBundle != "qwen38fn_exl3" {
				t.Fatal("selected/default set not migrated")
			}
			b := c.Runtime.Catalog.Bundles[originalIndex]
			if b.ID != "qwen38fn_exl3" || b.Name != "Qwen 3.8 Flash-Next EXL3 3bit" || b.Description != paired.Description || b.ContextTokens != paired.ContextTokens {
				t.Fatalf("paired set configuration or position lost: %+v", b)
			}
			if c.Image.Endpoint != "http://127.0.0.1:8731" || c.Image.Mode != "basic" {
				t.Fatal("paired deployment lost")
			}
			if len(c.Runtime.Catalog.Bundles) != len(unrelated)+1 {
				t.Fatal("duplicate set remains")
			}
			for _, b := range c.Runtime.Catalog.Bundles {
				if previous, ok := unrelated[b.ID]; ok {
					raw, _ := json.Marshal(b)
					if string(raw) != previous {
						t.Fatalf("unrelated set changed: %s", b.ID)
					}
				}
			}
			before, _ := json.Marshal(c)
			c.Normalize()
			after, _ := json.Marshal(c)
			if string(before) != string(after) {
				t.Fatal("migration not idempotent")
			}
		})
	}
}

func TestMiniMaxH3PortMigrationPreservesASR(t *testing.T) {
	c, _, err := Load(filepath.Join(t.TempDir(), "sparktalk.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	for i := range c.Runtime.Catalog.Bundles {
		b := &c.Runtime.Catalog.Bundles[i]
		if b.ID == "qwen38fn_exl3" {
			b.ID = "qwen38fn_exl3-mmh3"
		}
	}
	c.Runtime.ActiveBundle = "qwen38fn_exl3-mmh3"
	c.Runtime.Bundle = "qwen38fn_exl3-mmh3"
	c.Runtime.BuiltinRevision = 30
	for i := range c.Runtime.Catalog.Components {
		x := &c.Runtime.Catalog.Components[i]
		if x.ID == "qwim-mmh3" {
			x.Endpoint = "http://127.0.0.1:8693"
			x.HealthURL = x.Endpoint + "/health"
		}
	}
	c.Normalize()
	x, _ := c.Runtime.Catalog.ResolveComponent("qwen38fn_exl3", "qwim-mmh3")
	if x.Endpoint != "http://127.0.0.1:8730" || x.HealthURL != x.Endpoint+"/health" || c.Image.Endpoint != x.Endpoint {
		t.Fatalf("joint runtime port migration failed: %+v, image=%s", x, c.Image.Endpoint)
	}
	if c.ASR.Endpoint != "http://127.0.0.1:8693" {
		t.Fatalf("ASR port changed: %s", c.ASR.Endpoint)
	}
}
