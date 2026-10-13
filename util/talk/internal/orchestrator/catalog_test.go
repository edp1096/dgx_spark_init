package orchestrator

import (
	"strings"
	"testing"
)

func TestEmbeddedCatalogIsComplete(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"flash-next", "qwen38fn_exl3", "qwen38fn_exl3_q4", "flash-next-radixark"} {
		bundle, ok := catalog.Bundle(id)
		if !ok || bundle.MemoryGiB <= 0 || bundle.ModelID == "" {
			t.Fatalf("invalid bundle %q: %+v", id, bundle)
		}
		for _, componentID := range bundle.Components {
			component, ok := catalog.Component(componentID)
			if !ok {
				t.Fatalf("missing component %q", componentID)
			}
			if _, err := composeAsset(component.ComposeAsset); err != nil {
				t.Fatalf("missing compose asset for %q: %v", componentID, err)
			}
		}
	}
}

func TestFlashNextRuntimeProfileUsesShortLocalName(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	component, ok := catalog.Component("flash-next")
	if !ok || component.Container != "sglang-qwen38-fn" {
		t.Fatalf("unexpected Flash-Next component: %+v", component)
	}
	data, err := composeAsset(component.ComposeAsset)
	if err != nil {
		t.Fatal(err)
	}
	compose := string(data)
	for _, required := range []string{"dgx-sglang-qwen38-qad:sm121", "container_name: sglang-qwen38-fn"} {
		if !strings.Contains(compose, required) {
			t.Fatalf("Flash-Next compose is missing %q", required)
		}
	}
	if strings.Contains(compose, "qwen38-flash-next") {
		t.Fatal("Flash-Next compose still uses the long local runtime name")
	}
}

func TestRetired27BEXL3IsNotEmbedded(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := catalog.Component("qwen27-exl3"); ok {
		t.Fatal("retired component survived")
	}
	if _, ok := catalog.Bundle("qwen27-exl3"); ok {
		t.Fatal("retired bundle survived")
	}
	for _, path := range []string{"assets/compose.qwen27-exl3.yaml", "assets/qwen27-exl3/Dockerfile", "assets/recipes/qwen27-exl3.tar.gz"} {
		if _, err := assets.ReadFile(path); err == nil {
			t.Fatalf("retired asset still embedded: %s", path)
		}
	}
	component := Component{ComposeAsset: "compose.qwen27-exl3.yaml"}
	if recipeID(component) != "" || embeddedBuildAsset(component.ComposeAsset) != "" {
		t.Fatal("retired preparation recipe survived")
	}
}

func TestRetiredGemmaRuntimesHaveNoEmbeddedAssets(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"gemma", "gemma26", "gemma31"} {
		if _, ok := catalog.Component(id); ok {
			t.Fatalf("retired component: %s", id)
		}
		if _, ok := catalog.Bundle(id); ok {
			t.Fatalf("retired bundle: %s", id)
		}
	}
	for _, asset := range []string{"compose.gemma26.yaml", "compose.gemma31.yaml"} {
		if _, err := composeAsset(asset); err == nil {
			t.Fatalf("retired compose still available: %s", asset)
		}
		if embeddedBuildAsset(asset) != "" || len(componentModelAssets(Component{ComposeAsset: asset})) != 0 {
			t.Fatalf("retired preparation: %s", asset)
		}
	}
	for _, path := range []string{"assets/gemma26/Dockerfile", "assets/gemma31/Dockerfile.exl3", "assets/gemma31/launch_exl3.py"} {
		if _, err := assets.ReadFile(path); err == nil {
			t.Fatalf("retired build input embedded: %s", path)
		}
	}
	if _, ok := catalog.Component("extra-embedding"); !ok {
		t.Fatal("EmbeddingGemma removed with LLM sets")
	}
}
