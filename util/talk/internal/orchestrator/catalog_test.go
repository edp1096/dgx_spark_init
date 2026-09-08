package orchestrator

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestEmbeddedCatalogIsComplete(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"flash-next", "gemma"} {
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
	for _, required := range []string{"dgx-sglang-qwen38-fn:sm121", "container_name: sglang-qwen38-fn"} {
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

func TestGemmaRuntimeProfileStaysInSync(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	bundle, ok := catalog.Bundle("gemma")
	if !ok || bundle.ContextTokens != 65536 {
		t.Fatalf("unexpected Gemma context: %+v", bundle)
	}
	component, ok := catalog.Component("gemma31")
	if !ok || component.Container != "sglang-gemma4-31b" {
		t.Fatalf("unexpected gemma31 component: %+v", component)
	}
	data, err := composeAsset(component.ComposeAsset)
	if err != nil {
		t.Fatal(err)
	}
	compose := string(data)
	for _, required := range []string{
		"dgx-sglang-gemma4-31b-dflash:2ef0fe4-toolindex1",
		"SPARKTALK_BUILD_DIR",
		"dockerfile: Dockerfile.dflash",
		"container_name: sglang-gemma4-31b",
		"/opt/gemma4/chat_template.jinja",
		"eabd648301ce28583cc14757912e5e0f84e152e1",
		"--speculative-draft-kv-cache-dtype",
		"--num-continuous-decode-steps",
	} {
		if !strings.Contains(compose, required) {
			t.Fatalf("Gemma compose is missing %q", required)
		}
	}
}

func TestGemmaBuildAssetsAreEmbeddedAndMatchStandalone(t *testing.T) {
	for _, name := range []string{"Dockerfile.dflash", "patch_tool_index.py", "chat_template.jinja"} {
		embedded, err := assets.ReadFile("assets/gemma31/" + name)
		if err != nil {
			t.Fatal(err)
		}
		standalone, err := os.ReadFile("../../../../compose_yaml/gemma4_31b_sglang/" + name)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(embedded, standalone) {
			t.Fatalf("Gemma build asset out of sync: %s", name)
		}
	}
}
