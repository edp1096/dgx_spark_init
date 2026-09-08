package orchestrator

import (
	"bytes"
	"fmt"
	"gopkg.in/yaml.v3"
	"os"
	"strings"
	"testing"
)

func TestEmbeddedCatalogIsComplete(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"qwen27-exl3", "flash-next", "gemma"} {
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

func TestEXL3RuntimeProfileStaysInSync(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	bundle, ok := catalog.Bundle("qwen27-exl3")
	if !ok || bundle.ContextTokens != 131072 || bundle.ModelType != "qwen3.8-exl3" {
		t.Fatalf("unexpected EXL3 bundle: %+v", bundle)
	}
	component, ok := catalog.Component("qwen27-exl3")
	if !ok || component.MemoryGiB != 32 {
		t.Fatalf("unexpected EXL3 component: %+v", component)
	}
	data, err := composeAsset(component.ComposeAsset)
	if err != nil {
		t.Fatal(err)
	}
	var recipe map[string]any
	if err := yaml.Unmarshal(data, &recipe); err != nil {
		t.Fatal(err)
	}
	service := recipe["services"].(map[string]any)["runtime"].(map[string]any)
	command := service["command"].([]any)
	flags := map[string]string{}
	for i := 0; i+1 < len(command); i++ {
		if flag, ok := command[i].(string); ok && strings.HasPrefix(flag, "--") {
			flags[flag] = fmt.Sprint(command[i+1])
		}
	}
	for flag, expected := range map[string]string{"--cache_size": "131072", "--draft_model": "mtp", "--cache_quant": "nvfp4"} {
		if flags[flag] != expected {
			t.Fatalf("%s=%s want %s", flag, flags[flag], expected)
		}
	}
	if embeddedBuildAsset(component.ComposeAsset) == "" {
		t.Fatal("embedded build missing")
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
