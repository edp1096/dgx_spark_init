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
	for _, id := range []string{"qwen27", "qwen27-exl3", "flash-next", "flash-next-exl3", "gemma"} {
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

func TestFlashNextEXL3RuntimeProfileStaysInSync(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	bundle, ok := catalog.Bundle("flash-next-exl3")
	if !ok || bundle.ContextTokens != 262144 || bundle.ModelType != "qwen3.8" {
		t.Fatalf("unexpected Flash-Next EXL3 bundle: %+v", bundle)
	}
	component, ok := catalog.Component("flash-next-exl3")
	if !ok || component.ProgressKind != "exl3" {
		t.Fatalf("unexpected Flash-Next EXL3 component: %+v", component)
	}
	data, err := composeAsset(component.ComposeAsset)
	if err != nil {
		t.Fatal(err)
	}
	compose := string(data)
	for _, required := range []string{
		"dgx-exl3-qwen38fn:1.4.6-ablit1",
		"Qwen3.8-Flash-Next-Abliterated-EXL3-4.05bpw",
		"exl3-qwen38fn-4.05bpw",
		"direction.safetensors",
		"--cache_size\n      - \"262144\"",
	} {
		if !strings.Contains(compose, required) {
			t.Fatalf("Flash-Next EXL3 compose is missing %q", required)
		}
	}
	if strings.Contains(compose, "--cache_quant") {
		t.Fatal("Flash-Next QSA requires the fp16 cache path")
	}
}

func TestFlashNextRuntimeProfileUsesShortLocalName(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	component, ok := catalog.Component("flash-next")
	if !ok || component.Container != "sglang-qwen38fn" {
		t.Fatalf("unexpected Flash-Next component: %+v", component)
	}
	data, err := composeAsset(component.ComposeAsset)
	if err != nil {
		t.Fatal(err)
	}
	compose := string(data)
	for _, required := range []string{"dgx-sglang-qwen38fn:sm121", "container_name: sglang-qwen38fn"} {
		if !strings.Contains(compose, required) {
			t.Fatalf("Flash-Next compose is missing %q", required)
		}
	}
	if strings.Contains(compose, "qwen38-flash-next") {
		t.Fatal("Flash-Next compose still uses the long local runtime name")
	}
}

func TestQwen27RuntimeProfileUsesSizedLocalName(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	component, ok := catalog.Component("qwen27")
	if !ok || component.Container != "sglang-qwen38-27b" {
		t.Fatalf("unexpected Qwen 27B component: %+v", component)
	}
	data, err := composeAsset(component.ComposeAsset)
	if err != nil {
		t.Fatal(err)
	}
	compose := string(data)
	for _, required := range []string{"dgx-sglang-qwen38-27b-dflash2:2ef0fe4", "container_name: sglang-qwen38-27b"} {
		if !strings.Contains(compose, required) {
			t.Fatalf("Qwen 27B compose is missing %q", required)
		}
	}
}

func TestEXL3RuntimeProfileStaysInSync(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	bundle, ok := catalog.Bundle("qwen27-exl3")
	if !ok || bundle.ContextTokens != 262144 || bundle.ModelType != "qwen3.8-exl3" {
		t.Fatalf("unexpected EXL3 bundle: %+v", bundle)
	}
	component, ok := catalog.Component("qwen27-exl3")
	if !ok || component.MemoryGiB != 25 {
		t.Fatalf("unexpected EXL3 component: %+v", component)
	}
	data, err := composeAsset(component.ComposeAsset)
	if err != nil {
		t.Fatal(err)
	}
	compose := string(data)
	for _, required := range []string{
		"dgx-exl3-qwen38-27b:63b32f0",
		"Qwen3.8-27B-Uncensored-EXL3-4bpw",
		"exl3-qwen38-27b-uncensored-4bpw",
		"--draft_model\n      - mtp",
		"--cache_size\n      - \"262144\"",
		"--cache_quant\n      - nvfp4",
	} {
		if !strings.Contains(compose, required) {
			t.Fatalf("EXL3 compose is missing %q", required)
		}
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
		"dgx-sglang-gemma4-31b-dflash:2ef0fe4",
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
