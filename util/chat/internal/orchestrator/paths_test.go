package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEXL3PreparationAndLaunchShareHostPaths(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	c := newController(cat)
	data := t.TempDir()
	cache := filepath.Join(t.TempDir(), "custom-hf")
	c.ConfigurePaths(data, cache)
	for _, id := range []string{"qwen27-exl3"} {
		component, _ := cat.Component(id)
		dir, err := c.materializeRecipe(context.Background(), component)
		if err != nil {
			t.Fatal(err)
		}
		prepared, err := os.ReadFile(filepath.Join(dir, ".env"))
		if err != nil {
			t.Fatal(err)
		}
		env := runtimePathEnvironment(component, data, cache)
		for _, entry := range env {
			pair := strings.SplitN(entry, "=", 2)
			if len(pair) != 2 {
				continue
			}
			key := ""
			switch pair[0] {
			case "SPARKTALK_EXL3_CACHE":
				key = "EXL3_CACHE_PATH"
			case "SPARKTALK_EXL3_QWEN38_FN_ABLIT":
				key = "ABLIT_OUTPUT_PATH"
			}
			if key != "" && !strings.Contains(string(prepared), key+"="+shellQuote(pair[1])) {
				t.Fatalf("%s launch and preparation disagree: %s", id, entry)
			}
		}
	}
}

func TestExecutionHostPathsDoNotFallBackToLocalAccount(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	cat.Hosts["third"] = Host{Address: "192.0.2.2", DataDir: "/srv/sparktalk", ModelCache: "/mnt/models/huggingface"}
	c := newController(cat)
	c.ConfigurePaths("/tmp/local-app", "/tmp/local-hf")
	component := Component{Host: "third", ComposeAsset: "compose.qwen27-exl3.yaml"}
	data, cache, err := c.runtimeHostPaths(component)
	if err != nil {
		t.Fatal(err)
	}
	if data != "/srv/sparktalk" || cache != "/mnt/models/huggingface" {
		t.Fatalf("wrong host paths: %s %s", data, cache)
	}
	env := strings.Join(runtimePathEnvironment(component, data, cache), "\n")
	if !strings.Contains(env, "SPARKTALK_EXL3_CACHE=/mnt/models/exl3-qwen38-27b") {
		t.Fatal(env)
	}
	component.ComposeAsset = "compose.nemotron-asr.yaml"
	if !strings.Contains(strings.Join(runtimePathEnvironment(component, data, cache), "\n"), "SPARKTALK_NEMO_MODEL_DIR=/mnt/models/nemo-speech") {
		t.Fatal("ASR ignores host model cache")
	}
}
