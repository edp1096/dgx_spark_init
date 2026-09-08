package orchestrator

import (
	"strings"
	"testing"
)

func TestExecutionHostPathsDoNotFallBackToLocalAccount(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	cat.Hosts["third"] = Host{Address: "192.0.2.2", DataDir: "/srv/sparktalk", ModelCache: "/mnt/models/huggingface"}
	c := newController(cat)
	c.ConfigurePaths("/tmp/local-app", "/tmp/local-hf")
	component := Component{Host: "third", ComposeAsset: "compose.magpie-tts.yaml"}
	data, cache, err := c.runtimeHostPaths(component)
	if err != nil {
		t.Fatal(err)
	}
	if data != "/srv/sparktalk" || cache != "/mnt/models/huggingface" {
		t.Fatalf("wrong host paths: %s %s", data, cache)
	}
	env := strings.Join(runtimePathEnvironment(component, data, cache), "\n")
	if !strings.Contains(env, "SPARKTALK_MAGPIE_MODEL_DIR=/mnt/models/nemo-speech/magpie-v2607") {
		t.Fatal(env)
	}
	component.ComposeAsset = "compose.nemotron-asr.yaml"
	if !strings.Contains(strings.Join(runtimePathEnvironment(component, data, cache), "\n"), "SPARKTALK_NEMO_MODEL_DIR=/mnt/models/nemo-speech") {
		t.Fatal("ASR ignores host model cache")
	}
}
