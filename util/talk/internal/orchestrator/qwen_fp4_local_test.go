package orchestrator

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestFP4LocalUsesSeparatePinnedRuntimeAndLocalWeights(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	b, ok := cat.Bundle("flash-next-fp4-local")
	if !ok || b.ModelID != QwenFP4Local || b.ModelType != "qwen3.8-vllm" || b.ContextTokens != 1048576 {
		t.Fatalf("incorrect FP4-Local profile: %+v", b)
	}
	x, ok := cat.ResolveComponent(b.ID, b.ID)
	if !ok || x.ComposeAsset != "compose.qwen-fp4-local.yaml" || x.Port != 18004 || x.StartupMemoryGiB < 115.4 || x.ProgressKind != "b12x" {
		t.Fatalf("incorrect deployment: %+v", x)
	}
	if x.ModelPresentation == nil || x.ModelPresentation.Weights[0].Format != "FP4-Local" {
		t.Fatal("missing actual weight format")
	}
	changed := x
	changed.RuntimeOptions = map[string]string{"MAX_MODEL_LEN": "262144"}
	if validateRecipeOptions(changed) == nil {
		t.Fatal("runtime overrides must not silently differ from the fixed compose profile")
	}
	changed.RuntimeOptions = map[string]string{"MODEL_VARIANT": "abliterated"}
	if err := validateRecipeOptions(changed); err != nil {
		t.Fatal("the model preparation UI must accept the one supported variant", err)
	}
	if !b.WorkloadSwap {
		t.Fatal("auxiliary models must load on demand after the LLM")
	}
	for _, member := range []string{"nemotron-asr", "qwen3-tts", "extra-embedding"} {
		aux, ok := cat.ResolveComponent(b.ID, member)
		if !ok || aux.KeepResident || !aux.StartAfterLLM {
			t.Fatal("auxiliary cold startup overlaps the LLM", member, aux)
		}
	}
	items := componentModelAssets(x)
	if len(items) != 1 || !items[0].LocalOnly || items[0].Repo != QwenFP4Local || items[0].Revision != "" {
		t.Fatalf("unpublished weights must remain local: %+v", items)
	}
	if items[0].SHA256["qualification/build-transfer-manifest.json"] != "a8964de5b214ef02fe55091ee5952a126f75d96b66bc37ad30f90863b5caa075" {
		t.Fatal("checkpoint identity is not the verified build")
	}
	qad, _ := cat.ResolveComponent("flash-next", "flash-next")
	if qad.ComposeAsset != "compose.flash-next.yaml" || qad.Model == QwenFP4Local {
		t.Fatal("existing QAD recipe was replaced")
	}
}

func TestFP4LocalComposeMatchesStandaloneAndQualifiedRoPE(t *testing.T) {
	raw, err := assets.ReadFile("assets/compose.qwen-fp4-local.yaml")
	if err != nil {
		t.Fatal(err)
	}
	standalone, err := os.ReadFile(filepath.Join("../../../../compose_yaml/qwen38_fn_fp4_local_vllm/compose.yaml"))
	if err != nil || !bytes.Equal(raw, standalone) {
		t.Fatal("managed and standalone compose differ", err)
	}
	var doc map[string]any
	if err := yaml.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	s := doc["services"].(map[string]any)["runtime"].(map[string]any)
	if s["image"] != "local/qwen38-fp4-local:csf-yarn-20261016" || s["pull_policy"] != "never" {
		t.Fatal("wrong runtime image")
	}
	if s["network_mode"] != "bridge" {
		t.Fatal("single-service runtime should reuse the qualified Docker bridge without allocating another subnet")
	}
	command := s["command"].([]any)
	args := map[string]string{}
	for i := 0; i+1 < len(command); i++ {
		flag, _ := command[i].(string)
		if strings.HasPrefix(flag, "--") {
			args[flag], _ = command[i+1].(string)
		}
	}
	for flag, want := range map[string]string{"--served-model-name": QwenFP4Local, "--max-model-len": "1048576", "--quantization": "modelopt_mixed", "--load-format": "b12x", "--kv-cache-dtype": "fp8", "--gpu-memory-utilization": "0.8", "--max-num-seqs": "1"} {
		if args[flag] != want {
			t.Errorf("%s = %q, want %q", flag, args[flag], want)
		}
	}
	var speculative map[string]any
	if err := json.Unmarshal([]byte(args["--speculative-config"]), &speculative); err != nil || speculative["method"] != "mtp" || speculative["num_speculative_tokens"] != float64(4) {
		t.Fatal("MTP configuration changed", err)
	}
	config, err := assets.ReadFile("assets/qwen-fp4-local/runtime-config.json")
	if err != nil {
		t.Fatal(err)
	}
	hash := sha256.Sum256(config)
	// This config is the identical overlay used for the completed 1M tests.
	want, err := os.ReadFile(filepath.Join("../../../../compose_yaml/qwen38_fn_fp4_local_vllm/runtime-config.json"))
	if err != nil || !bytes.Equal(config, want) || hex.EncodeToString(hash[:]) != "e80038e1bf8879e3d022c8c68c33fe1105e521849230213e253e83a9856a26e8" {
		t.Fatal("runtime overlay drift", err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(config, &cfg); err != nil {
		t.Fatal(err)
	}
	text := cfg["text_config"].(map[string]any)
	rope := text["rope_parameters"].(map[string]any)
	if text["max_position_embeddings"] != float64(1048576) || rope["rope_type"] != "yarn" || rope["factor"] != float64(4) || rope["original_max_position_embeddings"] != float64(262144) {
		t.Fatal("wrong YaRN settings")
	}
}

func TestB12XProgressFollowsActualLoaderAndKernelStages(t *testing.T) {
	logs := "b12x routing checkpoint shards: 9/21 shards routed, 43.58 GB selected on rank 0, 1:10\n"
	if p := inferB12XProgress(logs); p.Key != "b12x-shards" || !strings.Contains(p.Detail, "9/21") {
		t.Fatal(p)
	}
	logs += "Loading weights took 123.53 seconds\nb12x compiling moe.decode: 42/85 ready, 0 measured, 77 cached, 4 compilations, 0:06\n"
	if p := inferB12XProgress(logs); p.Key != "b12x-weights" {
		t.Fatal(p)
	}
	logs += "GPU KV cache size: 1,390,603 tokens\nb12x priming sequence.gdn_prefill: 6/81 ready, 0 measured, 9 cached, 0 compilations, 0:00\n"
	if p := inferB12XProgress(logs); p.Key != "b12x-state" {
		t.Fatal(p)
	}
	logs += "JIT kernel warmup starting.\n"
	if p := inferB12XProgress(logs); p.Key != "b12x-warmup" {
		t.Fatal(p)
	}
	logs += "Application startup complete.\n"
	if p := inferB12XProgress(logs); p.Progress != 1 {
		t.Fatal(p)
	}
}
