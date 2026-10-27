package orchestrator

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmbeddedRecipesContainNoPrivateEnvironment(t *testing.T) {
	for _, id := range []string{"glm53", "glm53-fp4", "ds41", "qwen38-tp2"} {
		data, err := assets.ReadFile("assets/recipes/" + id + ".tar.gz")
		if err != nil {
			t.Fatal(err)
		}
		gz, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		tr := tar.NewReader(gz)
		found := map[string]bool{}
		for {
			h, err := tr.Next()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatal(err)
			}
			if filepath.IsAbs(h.Name) || strings.Contains(h.Name, "../") || strings.HasPrefix(filepath.Base(h.Name), ".env") {
				t.Fatalf("private/unsafe archive member: %s", h.Name)
			}
			found[h.Name] = true
		}
		gz.Close()
		for _, name := range []string{"manage.sh", "runtime.sh", "models.sh", "env.sample"} {
			if !found[name] {
				t.Fatalf("%s lacks %s", id, name)
			}
		}
	}
}
func TestEmbeddedRecipeMaterializesInAppDataDirectory(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	c := newController(cat)
	data, cache := t.TempDir(), t.TempDir()
	c.ConfigurePaths(data, cache)
	for _, id := range []string{"glm53", "ds41"} {
		component, ok := cat.Component(id)
		if !ok {
			t.Fatal(id)
		}
		dir, err := c.materializeRecipe(context.Background(), component)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.HasPrefix(dir, data+string(os.PathSeparator)) {
			t.Fatal(dir)
		}
		env, err := os.ReadFile(filepath.Join(dir, ".env"))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(env), "HF_CACHE="+shellQuote(cache)) {
			t.Fatal("cache path not owned by app configuration")
		}
		if id == "glm53" {
			for _, value := range []string{
				"GLM53_CACHE_PATH=" + shellQuote(filepath.Join(filepath.Dir(cache), "glm53-fp4-local")),
				"VLLM_BIND='0.0.0.0'",
			} {
				if !strings.Contains(string(env), value) {
					t.Errorf("GLM recipe missing %s", value)
				}
			}
		}
		if component.ManagePath != "" {
			t.Fatal("legacy workspace path survived normalization")
		}
		info, _ := os.Stat(filepath.Join(dir, ".env"))
		if info.Mode().Perm() != 0600 {
			t.Fatal("environment is not private")
		}
	}
}

func TestGLMEmbeddedRecipeMatchesIndependentSources(t *testing.T) {
	data, err := assets.ReadFile("assets/recipes/glm53.tar.gz")
	if err != nil {
		t.Fatal(err)
	}
	gz, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if h.Typeflag != tar.TypeReg {
			continue
		}
		packed, err := io.ReadAll(tr)
		if err != nil {
			t.Fatal(err)
		}
		source, err := os.ReadFile(filepath.Join("recipe_sources", "glm53", h.Name))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(packed, source) {
			t.Errorf("repack GLM recipe: %s differs from independent source", h.Name)
		}
		standalone, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "compose_yaml", "zzz_not_use", "glm53f_sglang", h.Name))
		if err != nil || !bytes.Equal(source, standalone) {
			t.Errorf("GLM standalone/embedded source mismatch: %s: %v", h.Name, err)
		}

	}
}

func TestDS41TensorFoldBundle(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	b, _ := cat.Bundle("ds41")
	c, _ := cat.Component("ds41")
	if b.ContextTokens != 1048576 || !b.StartSupport || !b.WorkloadSwap || c.Model != "DeepSeek-V4.1-Flash-TF" || !c.isCluster() {
		t.Fatalf("wrong TensorFold profile: %+v", b)
	}
	for _, id := range []string{"nemotron-asr", "qwen3-tts"} {
		x, ok := cat.ResolveComponent("ds41", id)
		if !ok || x.Host != "local" || !x.StartAfterLLM || x.KeepResident || workloadGroup(id) == "" {
			t.Fatalf("speech must defer to the local memory-checked workload lease: %+v", x)
		}
	}
	controller := newController(cat)
	controller.ConfigurePaths(t.TempDir(), t.TempDir())
	dir, err := controller.materializeRecipe(context.Background(), c)
	if err != nil {
		t.Fatal(err)
	}
	for _, file := range []string{"profile.env", "pfdense-table.json", "launch.sh", "models.sh", "runtime.sh"} {
		got, err := os.ReadFile(filepath.Join(dir, file))
		if err != nil {
			t.Fatal(err)
		}
		want, _ := os.ReadFile(filepath.Join("recipe_sources", "ds41", file))
		if !bytes.Equal(got, want) {
			t.Fatal("embedded TensorFold source differs", file)
		}
	}
	profile, _ := os.ReadFile(filepath.Join(dir, "profile.env"))
	for _, setting := range []string{"TF_DSV41_CONTEXT=1048576", "TF_DSV41_POOL_TOKENS=2100000", "TF_DSV41_SLOTS=4", "TF_DSV41_FLOOR_HARD_GIB=4", "TF_DSV41_IMAGES=native", "TF_DSV41_LOOKUP=1", "TF_DSV41_ENGRAM_PREFETCH=1"} {
		if !strings.Contains(string(profile), setting) {
			t.Fatal("qualified engine setting lost", setting)
		}
	}
}

func TestDS41TensorFoldOptions(t *testing.T) {
	cat, _ := LoadCatalog()
	c, _ := cat.Component("ds41")
	if c.RuntimeOptions["MODEL_VARIANT"] != "abliterated" || modelPresentation(c).SelectedVariant != "abliterated" {
		t.Fatal("Abliterated must be the default; original must remain selectable")
	}
	for _, variant := range []string{"official", "abliterated"} {
		c.RuntimeOptions = map[string]string{"MODEL_VARIANT": variant, "MAX_MODEL_LEN": "1048576"}
		if err := validateRecipeOptions(c); err != nil {
			t.Fatal(err)
		}
		if modelPresentation(c).SelectedVariant != variant {
			t.Fatal("variant presentation mismatch")
		}
	}
	for _, value := range []string{"auto", "1", "2", "3", "4", "5"} {
		c.RuntimeOptions = map[string]string{"MTP_TOKENS": value}
		if err := validateRecipeOptions(c); err != nil {
			t.Fatal(err)
		}
	}
	for _, value := range []string{"0", "6", "01", "1.0", "static", "1\nBAD=1"} {
		c.RuntimeOptions = map[string]string{"MTP_TOKENS": value}
		if validateRecipeOptions(c) == nil {
			t.Fatal("invalid MTP depth accepted", value)
		}
	}
	for _, options := range []map[string]string{{"MAX_MODEL_LEN": "65536"}, {"DSV41_PRELOAD_COUNT": "224"}, {"MODEL_VARIANT": "unknown"}, {"MAX_NUM_SEQS": "8"}, {"MTP_TOKENS": "0"}} {
		c.RuntimeOptions = options
		if validateRecipeOptions(c) == nil {
			t.Fatal("unqualified TensorFold settings accepted", options)
		}
	}
}

func TestDS41MissingVariantUsesTheSameDefaultInPresentationAndRecipe(t *testing.T) {
	cat, _ := LoadCatalog()
	x, _ := cat.Component("ds41")
	x.RuntimeOptions = nil
	if modelPresentation(x).SelectedVariant != "abliterated" {
		t.Fatal("wrong presentation default")
	}
	c := newController(cat)
	c.ConfigurePaths(t.TempDir(), t.TempDir())
	dir, err := c.materializeRecipe(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	env, err := os.ReadFile(filepath.Join(dir, ".env"))
	if err != nil || !strings.Contains(string(env), "MODEL_VARIANT='abliterated'") {
		t.Fatal("materialized recipe disagrees with default weight selector", err)
	}
}

func TestGLMCheckpointPreflight(t *testing.T) {
	script, err := filepath.Abs(filepath.Join("recipe_sources", "glm53", "check_models.py"))
	if err != nil {
		t.Fatal(err)
	}
	code := `import importlib.util,json,struct,tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('check',__import__('sys').argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory() as folder:
 root=Path(folder);model=root/'model';draft=root/'draft';model.mkdir();draft.mkdir()
 (model/'config.json').write_text(json.dumps({'architectures':['Glm5NextForConditionalGeneration'],'quantization_config':{'quant_method':'modelopt','quant_algo':'NVFP4'}}))
 for name in ['tokenizer.json','tokenizer_config.json']:(model/name).write_text('{}')
 header=json.dumps({str(i):{'dtype':'U8','shape':[1],'data_offsets':[0,1]} for i in range(33)}).encode();blob=struct.pack('<Q',len(header))+header+b'x'
 index={'weight_map':{str(i):f'model-{i}.safetensors' for i in range(33)}}
 (model/'model.safetensors.index.json').write_text(json.dumps(index))
 for name in index['weight_map'].values():(model/name).write_bytes(blob)
 (draft/'model.safetensors').write_bytes(blob)
 draft_config={'architectures':['DFlash2DraftModel'],'num_target_layers':45,'vocab_size':154880,'dflash_config':{'target_layer_ids':[5,14,24,33,42]}}
 (draft/'config.json').write_text(json.dumps(draft_config))
 m.check(model,draft)
 (draft/'model.safetensors').write_bytes(blob[:-1])
 try:m.check(model,draft)
 except ValueError:pass
 else:raise AssertionError('truncated draft accepted')
 (draft/'model.safetensors').write_bytes(blob)
 bad=model/'model-0.safetensors';bad.write_bytes(blob[:-1])
 try:m.check(model,draft)
 except ValueError:pass
 else:raise AssertionError('truncated shard accepted')
 bad.unlink()
 try:m.check(model,draft)
 except FileNotFoundError:pass
 else:raise AssertionError('missing shard accepted')
`
	if out, err := exec.Command("python3", "-c", code, script).CombinedOutput(); err != nil {
		t.Fatalf("preflight validation: %v %s", err, out)
	}
	out, err := exec.Command("python3", script, filepath.Join(t.TempDir(), "missing"), t.TempDir(), "헤드").CombinedOutput()
	if err == nil || !strings.Contains(string(out), "모델만 준비") {
		t.Fatalf("missing actionable recovery guidance: %v %s", err, out)
	}
}

// runtime.sh's network/start actions require this helper in a freshly extracted
// package, not merely in the developer's standalone checkout.
func TestGLMNVFP4ModelSelectionIgnoresRetiredPaths(t *testing.T) {
	script, err := filepath.Abs(filepath.Join("recipe_sources", "glm53", "select_model.sh"))
	if err != nil {
		t.Fatal(err)
	}
	for variant, suffix := range map[string]string{
		"official":    "nvidia/GLM-5.3-Flash-NVFP4",
		"abliterated": "edp1096/Huihui-GLM-5.3-Flash-abliterated-NVFP4",
	} {
		cmd := exec.Command("bash", "-c", `source "$1"; printf '%s' "$MODEL_HOST_PATH"`, "bash", script)
		cmd.Env = append(os.Environ(), "HF_CACHE=/tmp/test-hf", "MODEL_VARIANT="+variant, "MODEL_HOST_PATH=/tmp/retired-exl3")
		out, err := cmd.CombinedOutput()
		if err != nil || string(out) != "/tmp/test-hf/"+suffix {
			t.Fatalf("%s: %v %s", variant, err, out)
		}
	}
}

func TestGLMRecipeRejectsUnqualifiedMTP(t *testing.T) {
	for _, value := range []string{"1", "3", "4"} {
		if validateRecipeOptions(Component{ID: "glm53", Controller: "glm53-cluster", RuntimeOptions: map[string]string{"MTP_TOKENS": value}}) == nil {
			t.Fatalf("accepted unqualified MTP configuration: %s", value)
		}
	}
	if err := validateRecipeOptions(Component{ID: "glm53", Controller: "glm53-cluster", RuntimeOptions: map[string]string{"MTP_TOKENS": "0"}}); err != nil {
		t.Fatal(err)
	}
}

func TestGLMSGLangOptions(t *testing.T) {
	for _, value := range []string{"0", "5"} {
		c := Component{ID: "glm53", Controller: "glm53-cluster", ProgressKind: "sglang", RuntimeOptions: map[string]string{"DFLASH_TOKENS": value}}
		if err := validateRecipeOptions(c); err != nil {
			t.Fatal(err)
		}
	}
	for key, value := range map[string]string{"DFLASH_TOKENS": "7", "MAX_NUM_SEQS": "2", "KV_CACHE_MEMORY": "12884901888", "GPU_MEMORY_UTILIZATION": "0.85"} {
		c := Component{ID: "glm53", Controller: "glm53-cluster", ProgressKind: "sglang", RuntimeOptions: map[string]string{key: value}}
		if validateRecipeOptions(c) == nil {
			t.Fatalf("accepted incompatible GLM option %s=%s", key, value)
		}
	}
}
