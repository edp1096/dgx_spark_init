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
	for _, id := range []string{"glm53", "ds4fve", "ds41", "qwen38-tp2"} {
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
	for _, id := range []string{"glm53", "ds4fve", "ds41"} {
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
				"GLM53_CACHE_PATH=" + shellQuote(filepath.Join(filepath.Dir(cache), "glm53-nvfp4")),
				"VLLM_BIND='127.0.0.1'",
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

func TestDeepSeekRecoveryOptionsAreValidatedAndMaterialized(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	component, ok := cat.Component("ds4fve")
	if !ok {
		t.Fatal("missing DeepSeek component")
	}
	for _, key := range []string{"DSPARK_ENABLE_DSML_RECOVERY", "DSPARK_ENABLE_DSPARK_SWA_PREFIX"} {
		for _, value := range []string{"0", "1"} {
			item := component
			item.RuntimeOptions = map[string]string{key: value}
			if err := validateRecipeOptions(item); err != nil {
				t.Fatal(err)
			}
			controller := newController(cat)
			controller.ConfigurePaths(t.TempDir(), t.TempDir())
			dir, err := controller.materializeRecipe(context.Background(), item)
			if err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(filepath.Join(dir, ".env"))
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(data), key+"="+shellQuote(value)) {
				t.Fatalf("option did not reach recipe: %s", key)
			}
		}
		for _, value := range []string{"", "true", "2", "1\nBAD=1"} {
			item := component
			item.RuntimeOptions = map[string]string{key: value}
			if validateRecipeOptions(item) == nil {
				t.Fatalf("accepted invalid %s=%q", key, value)
			}
		}
		item := component
		item.Controller = "glm53-cluster"
		item.RuntimeOptions = map[string]string{key: "1"}
		if validateRecipeOptions(item) == nil {
			t.Fatal("accepted DeepSeek option for GLM")
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
		standalone, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "compose_yaml", "glm53f_sglang", h.Name))
		if err != nil || !bytes.Equal(source, standalone) {
			t.Errorf("GLM standalone/embedded source mismatch: %s: %v", h.Name, err)
		}

	}
}

func TestPackagedModelPatchesMatchStandalone(t *testing.T) {
	for id, folder := range map[string]string{"ds4fve": "ds4fve_vllm"} {
		data, err := assets.ReadFile("assets/recipes/" + id + ".tar.gz")
		if err != nil {
			t.Fatal(err)
		}
		gz, err := gzip.NewReader(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
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
			sourcePath := filepath.Join("../../../../compose_yaml", folder, h.Name)
			source, err := os.ReadFile(sourcePath)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(packed, source) {
				t.Errorf("repack %s: %s differs", id, h.Name)
			}
		}
		gz.Close()
	}
}

func TestDS41StreamingBundle(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	bundle, ok := cat.Bundle("ds41")
	if !ok {
		t.Fatal("missing DS41 bundle")
	}
	if bundle.ContextTokens != 65536 || !bundle.StartSupport {
		t.Fatal("wrong DS41 startup defaults")
	}
	if len(cat.StartBundleMembers(bundle).Components) != 7 {
		t.Fatal("ASR, TTS and all four Extra services must start with DS41")
	}
	if len(cat.ModelBundle(bundle).Components) != 3 {
		t.Fatal("stopping a set must preserve shared Extra services")
	}
	for _, id := range []string{"nemotron-asr", "extra-media", "extra-ssh", "extra-collector", "extra-documents"} {
		c, ok := cat.ResolveComponent("ds41", id)
		want := "local"
		if id == "nemotron-asr" {
			want = "worker"
		}
		if !ok || c.Host != want {
			t.Fatalf("wrong service binding: %s", id)
		}
	}
	c, _ := cat.Component("ds41")
	if !c.isCluster() || recipeID(c) != "ds41" {
		t.Fatal("DS41 must manage both ranks")
	}
	data, _ := assets.ReadFile("assets/recipes/ds41.tar.gz")
	gz, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	root := filepath.Join("..", "..", "..", "..", "compose_yaml", "ds41f_vllm")
	seenDenseProfile := false
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if h.Name == "dense-prefill-profile.json" {
			seenDenseProfile = true
		}
		if h.Name != "launch.sh" && h.Name != "expert-hot-profile.json" && h.Name != "dense-prefill-profile.json" && !strings.HasSuffix(h.Name, ".py") && !strings.HasPrefix(h.Name, "patches/") {
			continue
		}
		want, err := os.ReadFile(filepath.Join(root, h.Name))
		if err != nil {
			t.Fatal(err)
		}
		got, err := io.ReadAll(tr)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Fatalf("DS41 package differs from tested runtime: %s", h.Name)
		}
	}
	if !seenDenseProfile {
		t.Fatal("DS41 recipe must include the qualified dense prefill profile")
	}
}

func TestDS41PreloadOption(t *testing.T) {
	cat, _ := LoadCatalog()
	component, _ := cat.Component("ds41")
	for _, value := range []string{"0", "128", "224"} {
		component.RuntimeOptions = map[string]string{"DSV41_PRELOAD_COUNT": value}
		if err := validateRecipeOptions(component); err != nil {
			t.Fatal(err)
		}
		c := newController(cat)
		c.ConfigurePaths(t.TempDir(), t.TempDir())
		dir, err := c.materializeRecipe(context.Background(), component)
		if err != nil {
			t.Fatal(err)
		}
		env, _ := os.ReadFile(filepath.Join(dir, ".env"))
		if !strings.Contains(string(env), "DSV41_PRELOAD_COUNT="+shellQuote(value)) {
			t.Fatal("preload override lost")
		}
	}
	for _, value := range []string{"-1", "225", "1.5", "x", "01"} {
		component.RuntimeOptions = map[string]string{"DSV41_PRELOAD_COUNT": value}
		if validateRecipeOptions(component) == nil {
			t.Fatal("invalid preload count accepted", value)
		}
	}
	component.Controller = "dspark-cluster"
	component.RuntimeOptions = map[string]string{"DSV41_PRELOAD_COUNT": "128"}
	if validateRecipeOptions(component) == nil {
		t.Fatal("preload option accepted for another engine")
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
func TestDS4FVERailHelperIsPackaged(t *testing.T) {
	data, err := assets.ReadFile("assets/recipes/ds4fve.tar.gz")
	if err != nil {
		t.Fatal(err)
	}
	gz, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	found := false
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if h.Name == "ensure_rail.py" && h.Typeflag == tar.TypeReg {
			body, err := io.ReadAll(tr)
			if err != nil || len(body) == 0 {
				t.Fatal("empty rail helper", err)
			}
			found = true
		}
	}
	if !found {
		t.Fatal("DS4FVE start/network depends on missing ensure_rail.py")
	}
}

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
