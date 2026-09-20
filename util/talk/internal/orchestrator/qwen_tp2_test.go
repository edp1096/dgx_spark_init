package orchestrator

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestQwenTP2BundleAndEmbeddedRuntime(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	model := "edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4"
	for _, id := range []string{"flash-next-tp2"} {
		component, _ := cat.Component(id)
		bundle, _ := cat.Bundle(id)
		if component.Model != model || bundle.ModelID != model {
			t.Fatalf("stale %s model identity", id)
		}
	}
	tp1, err := assets.ReadFile("assets/compose.flash-next.yaml")
	if err != nil || !strings.Contains(string(tp1), "/hf/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd") {
		t.Fatalf("stale TP1 model path: %v", err)
	}
	b, ok := cat.Bundle("flash-next-tp2")
	if !ok || b.ContextTokens != 1048576 {
		t.Fatal("1M bundle missing")
	}
	c, _ := cat.Component("flash-next-tp2")
	if !c.isCluster() || recipeID(c) != "qwen38-tp2" {
		t.Fatal("cluster controller missing")
	}
	for _, id := range []string{"flux2", "nemotron-asr", "magpie-tts"} {
		item, ok := cat.ResolveComponent(b.ID, id)
		if !ok || !item.StartAfterLLM {
			t.Fatalf("%s must start after LLM", id)
		}
	}
	data, err := assets.ReadFile("assets/recipes/qwen38-tp2.tar.gz")
	if err != nil {
		t.Fatal(err)
	}
	gz, err := gzip.NewReader(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	files := map[string][]byte{}
	for {
		h, e := tr.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			t.Fatal(e)
		}
		files[h.Name], err = io.ReadAll(tr)
		if err != nil {
			t.Fatal(err)
		}
	}
	source := filepath.Join("..", "..", "..", "..", "compose_yaml", "qwen38_fn_sglang")
	for _, name := range []string{"ensure_rail.py", "manage_tp2.py", "compose.tp2.yaml", "tp2/entrypoint.py", "tp2/watchdog.py"} {
		want, e := os.ReadFile(filepath.Join(source, name))
		if e != nil {
			t.Fatal(e)
		}
		if !bytes.Equal(files[name], want) {
			t.Fatalf("stale embedded %s", name)
		}
	}
	for _, name := range []string{"manage.sh", "runtime.sh", "models.sh", "env.sample"} {
		want, err := os.ReadFile(filepath.Join("recipe_sources", "qwen38-tp2", name))
		if err != nil || !bytes.Equal(files[name], want) {
			t.Fatalf("missing/stale wrapper %s: %v", name, err)
		}
	}
	dir := t.TempDir()
	ctrl := newController(cat)
	ctrl.ConfigurePaths(dir, filepath.Join(dir, "hf"))
	c.AutoAddress = false
	recipe, e := ctrl.materializeRecipe(context.Background(), c)
	if e != nil {
		t.Fatal(e)
	}
	env, e := os.ReadFile(filepath.Join(recipe, ".env"))
	if e != nil {
		t.Fatal(e)
	}
	for _, want := range []string{"MAX_MODEL_LEN='1048576'", "API_PORT='8012'", "HEAD_CONTAINER='sglang-qwen38-fn-tp2-0'", "WORKER_CONTAINER='sglang-qwen38-fn-tp2-1'"} {
		if !strings.Contains(string(env), want) {
			t.Fatalf("missing %s", want)
		}
	}
	c.RuntimeOptions = map[string]string{"MAX_MODEL_LEN": "2097152"}
	if validateRecipeOptions(c) == nil {
		t.Fatal("unsupported context accepted")
	}
}

func TestQwenTP2ReportsMTPProgress(t *testing.T) {
	c := Component{ID: "flash-next-tp2", Controller: "qwen38-cluster", ProgressKind: "sglang"}
	p := inferProgress(c, "Load weight end. elapsed=400.0 s\nLoad weight begin.\nSGLANG_WEIGHT_PROGRESS current=40 total=206 elapsed_seconds=8.1 eta_seconds=33.6\n")
	if p.Phase != "MTP 가중치 적재" {
		t.Fatalf("wrong draft label: %#v", p)
	}
}
