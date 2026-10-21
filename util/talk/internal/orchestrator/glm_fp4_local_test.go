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

func TestGLMFP4ProfileAndIndependentRecipe(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	x, _ := cat.Component("glm53")
	if x.Controller != "glm53-fp4-cluster" || !x.isCluster() || recipeID(x) != "glm53-fp4" {
		t.Fatal(x)
	}
	if err := validateRecipeOptions(x); err != nil {
		t.Fatal(err)
	}
	for key, value := range map[string]string{"MTP_TOKENS": "3", "MAX_MODEL_LEN": "8192", "MAX_NUM_SEQS": "2", "GPU_MEMORY_UTILIZATION": "0.95", "DFLASH_TOKENS": "5"} {
		invalid := x
		invalid.RuntimeOptions = map[string]string{key: value}
		if validateRecipeOptions(invalid) == nil {
			t.Fatal("unqualified override accepted", key, value)
		}
	}
	for _, v := range []string{"1", "5"} {
		x.RuntimeOptions = map[string]string{"MTP_TOKENS": v}
		if err := validateRecipeOptions(x); err != nil {
			t.Fatal(err)
		}
	}
	data, err := assets.ReadFile("assets/recipes/glm53-fp4.tar.gz")
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
		h, e := tr.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			t.Fatal(e)
		}
		if h.Typeflag != tar.TypeReg {
			continue
		}
		packed, e := io.ReadAll(tr)
		if e != nil {
			t.Fatal(e)
		}
		source := filepath.Join("recipe_sources", "glm53-fp4", h.Name)
		standalone := filepath.Join("../../../../compose_yaml/glm53f_fp4_local_vllm", h.Name)
		a, e := os.ReadFile(source)
		if e != nil {
			t.Fatal(e)
		}
		b, e := os.ReadFile(standalone)
		if e != nil {
			t.Fatal(e)
		}
		if !bytes.Equal(a, b) || !bytes.Equal(a, packed) {
			t.Fatal("recipe drift", h.Name)
		}
		ai, _ := os.Stat(source)
		bi, _ := os.Stat(standalone)
		if os.SameFile(ai, bi) {
			t.Fatal("recipes must have independent ownership", h.Name)
		}
	}
	c := newController(cat)
	c.ConfigurePaths(t.TempDir(), t.TempDir())
	x, _ = cat.Component("glm53")
	x.Container = "my-glm-head"
	x.WorkerContainer = "my-glm-worker"
	dir, err := c.materializeRecipe(context.Background(), x)
	if err != nil {
		t.Fatal(err)
	}
	env, err := os.ReadFile(filepath.Join(dir, ".env"))
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"GLM53_HEAD_CONTAINER='my-glm-head'", "GLM53_WORKER_CONTAINER='my-glm-worker'", "MTP_TOKENS='5'"} {
		if !strings.Contains(string(env), want) {
			t.Fatal("missing runtime deployment", want)
		}
	}
}
