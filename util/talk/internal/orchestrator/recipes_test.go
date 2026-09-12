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

func TestEmbeddedRecipesContainNoPrivateEnvironment(t *testing.T) {
	for _, id := range []string{"glm53", "ds4fve", "ds41"} {
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
				"ABLIT_HOST_PATH=" + shellQuote(filepath.Join(cache, "glm53-lovesenko-oproj")),
				"ABLIT_LAYERS='0-44'", "ABLIT_INCLUDE_MTP='0'",
				"ABLIT_DONOR='lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated'",
				"ABLIT_DONOR_REVISION='c8f58e6aa9117c73607d692978b22f091d80450c'",
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
	if len(cat.StartBundleMembers(bundle).Components) != 6 {
		t.Fatal("ASR and all four Extra services must start with DS41")
	}
	if len(cat.ModelBundle(bundle).Components) != 2 {
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
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if h.Name != "launch.sh" && h.Name != "expert-hot-profile.json" && !strings.HasSuffix(h.Name, ".py") && !strings.HasPrefix(h.Name, "patches/") {
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
