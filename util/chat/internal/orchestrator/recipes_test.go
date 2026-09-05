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
	for _, id := range []string{"glm53", "ds4fve", "qwen27-exl3", "flash-next-exl3"} {
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
	for _, id := range []string{"glm53", "ds4fve", "qwen27-exl3", "flash-next-exl3"} {
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
