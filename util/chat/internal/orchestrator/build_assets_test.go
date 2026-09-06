package orchestrator

import (
	"bytes"
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEmbeddedBuildContextsReproduceStandaloneSources(t *testing.T) {
	for name, folder := range map[string]string{"gemma31": "sglang_gemma4_31b", "flash-next": "sglang_qwen38_fn"} {
		t.Run(name, func(t *testing.T) {
			destination := t.TempDir()
			if err := materializeBuildAssets(context.Background(), Host{}, name, destination); err != nil {
				t.Fatal(err)
			}
			prefix := "assets/" + name + "/"
			err := fs.WalkDir(assets, "assets/"+name, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return err
				}
				if d.IsDir() {
					return nil
				}
				relative := strings.TrimPrefix(path, prefix)
				expected, err := os.ReadFile(filepath.Join("../../../../compose_yaml", folder, relative))
				if err != nil {
					return err
				}
				actual, err := os.ReadFile(filepath.Join(destination, relative))
				if err != nil {
					return err
				}
				if !bytes.Equal(expected, actual) {
					t.Errorf("embedded build differs: %s", path)
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
