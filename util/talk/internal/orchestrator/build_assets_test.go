package orchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestIndependentModelBuildContextsAreOwnedByTalk(t *testing.T) {
	manifest, err := os.ReadFile("runtime_sources.json")
	if err != nil {
		t.Fatal(err)
	}
	var sources []struct{ Source, Target string }
	if err := json.Unmarshal(manifest, &sources); err != nil {
		t.Fatal(err)
	}
	for _, item := range sources {
		if item.Target == "compose.qwen-fp4-local.yaml" || item.Target == "compose.qwen38fn_exl3.yaml" || strings.HasPrefix(item.Target, "qwen-fp4-local/") || strings.HasPrefix(item.Target, "qwen38fn_exl3/") || strings.HasPrefix(item.Target, "gemma31/") || item.Target == "compose.gemma31.yaml" {
			t.Fatalf("Talk model files must not be overwritten by standalone packaging: %s", item.Target)
		}
	}
	for _, name := range []string{"qwen-fp4-local", "qwen38fn_exl3"} {
		t.Run(name, func(t *testing.T) {
			destination := t.TempDir()
			if err := materializeBuildAssets(context.Background(), Host{}, name, destination); err != nil {
				t.Fatal(err)
			}
			root := "assets/" + name
			if err := fs.WalkDir(assets, root, func(path string, entry fs.DirEntry, err error) error {
				if err != nil || entry.IsDir() {
					return err
				}
				relative := strings.TrimSuffix(strings.TrimPrefix(path, root+"/"), ".asset")
				info, err := os.Lstat(filepath.Join(destination, relative))
				if err != nil {
					return err
				}
				if !info.Mode().IsRegular() {
					t.Fatalf("materialized input must be an independent file: %s", relative)
				}
				expected, err := assets.ReadFile(path)
				if err != nil {
					return err
				}
				actual, err := os.ReadFile(filepath.Join(destination, relative))
				if err == nil && !bytes.Equal(expected, actual) {
					t.Errorf("materialized input differs from Talk's embedded file: %s", relative)
				}
				return err
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestEmbeddedBuildContextsReproduceStandaloneSources(t *testing.T) {
	for name, folder := range map[string]string{"support-services": "sparktalk_extra", "extra-documents": "sparktalk_extra/docms", "nemotron-asr": "nemotron_asr", "qwen3-tts": "qwen3_tts_gguf", "flash-next": "qwen38_fn_sglang", "qwen-image21": "qwen-image21"} {
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
				relative := strings.TrimSuffix(strings.TrimPrefix(path, prefix), ".asset")
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
