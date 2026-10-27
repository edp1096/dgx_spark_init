package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestMusicBundlePreparationIncludesNestedCheckpoints(t *testing.T) {
	for _, compose := range []string{"compose.extra-music-ace.yaml"} {
		items := componentModelAssets(Component{ComposeAsset: compose})
		if len(items) < 2 {
			t.Fatal("music bundle lost dependent models", compose)
		}
		for _, item := range items {
			if len(item.Files) == 0 || len(item.Revision) != 40 {
				t.Fatal("bundled model must not use single-checkpoint root detection", item.Repo)
			}
		}
	}
}

func TestMusicLeaseRestoresImageAndPreservesLLM(t *testing.T) {
	for _, suffix := range []string{"ace"} {
		t.Run(suffix, func(t *testing.T) {
			c, dir := residentAuxiliaryController(t)
			container := "sparktalk-music-" + suffix
			seedAuxiliary(t, dir, container)
			c.memoryProbe = func() SystemMemory {
				available := 20.0
				if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen-image21")); os.IsNotExist(err) {
					available = 90
				}
				return SystemMemory{AvailableGiB: available, FreeGiB: available}
			}
			release, err := c.AcquireWorkload(context.Background(), "qwen38fn_exl3", "extra-music-"+suffix, 1.5)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(filepath.Join(dir, "sparktalk-qwen-image21")); !os.IsNotExist(err) {
				t.Fatal("image was not displaced", err)
			}
			if err := release(); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(filepath.Join(dir, container)); !os.IsNotExist(err) {
				t.Fatal("music process retained during image restoration", err)
			}
			for _, name := range []string{"sparktalk-qwen-image21", "sparktalk-qwen38fn_exl3"} {
				if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
					t.Fatal("required resident model missing", name, err)
				}
			}
		})
	}
}

func TestMusicCatalogAndAssetsExcludeRetiredBackend(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := catalog.Component("extra-music-heart"); ok {
		t.Fatal("retired music still selectable")
	}
	if _, err := assets.ReadFile("assets/compose.extra-music-heart.yaml"); err == nil {
		t.Fatal("retired compose still packaged")
	}
	items := componentModelAssets(Component{ComposeAsset: "compose.extra-music-ace.yaml"})
	if len(items) != 2 {
		t.Fatal("expected ACE base bundle and XL", len(items))
	}
	for _, item := range items {
		if !strings.HasPrefix(item.Repo, "ACE-Step/") {
			t.Fatal("unrelated model download", item.Repo)
		}
	}
}
