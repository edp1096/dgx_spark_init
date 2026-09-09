package orchestrator

import (
	"context"
	"io/fs"
	"path/filepath"
	"strings"
)

func embeddedBuildAsset(compose string) string {
	for _, spec := range SupportSpecs() {
		if compose == "compose."+spec.ID+".yaml" {
			return spec.BuildAsset
		}
	}
	switch compose {
	case "compose.flux2.yaml":
		return "flux2-paint"
	case "compose.dreamlite.yaml":
		return "dreamlite"
	case "compose.extra-documents.yaml":
		return "extra-documents"
	case "compose.magpie-tts.yaml":
		return "magpie-tts"
	case "compose.gemma31.yaml":
		return "gemma31"
	case "compose.flash-next.yaml":
		return "flash-next"
	}
	return ""
}

// Build inputs travel with the executable, including nested patches. The target
// host needs Docker and upstream network access, never the workspace checkout.
func materializeBuildAssets(ctx context.Context, host Host, name, directory string) error {
	root := "assets/" + name
	return fs.WalkDir(assets, root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		target := filepath.Join(directory, strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(path, root), "/"), ".asset"))
		if entry.IsDir() {
			_, err := executeHost(ctx, host, nil, "mkdir", "-p", target)
			return err
		}
		content, err := assets.ReadFile(path)
		if err != nil {
			return err
		}
		_, err = executeHost(ctx, host, content, "sh", "-c", `umask 077; cat > "$1"`, "sh", target)
		return err
	})
}
