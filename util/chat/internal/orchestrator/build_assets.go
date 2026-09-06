package orchestrator

import (
	"context"
	"io/fs"
	"path/filepath"
	"strings"
)

func embeddedBuildAsset(compose string) string {
	switch compose {
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
		target := filepath.Join(directory, strings.TrimPrefix(strings.TrimPrefix(path, root), "/"))
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
