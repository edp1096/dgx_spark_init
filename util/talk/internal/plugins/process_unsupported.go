//go:build !linux || (!arm64 && !amd64)

package plugins

import (
	"fmt"
	"os/exec"
)

func configurePluginProcess(*exec.Cmd) error {
	return fmt.Errorf("external plugin sandbox requires Linux arm64 or amd64; built-in extensions remain available")
}
