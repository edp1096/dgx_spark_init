//go:build linux && (arm64 || amd64)

package plugins

import (
	"os/exec"
	"syscall"
)

func configurePluginProcess(cmd *exec.Cmd) error {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pdeathsig: syscall.SIGKILL}
	return nil
}
