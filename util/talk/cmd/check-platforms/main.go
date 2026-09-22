// check-platforms compiles all supported deployment targets without executing
// them or modifying installed binaries.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func main() {
	if err := check(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func check() error {
	dir, err := os.MkdirTemp("", "sparktalk-platforms-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	env := []string{}
	for _, item := range os.Environ() {
		if !strings.HasPrefix(item, "GOOS=") && !strings.HasPrefix(item, "GOARCH=") && !strings.HasPrefix(item, "CGO_ENABLED=") {
			env = append(env, item)
		}
	}
	for _, target := range []string{"linux/arm64", "linux/amd64", "windows/arm64", "windows/amd64"} {
		parts := strings.Split(target, "/")
		cmd := exec.Command("go", "build", "-o", filepath.Join(dir, "check"), "./cmd/chat")
		cmd.Env = append(append([]string{}, env...), "CGO_ENABLED=0", "GOOS="+parts[0], "GOARCH="+parts[1])
		if out, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("%s: %w\n%s", target, err, out)
		}
		fmt.Println(target, "OK")
	}
	return nil
}
