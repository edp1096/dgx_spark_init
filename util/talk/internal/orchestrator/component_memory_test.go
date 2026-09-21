package orchestrator

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestIndividualGPUStartChecksBeforeChangingContainer(t *testing.T) {
	for _, action := range []string{"start", "restart"} {
		t.Run(action, func(t *testing.T) {
			dir := t.TempDir()
			log := filepath.Join(dir, "calls")
			t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
			t.Setenv("MEM_TEST_LOG", log)
			err := os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$MEM_TEST_LOG\"\ncase \"$*\" in *inspect*) printf exited;; esac\n"), 0700)
			if err != nil {
				t.Fatal(err)
			}
			catalog, _ := LoadCatalog()
			for i := range catalog.Components {
				if catalog.Components[i].ID == "flux2" {
					catalog.Components[i].MemoryGiB = 1000000
				}
			}
			c, err := NewControllerWithCatalog(catalog)
			if err != nil {
				t.Fatal(err)
			}
			err = c.ComponentActionWithReserve("flux2", action, 4, "flash-next")
			if err == nil || !strings.Contains(err.Error(), "통합메모리 부족 예상") {
				t.Fatalf("unguarded action: %v", err)
			}
			if c.Operation().State != "failed" {
				t.Fatal("failure not reported")
			}
			data, _ := os.ReadFile(log)
			for _, mutation := range []string{"stop ", "rm ", "up -d"} {
				if strings.Contains(string(data), mutation) {
					t.Fatalf("mutated container: %s", data)
				}
			}
		})
	}
}
