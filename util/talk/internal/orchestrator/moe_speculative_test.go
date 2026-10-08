package orchestrator

import (
	"context"
	"gopkg.in/yaml.v3"
	"os"
	"path/filepath"
	"testing"
)

func TestMoESpeculativeDeploymentOptions(t *testing.T) {
	for _, id := range []string{"ornith35", "gemma26"} {
		for _, tokens := range []string{"0", "1", "3", "invalid"} {
			t.Run(id+"/"+tokens, func(t *testing.T) {
				dir := t.TempDir()
				if err := os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\ncase \"$*\" in *config) cat;; esac\n"), 0700); err != nil {
					t.Fatal(err)
				}
				t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
				c, _ := NewController()
				c.ConfigurePaths(dir, filepath.Join(dir, "models"))
				component, _ := c.Catalog().Component(id)
				component.RuntimeOptions = map[string]string{"MTP_TOKENS": tokens, "DRAFT_VOCAB": "off"}
				err := c.startComponent(context.Background(), component)
				if tokens == "invalid" {
					if err == nil {
						t.Fatal("invalid draft count accepted")
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				data, err := os.ReadFile(filepath.Join(dir, "runtime", id, "compose.yaml"))
				if err != nil {
					t.Fatal(err)
				}
				var document struct {
					Services map[string]struct{ Environment map[string]string }
				}
				if err := yaml.Unmarshal(data, &document); err != nil {
					t.Fatal(err)
				}
				env := document.Services["runtime"].Environment
				if env["MTP_TOKENS"] != tokens || env["DRAFT_VOCAB"] != "off" {
					t.Fatalf("options did not reach compose: %v", env)
				}
			})
		}
	}
}
