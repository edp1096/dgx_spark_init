// package-support copies canonical support-service sources and renders app Compose assets.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"gopkg.in/yaml.v3"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

type Spec struct {
	Key          string `json:"key"`
	ID           string `json:"id"`
	Name         string `json:"name"`
	Description  string `json:"description"`
	Port         int    `json:"port"`
	InternalPort int    `json:"internal_port"`
	Image        string `json:"image"`
	Version      string `json:"version"`
	BuildAsset   string `json:"build_asset"`
	Dockerfile   string `json:"dockerfile"`
}

func main() {
	check := flag.Bool("check", false, "verify generated assets without writing")
	flag.Parse()
	if err := run(*check); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func run(check bool) error {
	cwd, _ := os.Getwd()
	source := filepath.Clean(filepath.Join(cwd, "../../compose_yaml/sparktalk_extra"))
	dest := filepath.Join(cwd, "internal/orchestrator/assets")
	manifest, err := os.ReadFile(filepath.Join(source, "services.json"))
	if err != nil {
		return err
	}
	var specs []Spec
	if err = json.Unmarshal(manifest, &specs); err != nil {
		return err
	}
	expected := map[string][]byte{"support-services.json": manifest}
	copyFile := func(from, to string) error {
		b, e := os.ReadFile(filepath.Join(source, from))
		if e == nil {
			expected[assetPath(to)] = b
		}
		return e
	}
	copyTree := func(from, to string) error {
		return filepath.WalkDir(filepath.Join(source, from), func(p string, d fs.DirEntry, e error) error {
			if e != nil {
				return e
			}
			if d.IsDir() {
				if d.Name() == "node_modules" || d.Name() == "target" || d.Name() == "__pycache__" {
					return filepath.SkipDir
				}
				return nil
			}
			rel, _ := filepath.Rel(filepath.Join(source, from), p)
			b, e := os.ReadFile(p)
			if e == nil {
				expected[assetPath(filepath.Join(to, rel))] = b
			}
			return e
		})
	}
	for _, name := range []string{"go.mod", "go.sum", "Dockerfile.media", "Dockerfile.collector", "Dockerfile.ssh", "THIRD_PARTY_NOTICES.md"} {
		if err = copyFile(name, filepath.Join("support-services", name)); err != nil {
			return err
		}
	}
	if err = copyTree("cmd", "support-services/cmd"); err != nil {
		return err
	}
	for _, name := range []string{"Dockerfile", "package.json", "package-lock.json", "THIRD_PARTY_NOTICES.md"} {
		if err = copyFile("docms/"+name, "extra-documents/"+name); err != nil {
			return err
		}
	}
	matches, _ := filepath.Glob(filepath.Join(source, "docms/*.mjs"))
	for _, p := range matches {
		n := filepath.Base(p)
		if err = copyFile("docms/"+n, "extra-documents/"+n); err != nil {
			return err
		}
	}
	for _, name := range []string{"calc", "hwp-engine"} {
		if err = copyTree("docms/"+name, "extra-documents/"+name); err != nil {
			return err
		}
	}
	raw, err := os.ReadFile(filepath.Join(source, "compose.yaml"))
	if err != nil {
		return err
	}
	var compose map[string]any
	if err = yaml.Unmarshal(raw, &compose); err != nil {
		return err
	}
	services := compose["services"].(map[string]any)
	for _, s := range specs {
		service, ok := services[s.Key].(map[string]any)
		if !ok {
			return fmt.Errorf("missing canonical service %s", s.Key)
		}
		if s.Key == "documents" {
			raw, _ := yaml.Marshal(service)
			var compatibility map[string]any
			if err = yaml.Unmarshal(raw, &compatibility); err != nil {
				return err
			}
			compatibility["build"] = "."
			compatibility["ports"] = []string{"${BIND_ADDRESS:-127.0.0.1}:${PORT:-8696}:8696"}
			b, e := yaml.Marshal(map[string]any{"name": "extra_documents", "services": map[string]any{"documents": compatibility}})
			if e != nil {
				return e
			}
			path := filepath.Join(source, "docms/compose.yaml")
			old, _ := os.ReadFile(path)
			if !bytes.Equal(old, b) {
				if check {
					return fmt.Errorf("regenerate docms compatibility compose")
				}
				if e = os.WriteFile(path, b, 0644); e != nil {
					return e
				}
			}
		}
		service["image"] = s.Image
		service["build"] = map[string]any{"context": "${SPARKTALK_BUILD_DIR:?Embedded build assets required}", "dockerfile": s.Dockerfile}
		service["pull_policy"] = "never"
		service["ports"] = []string{fmt.Sprintf("${SPARKTALK_BIND_ADDR:-127.0.0.1}:${SPARKTALK_PORT:-%d}:%d", s.Port, s.InternalPort)}
		if s.Key == "ssh" {
			service["volumes"] = []string{"${SPARKTALK_DATA_DIR:-${HOME}/.local/share/sparktalk}/extra/ssh/keys:/run/sparktalk-extra/keys", "${SPARKTALK_DATA_DIR:-${HOME}/.local/share/sparktalk}/extra/ssh/state:/var/lib/sparktalk-extra"}
		}
		out := map[string]any{"services": map[string]any{"runtime": service}}
		if s.Key == "media" {
			out["volumes"] = compose["volumes"]
		}
		b, e := yaml.Marshal(out)
		if e != nil {
			return e
		}
		expected["compose."+s.ID+".yaml"] = b
	}
	for p, b := range expected {
		target := filepath.Join(dest, p)
		old, _ := os.ReadFile(target)
		if bytes.Equal(old, b) {
			continue
		}
		if check {
			return fmt.Errorf("regenerate support asset: %s", p)
		}
		if err = os.MkdirAll(filepath.Dir(target), 0755); err != nil {
			return err
		}
		if err = os.WriteFile(target, b, 0644); err != nil {
			return err
		}
	}
	for _, owned := range []string{"support-services", "extra-documents"} {
		err = filepath.WalkDir(filepath.Join(dest, owned), func(p string, d fs.DirEntry, e error) error {
			if e != nil {
				return e
			}
			if d.IsDir() {
				return nil
			}
			rel, _ := filepath.Rel(dest, p)
			if _, ok := expected[rel]; !ok && !strings.HasPrefix(d.Name(), ".") {
				if check {
					return fmt.Errorf("stale support asset: %s", rel)
				}
				return os.Remove(p)
			}
			return nil
		})
		if err != nil {
			return err
		}
	}
	fmt.Printf("Support assets verified: %d files\n", len(expected))
	return nil
}

func assetPath(p string) string {
	if strings.HasSuffix(p, ".go") || filepath.Base(p) == "go.mod" || filepath.Base(p) == "go.sum" {
		return p + ".asset"
	}
	return p
}
