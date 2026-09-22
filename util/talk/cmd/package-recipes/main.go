// package-recipes generates embedded recipes from an explicit source manifest.
// Runtime files and credentials not listed in the manifest are never included.
package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

type recipe struct {
	ID     string   `json:"id"`
	Source string   `json:"source"`
	Files  []string `json:"files"`
}

func main() {
	check := flag.Bool("check", false, "verify recipes without writing")
	flag.Parse()
	if err := run(*check); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
func validMember(name string) bool {
	if name == "" || path.IsAbs(name) || path.Clean(name) != name || strings.Contains(name, "\\") || strings.HasPrefix(name, "../") {
		return false
	}
	for _, part := range strings.Split(name, "/") {
		if part == ".." || strings.HasPrefix(part, ".") {
			return false
		}
	}
	return true
}
func render(root string, r recipe) ([]byte, error) {
	if !validMember(r.ID) || strings.Contains(r.ID, "/") || len(r.Files) == 0 {
		return nil, fmt.Errorf("invalid recipe ID or empty file list")
	}
	names := append([]string(nil), r.Files...)
	sort.Strings(names)
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for i, name := range names {
		if !validMember(name) || (i > 0 && name == names[i-1]) {
			return nil, fmt.Errorf("invalid or duplicate recipe member: %q", name)
		}
		source := filepath.Join(root, r.Source, filepath.FromSlash(name))
		// Do not follow symlinks inside the source tree, including parent directories.
		relative := ""
		for _, part := range strings.Split(name, "/") {
			relative = filepath.Join(relative, part)
			info, err := os.Lstat(filepath.Join(root, r.Source, relative))
			if err != nil {
				return nil, err
			}
			if info.Mode()&os.ModeSymlink != 0 {
				return nil, fmt.Errorf("symlink in recipe source: %s", name)
			}
		}
		info, err := os.Stat(source)
		if err != nil {
			return nil, err
		}
		if !info.Mode().IsRegular() {
			return nil, fmt.Errorf("not a regular file: %s", name)
		}
		data, err := os.ReadFile(source)
		if err != nil {
			return nil, err
		}
		mode := int64(0644)
		if info.Mode().Perm()&0111 != 0 {
			mode = 0755
		}
		h := &tar.Header{Name: name, Mode: mode, Size: int64(len(data)), Typeflag: tar.TypeReg, ModTime: time.Unix(0, 0), Format: tar.FormatPAX}
		if err = tw.WriteHeader(h); err != nil {
			return nil, err
		}
		if _, err = tw.Write(data); err != nil {
			return nil, err
		}
	}
	if err := tw.Close(); err != nil {
		return nil, err
	}
	if err := gz.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
func run(check bool) error {
	root, err := os.Getwd()
	if err != nil {
		return err
	}
	raw, err := os.ReadFile(filepath.Join(root, "internal/orchestrator/recipe_sources/packages.json"))
	if err != nil {
		return err
	}
	var specs []recipe
	if err = json.Unmarshal(raw, &specs); err != nil {
		return err
	}
	seen := map[string]bool{}
	var failures []error
	for _, r := range specs {
		if seen[r.ID] {
			return fmt.Errorf("duplicate recipe: %s", r.ID)
		}
		seen[r.ID] = true
		data, err := render(root, r)
		if err != nil {
			return fmt.Errorf("%s: %w", r.ID, err)
		}
		dest := filepath.Join(root, "internal/orchestrator/assets/recipes", r.ID+".tar.gz")
		old, err := os.ReadFile(dest)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
		if !bytes.Equal(old, data) {
			if check {
				failures = append(failures, fmt.Errorf("regenerate recipe: %s", r.ID))
				continue
			}
			if err = os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
				return err
			}
			f, err := os.CreateTemp(filepath.Dir(dest), ".recipe-")
			if err != nil {
				return err
			}
			tmp := f.Name()
			_, err = f.Write(data)
			if err == nil {
				err = f.Sync()
			}
			closeErr := f.Close()
			if err == nil {
				err = closeErr
			}
			if err == nil {
				err = os.Chmod(tmp, 0644)
			}
			if err == nil {
				err = os.Rename(tmp, dest)
			}
			_ = os.Remove(tmp)
			if err != nil {
				return err
			}
		}
		fmt.Printf("Recipe verified: %s (%d files)\n", r.ID, len(r.Files))
	}
	return errors.Join(failures...)
}
