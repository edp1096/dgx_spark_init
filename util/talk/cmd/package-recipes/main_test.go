package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestRecipeIsDeterministicAndOnlyIncludesDeclaredFiles(t *testing.T) {
	root := t.TempDir()
	os.WriteFile(filepath.Join(root, "runtime.sh"), []byte("#!/bin/sh\n"), 0755)
	os.WriteFile(filepath.Join(root, "ensure_rail.py"), []byte("# helper\n"), 0644)
	os.WriteFile(filepath.Join(root, ".env"), []byte("secret"), 0600)
	r := recipe{ID: "test", Source: ".", Files: []string{"runtime.sh", "ensure_rail.py"}}
	a, err := render(root, r)
	if err != nil {
		t.Fatal(err)
	}
	b, err := render(root, r)
	if err != nil || !bytes.Equal(a, b) {
		t.Fatal("non-deterministic recipe", err)
	}
	gz, err := gzip.NewReader(bytes.NewReader(a))
	if err != nil {
		t.Fatal(err)
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	names := []string{}
	for {
		h, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		names = append(names, h.Name)
		if h.Name == "runtime.sh" && h.Mode != 0755 {
			t.Fatal("lost executable bit")
		}
	}
	if len(names) != 2 || names[0] != "ensure_rail.py" || names[1] != "runtime.sh" {
		t.Fatal(names)
	}
	os.WriteFile(filepath.Join(root, "ensure_rail.py"), []byte("# changed\n"), 0644)
	b, err = render(root, r)
	if err != nil || bytes.Equal(a, b) {
		t.Fatal("source change not reflected")
	}
	os.Remove(filepath.Join(root, "ensure_rail.py"))
	if _, err = render(root, r); err == nil {
		t.Fatal("missing dependency accepted")
	}
}
func TestRecipeRejectsUnsafeNames(t *testing.T) {
	for _, name := range []string{"../escape", "/absolute", ".env", "a/.env", "a/../b", "a\\b"} {
		if validMember(name) {
			t.Fatal(name)
		}
	}
}
