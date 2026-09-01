package main

import (
	"path/filepath"
	"testing"
)

func TestResolveDefaultConfigPathUsesExecutableDirectory(t *testing.T) {
	root := t.TempDir()
	got := resolveDefaultConfigPath(filepath.Join(root, "dist", "media"), filepath.Join(root, "dist"), "")
	want := filepath.Join(root, "dist", "media.yaml")
	if got != want {
		t.Fatalf("config path = %q, want %q", got, want)
	}
}

func TestResolveDefaultConfigPathUsesExplicitOverride(t *testing.T) {
	root := t.TempDir()
	want := filepath.Join(root, "custom.yaml")
	got := resolveDefaultConfigPath(filepath.Join(root, "dist", "media"), root, want)
	if got != want {
		t.Fatalf("config path = %q, want %q", got, want)
	}
}

func TestResolveDefaultConfigPathDoesNotDependOnWorkingDirectory(t *testing.T) {
	root := t.TempDir()
	got := resolveDefaultConfigPath(filepath.Join(root, "bin", "media"), filepath.Join(root, "other"), "")
	want := filepath.Join(root, "bin", "media.yaml")
	if got != want {
		t.Fatalf("config path = %q, want %q", got, want)
	}
}

func TestResolveDefaultConfigPathFallsBackToWorkingDirectory(t *testing.T) {
	root := t.TempDir()
	got := resolveDefaultConfigPath("", root, "")
	want := filepath.Join(root, "media.yaml")
	if got != want {
		t.Fatalf("config path = %q, want %q", got, want)
	}
}
