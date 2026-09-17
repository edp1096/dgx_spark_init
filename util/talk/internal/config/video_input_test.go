package config

import (
	"path/filepath"
	"testing"
)

func TestVideoInputIsIsolatedByDeploymentAndPersists(t *testing.T) {
	c, _, err := Load(filepath.Join(t.TempDir(), "initial.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	c.Runtime.Mode = "external"
	c.Model.Endpoint = "http://server-a:8000"
	c.Model.DefaultModel = "ornith"
	c.Model.ModelType = "qwen3.5"
	c.Model.VideoInputs = map[string]string{"http://server-a:8000\nornith": "frames"}
	for _, tc := range []struct{ endpoint, model, typ, want string }{
		{"http://server-a:8000/", "ornith", "qwen3.5", "frames"},
		{"http://server-a:8000", "other", "qwen3.5", "native"},
		{"http://server-b:8000", "ornith", "qwen3.5", "native"},
		{"http://server-b:8000", "deepseek", "deepseek-v4", "frames"},
		{"http://server-a:8000", "ornith", "qwen3.5", "frames"},
	} {
		c.Model.Endpoint, c.Model.DefaultModel, c.Model.ModelType = tc.endpoint, tc.model, tc.typ
		if got := c.Model.VideoInputMode(); got != tc.want {
			t.Fatalf("%+v: got %s", tc, got)
		}
	}
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := Save(path, c); err != nil {
		t.Fatal(err)
	}
	loaded, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := loaded.Model.VideoInputs["http://server-a:8000\nornith"]; got != "frames" {
		t.Fatalf("override lost: %q", got)
	}
}

func TestManagedModelSwitchPreservesVideoOverrides(t *testing.T) {
	c, _, err := Load(filepath.Join(t.TempDir(), "config.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	c.Model.VideoInputs = map[string]string{"http://server-a:8000\nornith": "frames"}
	c.ApplyManagedBundle("flash-next")
	c.ApplyManagedBundle("gemma26")
	if c.Model.VideoInputs["http://server-a:8000\nornith"] != "frames" {
		t.Fatal("model switch discarded video override")
	}
	if c.Model.VideoInputMode() != "native" {
		t.Fatal("unrelated model inherited video override")
	}
}
