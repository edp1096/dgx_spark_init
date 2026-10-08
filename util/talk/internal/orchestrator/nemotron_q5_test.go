package orchestrator

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Lifecycle fixtures have no real weights. Model preparation is independent
// of their fake Docker state transitions, so explicitly model a verified cache.
func fakeQualifiedNemotronModel(t *testing.T, dir string) {
	t.Helper()
	script := "#!/bin/sh\ncase \"${1##*/}\" in\n " + nemotronQ5File + ") printf '%s  %s\\n' " + nemotronQ5SHA256 + " \"$1\";;\n *) exit 1;;\nesac\n"
	if err := os.WriteFile(filepath.Join(dir, "sha256sum"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
}

func TestModelDigestRejectsMissingAndCorruptFiles(t *testing.T) {
	path := filepath.Join(t.TempDir(), "checkpoint.gguf")
	data := []byte("GGUF qualified test checkpoint")
	digest := fmt.Sprintf("%x", sha256.Sum256(data))
	if hostModelMatches(context.Background(), Host{}, path, digest) {
		t.Fatal("missing model accepted")
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if !hostModelMatches(context.Background(), Host{}, path, digest) {
		t.Fatal("complete model rejected")
	}
	if err := os.WriteFile(path, append(data, 'x'), 0600); err != nil {
		t.Fatal(err)
	}
	if hostModelMatches(context.Background(), Host{}, path, digest) {
		t.Fatal("corrupt model accepted")
	}
}

func TestNemotronPreparationFailureCannotStartService(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	if err := os.WriteFile(filepath.Join(dir, "sha256sum"), []byte("#!/bin/sh\nexit 1\n"), 0700); err != nil {
		t.Fatal(err)
	}
	x, _ := c.Catalog().Component("nemotron-asr")
	if err := c.prepareOrStartComponent(context.Background(), x, false); err == nil {
		t.Fatal("unverified Q5 model started")
	}
	if _, err := os.Stat(filepath.Join(dir, x.Container)); !os.IsNotExist(err) {
		t.Fatal("container started after failed model verification")
	}
}

func TestNemotronEmbeddedRecipeUsesQualifiedQ5Model(t *testing.T) {
	data, err := composeAsset("compose.nemotron-asr.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "/models/"+nemotronQ5File) || !strings.Contains(string(data), "sparktalk-nemotron-asr:0.1.0") {
		t.Fatal("runtime does not select the qualified Q5 release")
	}
	helper, err := assets.ReadFile("assets/nemotron-asr/scripts/prepare_q5.py")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(helper), nemotronQ5SHA256) || !strings.Contains(string(helper), nemotronF16SHA256) {
		t.Fatal("embedded converter and runtime release hashes differ")
	}
	source := nemotronSourceAsset()
	if source.Revision == "" || len(source.SHA256) != 1 || !strings.HasSuffix(source.Files[0], ".nemo") {
		t.Fatal("first-install source is not pinned")
	}
}
