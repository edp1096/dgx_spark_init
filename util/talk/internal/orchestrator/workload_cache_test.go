package orchestrator

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestCacheAdviceScopesQwenImageToPinnedAuxiliaryRevisions(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux page cache contract")
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("Python unavailable")
	}
	dir := t.TempDir()
	qwenImage := "qwen-image-cache/hub/models--abenzerps--Qwen-Image-2.1-Uncensored-GGUF/snapshots/6b34e59458d3eb7ba6a6f86a116aed5253dc02c3/native.safetensors"
	encoder := "qwen-image-cache/hub/models--Comfy-Org--Qwen-Image-2.1/snapshots/cb504a4090723e43f17ad01cec0359490e2de613/text_encoders/native.safetensors"
	for _, name := range []string{qwenImage, encoder, "qwen-image-cache/hub/models--abenzerps--Qwen-Image-2.1-Uncensored-GGUF/snapshots/custom-revision/custom.safetensors", "qwen-image-cache/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4/snapshots/resident/llm.safetensors"} {
		path := filepath.Join(dir, name)
		if err = os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(path, []byte("fixture"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	script := auxiliaryCacheAdvice
	for _, root := range []string{"/qwen-image-cache", "/cache", "/asr"} {
		script = strings.ReplaceAll(script, root, filepath.Join(dir, strings.TrimPrefix(root, "/")))
	}
	prefix := `import os,json
recorded=[]
os.posix_fadvise=lambda fd,*args: recorded.append(os.readlink('/proc/self/fd/'+str(fd)))
`
	output, err := exec.Command(python, "-c", prefix+script+"\nprint(json.dumps(recorded))\n").CombinedOutput()
	if err != nil {
		t.Fatalf("cache advice: %v %s", err, output)
	}
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	var paths []string
	if err = json.Unmarshal([]byte(lines[len(lines)-1]), &paths); err != nil {
		t.Fatal(err)
	}
	if len(paths) != 2 {
		t.Fatalf("advised unrelated revision or LLM: %v", paths)
	}
	for _, path := range paths {
		if path != filepath.Join(dir, qwenImage) && path != filepath.Join(dir, encoder) {
			t.Fatal("unexpected cache target", path)
		}
	}
}

func TestQwenImageClosedFileCacheAdviceLive(t *testing.T) {
	if os.Getenv("SPARKTALK_QWEN_IMAGE_CACHE_LIVE") != "1" {
		t.Skip("explicit closed-cache integration")
	}
	c, err := NewController()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.ConfigurePaths(os.Getenv("SPARKTALK_QWEN_IMAGE_CACHE_DATA"), os.Getenv("SPARKTALK_QWEN_IMAGE_CACHE_ROOT"))
	b, _ := c.Catalog().Bundle("flash-next")
	before := readSystemMemory()
	if err = c.reclaimAuxiliaryFileCache(context.Background(), b); err != nil {
		t.Fatal(err)
	}
	after := readSystemMemory()
	t.Logf("CPU-only closed-cache advice passed; immediate free %.2f -> %.2f GiB", before.FreeGiB, after.FreeGiB)
}
