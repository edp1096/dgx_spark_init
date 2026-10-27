package orchestrator

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
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
	output, err := exec.Command(python, "-c", prefix+script+"\nprint(json.dumps(recorded))\n", "image").CombinedOutput()
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

func TestAuxiliaryCacheScopesRunningServicesProtectOnlyTheirFiles(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	defer c.Close()
	asr, _ := c.Catalog().Component("nemotron-asr")
	image, _ := c.Catalog().Component("qwim-mmh3")
	seedAuxiliary(t, dir, asr.Container, image.Container)
	// The joint image worker is not a member of this set. Its shared files must
	// still be excluded from the stopped-file sweep.
	b, _ := c.Catalog().Bundle("flash-next")
	scopes, images, err := c.auxiliaryCacheScopes(context.Background(), b)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(scopes, ",")
	if strings.Contains(got, "asr") || strings.Contains(got, "image") || !strings.Contains(got, "tts") {
		t.Fatalf("running ASR/image affected unrelated TTS scope: %v", scopes)
	}
	if len(images) != 1 || images[0].Container != image.Container {
		t.Fatalf("resident joint image not discovered: %v", images)
	}
	tts, _ := c.Catalog().Component("qwen3-tts")
	seedAuxiliary(t, dir, tts.Container)
	scopes, _, err = c.auxiliaryCacheScopes(context.Background(), b)
	if err != nil || strings.Contains(strings.Join(scopes, ","), "tts") {
		t.Fatalf("resident TTS cache included: %v %v", scopes, err)
	}
}

func TestClosedImageCacheExcludesOpenMappedAndAliasedFiles(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux file identity contract")
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("Python unavailable")
	}
	dir := t.TempDir()
	prefix := `import os,json,sys
from pathlib import Path
root=Path(sys.argv[1]);proc=root/'proc'/'42';(proc/'fd').mkdir(parents=True)
for name in ('closed','open','mapped','llm'):(root/name).write_bytes(b'fixture')
os.link(root/'open',root/'alias')
(proc/'fd'/'7').symlink_to(root/'open')
s=(root/'mapped').stat()
(proc/'maps').write_text('1000-2000 r--p 00000000 %x:%x %d /another/path\n' % (os.major(s.st_dev),os.minor(s.st_dev),s.st_ino))
sys.argv=[sys.argv[0],json.dumps([str(root/name) for name in ('closed','open','alias','mapped')])]
recorded=[]
os.posix_fadvise=lambda fd,*args: recorded.append(os.readlink('/proc/self/fd/'+str(fd)))
`
	script := strings.ReplaceAll(closedImageCacheAdvice, "Path('/proc')", "(root/'proc')")
	output, err := exec.Command(python, "-c", prefix+script+"\nassert recorded==[str(root/'closed')], recorded\n", dir).CombinedOutput()
	if err != nil {
		t.Fatalf("open/mmap/alias protection: %v %s", err, output)
	}
}

func TestResidentImageCacheQuiesceAndResume(t *testing.T) {
	for _, scenario := range []string{"idle", "busy", "queued", "unknown", "quiescing", "quiesce-race", "exec-error"} {
		t.Run(scenario, func(t *testing.T) {
			c, dir, _ := offlineAuxiliaryController(t)
			defer c.Close()
			x, _ := c.Catalog().Component("qwim-mmh3")
			script := "#!/bin/sh\nprintf 'exec\\n' >> \"$AUX_TEST_ROOT/cache-actions\"\n"
			if scenario == "exec-error" {
				script += "exit 1\n"
			}
			if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
				t.Fatal(err)
			}
			var actions []string
			c.client.Transport = auxiliaryTransport(func(r *http.Request) (*http.Response, error) {
				actions = append(actions, r.URL.Path)
				body, code := `{"status":"ok","busy":false,"queued":0}`, http.StatusOK
				if strings.HasSuffix(r.URL.Path, "/memory") {
					switch scenario {
					case "busy":
						body = `{"status":"ok","busy":true}`
					case "queued":
						body = `{"status":"ok","busy":false,"queued":1}`
					case "unknown":
						body = `{"status":"ok"}`
					case "quiescing":
						body = `{"status":"ok","busy":false,"quiescing":true}`
					}
				}
				if scenario == "quiesce-race" && strings.HasSuffix(r.URL.Path, "/quiesce") {
					code = http.StatusConflict
				}
				return &http.Response{StatusCode: code, Body: io.NopCloser(strings.NewReader(body))}, nil
			})
			err := c.reclaimResidentImageFileCache(context.Background(), x)
			if (err != nil) != (scenario == "exec-error") {
				t.Fatalf("error: %v", err)
			}
			actionLog, _ := os.ReadFile(filepath.Join(dir, "cache-actions"))
			executes := scenario == "idle" || scenario == "exec-error"
			if (len(actionLog) > 0) != executes {
				t.Fatalf("cache action during %s: %s", scenario, actionLog)
			}
			if executes && strings.Join(actions, ",") != "/v1/runtime/memory,/v1/runtime/quiesce,/v1/runtime/resume" {
				t.Fatalf("admission not restored: %v", actions)
			}
		})
	}
}

func TestFishCacheScopeIsIndependentOfResidentQwenTTS(t *testing.T) {
	c, dir, _ := offlineAuxiliaryController(t)
	defer c.Close()
	tts, _ := c.Catalog().Component("qwen3-tts")
	fish, _ := c.Catalog().Component("extra-fish")
	seedAuxiliary(t, dir, tts.Container)
	b, _ := c.Catalog().Bundle("qwen38fn_exl3")
	scopes, _, err := c.auxiliaryCacheScopes(context.Background(), b)
	if err != nil {
		t.Fatal(err)
	}
	got := strings.Join(scopes, ",")
	if strings.Contains(got, "tts") || !strings.Contains(got, "fish") {
		t.Fatal(scopes)
	}
	seedAuxiliary(t, dir, fish.Container)
	scopes, _, err = c.auxiliaryCacheScopes(context.Background(), b)
	if err != nil || strings.Contains(strings.Join(scopes, ","), "fish") {
		t.Fatal("running Fish weights must be protected", scopes, err)
	}
}

func TestStoppedNativeH3AndFishCacheAdviceExcludesLanguageModel(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("Python unavailable")
	}
	root := t.TempDir()
	files := []string{
		"hub/models--lilcheaty--MiniMax-H3-NVFP4/snapshots/8c5abfed61e1b6a170240792b65253fba1a65b7b/native.safetensors",
		"hub/models--watchpen--drama-3-int4-koen160/snapshots/fd0147b1bb988ea26718ee75e95de8b07a07c156/model-00001.safetensors",
		"hub/models--watchpen--drama-3-int4-koen160/snapshots/fd0147b1bb988ea26718ee75e95de8b07a07c156/codec.pth",
		"hub/models--watchpen--drama-3-int4-koen160/snapshots/unrelated/model.safetensors",
		"hub/models--Qwen--Qwen3.8/snapshots/resident/model.safetensors",
	}
	for _, name := range files {
		p := filepath.Join(root, name)
		os.MkdirAll(filepath.Dir(p), 0700)
		os.WriteFile(p, []byte("fixture"), 0600)
	}
	script := strings.ReplaceAll(auxiliaryCacheAdvice, "/qwen-image-cache", root)
	prefix := "import os,json\nrecorded=[]\nos.posix_fadvise=lambda fd,*args:recorded.append(os.readlink('/proc/self/fd/'+str(fd)))\n"
	out, err := exec.Command(python, "-c", prefix+script+"\nprint(json.dumps(recorded))", "fish,image").CombinedOutput()
	if err != nil {
		t.Fatal(err, string(out))
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	var paths []string
	if err = json.Unmarshal([]byte(lines[len(lines)-1]), &paths); err != nil {
		t.Fatal(err)
	}
	if len(paths) != 3 {
		t.Fatal(paths)
	}
	for _, p := range paths {
		if strings.Contains(p, "unrelated") || strings.Contains(p, "Qwen3.8") {
			t.Fatal("protected weights advised", p)
		}
	}
	if managedColdCUDAFreeGiB(Component{ComposeAsset: "compose.extra-fish.yaml"}) < 16 {
		t.Fatal("insufficient multi-process bootstrap reserve")
	}
}
