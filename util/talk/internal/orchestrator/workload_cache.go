package orchestrator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// MemAvailable includes clean model pages. CUDA context creation on GB10 also
// needs immediate free pages. Advise only closed auxiliary weight files away;
// never use global drop_caches or touch the resident Qwen checkpoint/KV.
func (c *Controller) reclaimAuxiliaryFileCache(ctx context.Context, b Bundle) error {
	// Running ASR/TTS protect their own files; they do not veto another
	// service's cold start. Image variants share checkpoint inodes, so inspect
	// catalog deployments as well as the requested bundle before advising them.
	scopes, images, err := c.auxiliaryCacheScopes(ctx, b)
	if err != nil {
		return err
	}
	if len(images) == 1 {
		if err := c.reclaimResidentImageFileCache(ctx, images[0]); err != nil {
			return err
		}
	}
	if len(scopes) == 0 {
		return nil
	}
	c.mu.RLock()
	cache := c.modelCache
	dataDir := c.dataDir
	c.mu.RUnlock()
	if !filepath.IsAbs(cache) {
		return fmt.Errorf("absolute model cache path required")
	}
	args := []string{"run", "--rm", "--network", "none", "--entrypoint", "python", "-v", "media-hf-cache:/cache:ro", "-v", cache + ":/qwen-image-cache:ro"}
	nemo := filepath.Join(filepath.Dir(cache), "nemo-speech")
	if st, err := os.Stat(nemo); err == nil && st.IsDir() {
		args = append(args, "-v", nemo+":/asr:ro")
	}
	tts := filepath.Join(filepath.Dir(cache), "qwen3-tts")
	if st, err := os.Stat(tts); err == nil && st.IsDir() {
		args = append(args, "-v", tts+":/tts:ro")
	}
	image, err := modelPreparationImage(ctx, Host{}, dataDir)
	if err != nil {
		return err
	}
	args = append(args, image, "-c", auxiliaryCacheAdvice, strings.Join(scopes, ","))
	commandCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(commandCtx, "docker", args...).CombinedOutput(); err != nil {
		return fmt.Errorf("부가 모델 파일 캐시 회수: %w: %s", err, out)
	}
	return nil
}

// A non-stopped or externally managed holder excludes the whole shared scope.
// An inspect failure must not turn into permission to advise its files.
func (c *Controller) auxiliaryCacheScopes(ctx context.Context, b Bundle) ([]string, []Component, error) {
	cat := c.Catalog()
	candidates := append(cat.Deployments(b.ID), cat.Components...)
	blocked := map[string]bool{}
	seen := map[string]bool{}
	var images []Component
	for _, x := range candidates {
		if !c.local(x) {
			continue
		}
		scope := ""
		switch {
		case isQwenImage21(x):
			scope = "image"
		case x.ComposeAsset == "compose.flux2.yaml":
			scope = "flux"
		case x.ComposeAsset == "compose.nemotron-asr.yaml":
			scope = "asr"
		case isManagedTTS(x):
			scope = "tts"
		default:
			continue
		}
		key := x.DeploymentKey() + ":" + scope
		if seen[key] {
			continue
		}
		seen[key] = true
		if x.Controller != "compose" {
			blocked[scope] = true
			if scope == "image" {
				images = append(images, Component{})
			}
			continue
		}
		state, err := c.inspectComponent(ctx, x)
		if err != nil && !isMissingContainer(err) {
			return nil, nil, err
		}
		if isMissingContainer(err) || state == "exited" || state == "created" || state == "dead" {
			continue
		}
		blocked[scope] = true
		if scope == "image" {
			// A single idle native owner can prove its files closed in its own PID
			// namespace. Multiple/unknown holders are conservatively left alone.
			if state != "running" {
				images = append(images, Component{})
			} else {
				images = append(images, x)
			}
		}
	}
	var scopes []string
	for _, scope := range []string{"flux", "image", "asr", "tts"} {
		if !blocked[scope] {
			scopes = append(scopes, scope)
		}
	}
	return scopes, images, nil
}

func (c *Controller) reclaimResidentImageFileCache(ctx context.Context, x Component) (result error) {
	if !isQwenImage21(x) {
		return nil
	}
	state, err := c.workloadIdleState(ctx, x)
	// Busy, already quiescing, or unobservable services retain their cache. The
	// caller still reclaims unrelated stopped services and rechecks real memory.
	if err != nil || *state.Busy || state.Queued > 0 || state.Quiescing {
		return nil
	}
	if err := c.idleWorkloadAction(ctx, x, "quiesce"); err != nil {
		return nil
	}
	defer func() {
		resumeCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		result = errors.Join(result, c.idleWorkloadAction(resumeCtx, x, "resume"))
	}()
	var paths []string
	for _, asset := range componentModelAssets(x) {
		for _, file := range asset.Files {
			paths = append(paths, strings.TrimRight(asset.Path, "/")+"/"+file)
		}
	}
	raw, err := json.Marshal(paths)
	if err != nil {
		return err
	}
	commandCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	_, err = executeHost(commandCtx, c.host(x.Host), nil, "docker", "exec", x.Container, "python3", "-c", closedImageCacheAdvice, string(raw))
	return err
}

// This runs inside the quiesced image container. Inode checks also protect
// mmap-backed storage and hardlinks whose names differ from the manifest.
const closedImageCacheAdvice = `import os,json,sys
from pathlib import Path
protected=set()
for proc in Path('/proc').glob('[0-9]*'):
 try:
  for line in (proc/'maps').read_text().splitlines():
   fields=line.split(None,5)
   major,minor=(int(n,16) for n in fields[3].split(':'))
   protected.add((os.makedev(major,minor),int(fields[4])))
  for fd in (proc/'fd').iterdir():
   try:
    st=fd.stat();protected.add((st.st_dev,st.st_ino))
   except FileNotFoundError:pass
 except FileNotFoundError:pass
seen=set()
for name in json.loads(sys.argv[1]):
 p=Path(name)
 if not p.is_file():continue
 with p.open('rb') as f:
  st=os.fstat(f.fileno());key=(st.st_dev,st.st_ino)
  if key in protected or key in seen:continue
  seen.add(key);os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
print('closed image weight files advised:',len(seen))
`

const auxiliaryCacheAdvice = `import os,sys
from pathlib import Path
scopes=set(sys.argv[1].split(','))
roots=[]
if 'flux' in scopes:
 roots.append(Path('/cache/local/flux2-klein-4b-uncensored-nvfp4'))
 for name in ('models--black-forest-labs--FLUX.2-klein-4b-nvfp4','models--Comfy-Org--flux2-klein-4B','models--fal--flux-2-klein-4B-outpaint-lora','models--fal--flux-2-klein-4B-object-remove-lora','models--fal--flux-2-klein-4B-background-remove-lora'):
  roots.append(Path('/cache/hub')/name/'snapshots')
if 'image' in scopes:
 for repo, revision in (('models--abenzerps--Qwen-Image-2.1-Uncensored-GGUF','6b34e59458d3eb7ba6a6f86a116aed5253dc02c3'),('models--Comfy-Org--Qwen-Image-2.1','cb504a4090723e43f17ad01cec0359490e2de613')):
  roots.append(Path('/qwen-image-cache/hub')/repo/'snapshots'/revision)
paths=[]
if 'asr' in scopes:
 paths.extend([Path('/asr/nemotron-3.5-asr-streaming-0.6b.q5_k.gguf'),Path('/asr/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf'),Path('/asr/Nemotron-3-Diarization.q8_0.gguf')])
if 'tts' in scopes:
 paths.extend([Path('/tts/qwen-talker-0.6b-customvoice-Q8_0.gguf'),Path('/tts/qwen-tokenizer-12hz-Q8_0.gguf')])
for root in roots:
 if root.is_dir():paths.extend(root.rglob('*.safetensors'))
seen=set()
for p in paths:
 if not p.is_file():continue
 with p.open('rb') as f:
  st=os.fstat(f.fileno());key=(st.st_dev,st.st_ino)
  if key in seen:continue
  seen.add(key);os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
print('stopped auxiliary weight files advised:',len(seen))
`

// Embedding admission must not evict or inspect resident ASR/TTS weights. Only
// this worker's closed checkpoint can be advised before its own cold start.
func (c *Controller) reclaimEmbeddingFileCache(ctx context.Context, x Component) error {
	if x.ComposeAsset != "compose.extra-embedding.yaml" {
		return fmt.Errorf("not an embedding worker")
	}
	if c.componentRunning(ctx, x) {
		if err := c.idleWorkloadAction(ctx, x, "quiesce"); err != nil {
			return err
		}
		defer c.idleWorkloadAction(ctx, x, "resume")
		state, err := c.workloadIdleState(ctx, x)
		if err != nil {
			return err
		}
		if state.Ready == nil || *state.Ready || *state.Busy {
			return fmt.Errorf("embedding checkpoint is not proven closed")
		}
	}
	data, cache, err := c.runtimeHostPaths(x)
	if err != nil {
		return err
	}
	host := c.host(x.Host)
	image, err := modelPreparationImage(ctx, host, data)
	if err != nil {
		return err
	}
	user, err := executionUser(ctx, host)
	if err != nil {
		return err
	}
	script := `import os
from pathlib import Path
p=Path('/embedding/model.safetensors')
if p.is_file():
 with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
print('closed embedding checkpoint advised')`
	_, err = executeHost(ctx, host, nil, "docker", "run", "--rm", "--network", "none", "--user", user, "--memory", "128m", "--memory-swap", "128m", "--entrypoint", "python", "-v", filepath.Join(cache, "google", "embeddinggemma-2")+":/embedding:ro", image, "-c", script)
	return err
}
