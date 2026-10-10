package orchestrator

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

// MemAvailable includes clean model pages. CUDA context creation on GB10 also
// needs immediate free pages. Advise only closed auxiliary weight files away;
// never use global drop_caches or touch the resident Qwen checkpoint/KV.
func (c *Controller) reclaimAuxiliaryFileCache(ctx context.Context, b Bundle) error {
	for _, id := range b.Components {
		if id != "flux2" && id != "nemotron-asr" && id != "qwen3-tts" {
			continue
		}
		x, _ := c.Catalog().ResolveComponent(b.ID, id)
		state, err := c.inspectComponent(ctx, x)
		if err != nil && !isMissingContainer(err) {
			return err
		}
		if state == "running" {
			return fmt.Errorf("%s 실행 중에는 가중치 파일 캐시를 회수하지 않습니다", x.Name)
		}
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
	args = append(args, image, "-c", auxiliaryCacheAdvice)
	commandCtx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(commandCtx, "docker", args...).CombinedOutput(); err != nil {
		return fmt.Errorf("부가 모델 파일 캐시 회수: %w: %s", err, out)
	}
	return nil
}

const auxiliaryCacheAdvice = `import os
from pathlib import Path
roots=[Path('/cache/local/flux2-klein-4b-uncensored-nvfp4')]
for name in ('models--black-forest-labs--FLUX.2-klein-4b-nvfp4','models--Comfy-Org--flux2-klein-4B','models--fal--flux-2-klein-4B-outpaint-lora','models--fal--flux-2-klein-4B-object-remove-lora','models--fal--flux-2-klein-4B-background-remove-lora'):
 roots.append(Path('/cache/hub')/name/'snapshots')
for repo, revision in (('models--abenzerps--Qwen-Image-2.1-Uncensored-GGUF','6b34e59458d3eb7ba6a6f86a116aed5253dc02c3'),('models--Comfy-Org--Qwen-Image-2.1','cb504a4090723e43f17ad01cec0359490e2de613')):
 roots.append(Path('/qwen-image-cache/hub')/repo/'snapshots'/revision)
paths=[Path('/asr/nemotron-3.5-asr-streaming-0.6b.q5_k.gguf'),Path('/asr/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf'),Path('/asr/Nemotron-3-Diarization.q8_0.gguf')]
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
print('closed auxiliary weight files advised:',len(seen))
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
