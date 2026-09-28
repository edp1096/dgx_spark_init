package orchestrator

import (
	"context"
	"fmt"
	"gopkg.in/yaml.v3"
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
		if id != "flux2" && id != "nemotron-asr" {
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
	c.mu.RUnlock()
	if !filepath.IsAbs(cache) {
		return fmt.Errorf("absolute model cache path required")
	}
	args := []string{"run", "--rm", "--network", "none", "--entrypoint", "python", "-v", "media-hf-cache:/cache:ro"}
	nemo := filepath.Join(filepath.Dir(cache), "nemo-speech")
	if st, err := os.Stat(nemo); err == nil && st.IsDir() {
		args = append(args, "-v", nemo+":/asr:ro")
	}
	data, err := composeAsset("compose.flux2.yaml")
	if err != nil {
		return err
	}
	var spec struct {
		Services map[string]struct {
			Image string `yaml:"image"`
		} `yaml:"services"`
	}
	if err := yaml.Unmarshal(data, &spec); err != nil {
		return err
	}
	image := spec.Services["runtime"].Image
	if image == "" {
		return fmt.Errorf("missing auxiliary image")
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
paths=[Path('/asr/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf'),Path('/asr/Nemotron-3-Diarization.q8_0.gguf')]
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
