package orchestrator

import (
	"context"
	"fmt"
	"path/filepath"
)

// The Q5 digest check warms file pages after workload admission. Beside the
// resident 1M NVFP4 engine and embedding worker, 6.3 GiB immediate free failed
// in cudaSetDevice; 7.3 GiB passed after advising closed ASR checkpoints.
func (c *Controller) prepareNemotronColdCUDA(ctx context.Context, x Component) error {
	if x.ComposeAsset != "compose.nemotron-asr.yaml" || x.Model != "nemotron-3.5-asr-streaming-0.6b" || !c.local(x) {
		return nil
	}
	core, ok := c.Catalog().Component("flash-next-radixark")
	if !ok || !c.componentRunning(ctx, core) {
		return nil
	}
	embed, ok := c.Catalog().ResolveComponent("flash-next-radixark", "extra-embedding")
	if !ok || !embed.KeepResident || !c.componentRunning(ctx, embed) {
		return nil
	}
	if c.componentRunning(ctx, x) {
		return fmt.Errorf("ASR 실행 중에는 가중치 파일 캐시를 회수하지 않습니다")
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
for name in ('nemotron-3.5-asr-streaming-0.6b.q5_k.gguf','nemotron-3.5-asr-streaming-0.6b.q8_0.gguf','Nemotron-3-Diarization.q8_0.gguf'):
 p=Path('/asr')/name
 if p.is_file():
  with p.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)
print('closed ASR checkpoint pages advised')`
	if _, err = executeHost(ctx, host, nil, "docker", "run", "--rm", "--network", "none", "--user", user, "--memory", "128m", "--memory-swap", "128m", "--entrypoint", "python", "-v", filepath.Join(filepath.Dir(cache), "nemo-speech")+":/asr:ro", image, "-c", script); err != nil {
		return err
	}
	memory := readSystemMemory()
	if c.memoryProbe != nil {
		memory = c.memoryProbe()
	}
	if memory.FreeGiB < 7 {
		if err := c.idleWorkloadAction(ctx, embed, "reclaim-cache"); err != nil {
			return err
		}
		memory = readSystemMemory()
		if c.memoryProbe != nil {
			memory = c.memoryProbe()
		}
	}
	if err := validateMemoryHeadroom(memory, memoryPlan{NeededGiB: x.startupMemoryGiB(), RequiresCUDAStart: true, MinimumCUDAFreeGiB: 7}, 1.5); err != nil {
		return fmt.Errorf("ASR 준비 완료 후 CUDA 기동 여유: %w", err)
	}
	return nil
}
