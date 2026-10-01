package orchestrator

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type modelAsset struct {
	Repo     string            `json:"repo"`
	Revision string            `json:"revision,omitempty"`
	Path     string            `json:"path"`
	SHA256   map[string]string `json:"sha256,omitempty"`
	Files    []string          `json:"files,omitempty"`
	Pipeline bool              `json:"pipeline,omitempty"`
	HubCache bool              `json:"hub_cache,omitempty"`
}

func componentModelAssets(c Component) []modelAsset {
	switch c.ComposeAsset {
	case "compose.flash-next.yaml":
		repo, rev, err := QwenQADCheckpoint(c.qwenQADVariant())
		if err != nil {
			return nil
		}
		if c.qwenQADVariant() == "huihui_lil" {
			return []modelAsset{{Repo: repo, Revision: rev, Path: "/hf/" + repo}}
		}
		return []modelAsset{{Repo: repo, Revision: rev, Path: "/hf/hub/models--" + strings.ReplaceAll(repo, "/", "--") + "/snapshots/" + rev, HubCache: true}}
	case "compose.gemma31.yaml":
		return []modelAsset{{Repo: "lyf/Huihui-gemma-4-31B-it-abliterated-v2-NVFP4", Revision: "985633e8001d0bfd9b621738234394744b992907", Path: "/hf/hub/models--lyf--Huihui-gemma-4-31B-it-abliterated-v2-NVFP4/snapshots/985633e8001d0bfd9b621738234394744b992907", HubCache: true}, {Repo: "z-lab/gemma-4-31B-it-DFlash", Revision: "eabd648301ce28583cc14757912e5e0f84e152e1", Path: "/hf/hub/models--z-lab--gemma-4-31B-it-DFlash/snapshots/eabd648301ce28583cc14757912e5e0f84e152e1", HubCache: true}}
	case "compose.dreamlite.yaml":
		return []modelAsset{{Repo: "carlofkl/DreamLite-mobile", Revision: "6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4", Path: "/hf/hub/models--carlofkl--DreamLite-mobile/snapshots/6695c3f4be230f0493fa5dbf78be3bc4d3bb2ab4", HubCache: true, Pipeline: true}}
	case "compose.ornith35.yaml":
		return []modelAsset{{Repo: "edp1096/Huihui-Ornith-1.5-35B-A3B-NVFP4", Revision: "4ca22bf685b6afe269ebbceb3c297ddeade912ca", Path: "/hf/edp1096/Huihui-Ornith-1.5-35B-A3B-NVFP4"}}
	case "compose.gemma26.yaml":
		return []modelAsset{{Repo: "edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4", Revision: "4bce3d23429ef48f7aac4982a4714cc16ad1cc24", Path: "/hf/edp1096/Huihui-Gemma-4-26B-A4B-it-NVFP4"}, {Repo: "google/gemma-4-26B-A4B-it-assistant", Revision: "6e5aaaf4c42b98394530b8fda2e95cadd65c151c", Path: "/hf/google/gemma-4-26B-A4B-it-assistant"}}
	case "compose.nemotron-asr.yaml":
		return []modelAsset{{Repo: "nvidia/nemotron-3.5-asr-streaming-0.6b", Revision: "ea30d66debe3740a08b573244286791d423d6b3e", Path: "/tmp/models", Files: []string{"nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"}, SHA256: map[string]string{"nemotron-3.5-asr-streaming-0.6b.q8_0.gguf": "3fc991d3badad7277c11030a7519832cddaf2057aafed6d4b25147e953a070b1"}}, {Repo: "nvidia/Nemotron-3-Diarization", Revision: "f667ed73aee57d40cc39428eb768b4fd87a0a29e", Path: "/tmp/models", Files: []string{"Nemotron-3-Diarization.q8_0.gguf"}, SHA256: map[string]string{"Nemotron-3-Diarization.q8_0.gguf": "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"}}}
	case "compose.magpie-tts.yaml":
		return []modelAsset{{Repo: "nvidia/magpie_tts_multilingual_357m", Revision: "5023df68bd3f5b5ce6d666a50979bc501af145cc", Path: "/tmp/models", Files: []string{"magpie_tts_multilingual_357m.nemo"}, SHA256: map[string]string{"magpie_tts_multilingual_357m.nemo": "ec675fa8c02b9c1d5382c5c2b5a6acec6492c1e8344866c07cf3892185d18953"}}, {Repo: "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps", Revision: "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7", Path: "/tmp/models", Files: []string{"nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf"}, SHA256: map[string]string{"nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf": "cc86d36d821a27cdc1d4ef600a3e2b0dabe76e88fcc2a8652d9543134c07ef2d"}}}
	}
	return nil
}

// Preparation is shared by first start and explicit preparation. It never starts
// a GPU service. Existing complete weights are checked before any network access.
func (c *Controller) ensureModelAssets(ctx context.Context, component Component, token string) error {
	if component.ComposeAsset == "compose.flux2.yaml" {
		return c.prepareFluxWeights(ctx, component, token)
	}
	items := componentModelAssets(component)
	if len(items) == 0 {
		return nil
	}
	data, cache, err := c.runtimeHostPaths(component)
	if err != nil {
		return err
	}
	host := c.host(component.Host)
	modelDir := filepath.Join(filepath.Dir(cache), "nemo-speech")
	if component.ComposeAsset == "compose.magpie-tts.yaml" {
		modelDir = filepath.Join(modelDir, "magpie-v2607")
	}
	if _, err = executeHost(ctx, host, nil, "mkdir", "-p", cache, modelDir); err != nil {
		return err
	}
	build := filepath.Join(data, "runtime", "model-prepare")
	if err = materializeBuildAssets(ctx, host, "model-prepare", build); err != nil {
		return err
	}
	source, _ := assets.ReadFile("assets/model-prepare/prepare.py")
	dockerfile, _ := assets.ReadFile("assets/model-prepare/Dockerfile")
	digest := sha256.Sum256(append(source, dockerfile...))
	image := fmt.Sprintf("sparktalk-model-prepare:1-%x", digest[:8])
	if _, err = executeHost(ctx, host, nil, "docker", "image", "inspect", image); err != nil {
		if _, err = executeHost(ctx, host, nil, "docker", "build", "-t", image, build); err != nil {
			return err
		}
	}
	if token == "" {
		c.mu.RLock()
		localData := c.dataDir
		c.mu.RUnlock()
		b, e := os.ReadFile(filepath.Join(localData, "credentials", "huggingface.token"))
		if e != nil && !os.IsNotExist(e) {
			return e
		}
		token = strings.TrimSpace(string(b))
	}
	cacheMount := cache
	if component.ComposeAsset == "compose.dreamlite.yaml" {
		cacheMount = "media-hf-cache"
	}
	user, err := executionUser(ctx, host)
	if err != nil {
		return err
	}
	payload, _ := json.Marshal(map[string]any{"items": items, "token": token})
	// Stdin is the only credential channel. The image has no GPU access.
	out, err := executeHost(ctx, host, payload, "docker", "run", "--rm", "-i", "--user", user, "-e", "HOME=/tmp", "--memory", "2g", "--memory-swap", "2g", "-e", "HF_HUB_OFFLINE=0", "-e", "HF_HOME=/hf", "-e", "HF_HUB_DISABLE_PROGRESS_BARS=1", "-e", "HF_XET_HIGH_PERFORMANCE=0", "-e", "HF_XET_NUM_CONCURRENT_RANGE_GETS=1", "-v", cacheMount+":/hf", "-v", modelDir+":/tmp/models", image)
	detail := string(out)
	if token != "" {
		detail = strings.ReplaceAll(detail, token, "[redacted]")
	}
	if report := recipeReporter(ctx); report != nil {
		report(detail)
	}
	if err != nil {
		return fmt.Errorf("모델 준비 실패: %s", detail)
	}
	if component.ComposeAsset == "compose.magpie-tts.yaml" {
		return c.prepareMagpieWeights(ctx, host, data, modelDir)
	}
	return nil
}

func (c *Controller) prepareMagpieWeights(ctx context.Context, host Host, data, modelDir string) error {
	const model = "magpie-v2607-pr17-speaker-order-v2.f16.gguf"
	// Conversion uses the same pinned, speaker-order-patched source as runtime.
	if _, err := executeHost(ctx, host, nil, "sh", "-c", `test -s "$1/`+model+`" && test -s "$1/nano-codec.decoder.f16.gguf" && test -s "$1/extracted/model_config.yaml" && test -s "$1/extracted/.sparktalk-extracted"`, "sh", modelDir); err == nil {
		return nil
	}
	build := filepath.Join(data, "runtime", "magpie-model-converter")
	if err := materializeBuildAssets(ctx, host, "magpie-tts", build); err != nil {
		return err
	}
	const image = "sparktalk-magpie-converter:v2607-longform2"
	if _, err := executeHost(ctx, host, nil, "docker", "image", "inspect", image); err != nil {
		if _, err = executeHost(ctx, host, nil, "docker", "build", "--target", "converter", "-t", image, build); err != nil {
			return err
		}
	}
	script := `set -eu
cd /models
mkdir -p extracted
tar -xf magpie_tts_multilingual_357m.nemo -C extracted
printf ready > extracted/.sparktalk-extracted
if [ ! -s nano-codec.decoder.f16.gguf ]; then cp nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf nano-codec.decoder.f16.gguf; fi
if [ ! -s ` + model + ` ]; then python /src/convert_model.py /models/magpie_tts_multilingual_357m.nemo --outtype f16 --outfile /models/` + model + `.partial; mv ` + model + `.partial ` + model + `; fi`
	user, err := executionUser(ctx, host)
	if err != nil {
		return err
	}
	_, err = executeHost(ctx, host, nil, "docker", "run", "--rm", "--user", user, "-e", "HOME=/tmp", "--memory", "6g", "--memory-swap", "6g", "-v", modelDir+":/models", "--entrypoint", "sh", image, "-c", script)
	return err
}

func (c *Controller) prepareFluxWeights(ctx context.Context, component Component, token string) error {
	host := c.host(component.Host)
	const image = "sparktalk-flux2-paint:resident3"
	probe := `import os,sys
from pathlib import Path
sys.path.insert(0,'/opt/nvfp4-api')
from prepare_models import FILES,LOCAL_UNCENSORED_NVFP4
from prepare_lora import ADAPTERS,BFS_ADAPTER
from huggingface_hub import try_to_load_from_cache
if not LOCAL_UNCENSORED_NVFP4.is_file():
 print('conversion');sys.exit(2)
files=[try_to_load_from_cache(repo,name) for repo,name,_ in FILES]
files += [try_to_load_from_cache('fal/flux-2-klein-4B-'+kind+'-lora',name,revision=rev) for kind,rev,name in ADAPTERS]
files += [try_to_load_from_cache(BFS_ADAPTER[0],BFS_ADAPTER[2],revision=BFS_ADAPTER[1])]
if not all(isinstance(p,str) and Path(p).is_file() for p in files) or not Path('/root/.cache/huggingface/rembg/u2net.onnx').is_file():sys.exit(1)
print('ready')`
	common := []string{"docker", "run", "--rm", "-i", "--memory", "112g", "--memory-swap", "112g", "-v", "media-hf-cache:/root/.cache/huggingface", "--entrypoint", "python"}
	out, err := executeHost(ctx, host, nil, append(append([]string{}, common...), image, "-c", probe)...)
	if err == nil {
		return nil
	}
	convert := strings.Contains(string(out), "conversion")
	if convert {
		memory, e := c.hostMemory(ctx, component.Host)
		if e != nil {
			return e
		}
		if memory.AvailableGiB < 12 {
			return fmt.Errorf("FLUX 최초 텍스트 인코더 변환에는 12 GiB 여유가 필요합니다. 언어 모델 기동 전에 FLUX 전체 준비를 실행하세요")
		}
		common = append(common, "--gpus", "all")
	}
	if token == "" {
		c.mu.RLock()
		dir := c.dataDir
		c.mu.RUnlock()
		b, e := os.ReadFile(filepath.Join(dir, "credentials", "huggingface.token"))
		if e != nil && !os.IsNotExist(e) {
			return e
		}
		token = strings.TrimSpace(string(b))
	}
	script := `import os,sys,runpy,subprocess
from pathlib import Path
token=sys.stdin.readline().strip()
if token:os.environ['HF_TOKEN']=token
os.environ['HF_HUB_OFFLINE']='0'
sys.path[:0]=['/opt/ComfyUI','/opt/nvfp4-api']
from prepare_models import LOCAL_UNCENSORED_NVFP4
if not LOCAL_UNCENSORED_NVFP4.is_file():
 sys.argv=['quantize_uncensored_text_encoder.py']
 runpy.run_path('/opt/nvfp4-api/quantize_uncensored_text_encoder.py',run_name='__main__')
runpy.run_path('/opt/nvfp4-api/prepare_models.py',run_name='__main__')
runpy.run_path('/opt/nvfp4-api/prepare_lora.py',run_name='__main__')
subprocess.run(['/opt/rembg-venv/bin/python','/opt/nvfp4-api/rembg_worker.py','--prepare'],check=True)`
	out, err = executeHost(ctx, host, []byte(token+"\n"), append(common, image, "-u", "-c", script)...)
	detail := string(out)
	if token != "" {
		detail = strings.ReplaceAll(detail, token, "[redacted]")
	}
	if r := recipeReporter(ctx); r != nil {
		r(detail)
	}
	if err != nil {
		return fmt.Errorf("FLUX 최초 준비 실패: %s", detail)
	}
	return nil
}
