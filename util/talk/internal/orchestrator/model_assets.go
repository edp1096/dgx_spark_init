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
	Repo      string            `json:"repo"`
	Revision  string            `json:"revision,omitempty"`
	Path      string            `json:"path"`
	SHA256    map[string]string `json:"sha256,omitempty"`
	Files     []string          `json:"files,omitempty"`
	Pipeline  bool              `json:"pipeline,omitempty"`
	HubCache  bool              `json:"hub_cache,omitempty"`
	LocalOnly bool              `json:"local_only,omitempty"`
}

func componentModelAssets(c Component) []modelAsset {
	switch c.ComposeAsset {
	case "compose.qwen38fn_exl3.yaml":
		return []modelAsset{{Repo: "alesha-pro/Huihui-Qwen3.8-Flash-Next-abliterated-exl3-3bit-hq_h6_ng6", Revision: "3b585c458f9fcf3322e76cff2c635c2cb81c5869", Path: "/hf/hub/models--alesha-pro--Huihui-Qwen3.8-Flash-Next-abliterated-exl3-3bit-hq_h6_ng6/snapshots/3b585c458f9fcf3322e76cff2c635c2cb81c5869", HubCache: true}}
	case "compose.qwim-mmh3.yaml":
		assets := componentModelAssets(Component{ComposeAsset: "compose.qwen-image21.yaml"})
		return append(assets,
			modelAsset{Repo: "lilcheaty/MiniMax-H3-NVFP4", Revision: "8c5abfed61e1b6a170240792b65253fba1a65b7b", Path: "/hf/hub/models--lilcheaty--MiniMax-H3-NVFP4/snapshots/8c5abfed61e1b6a170240792b65253fba1a65b7b", HubCache: true, Files: []string{"minimax_h3_fl2va_pruned_nvfp4.safetensors"}, SHA256: map[string]string{"minimax_h3_fl2va_pruned_nvfp4.safetensors": "72fa9269ce551fb63ff42a32d9b46d0c122e84b4b2c511e22fa698287b088f70"}},
			modelAsset{Repo: "Comfy-Org/Krea-2", Revision: "eb1eddd3983a54678545a9b2c178c5853b30f7be", Path: "/hf/hub/models--Comfy-Org--Krea-2/snapshots/eb1eddd3983a54678545a9b2c178c5853b30f7be", HubCache: true, Files: []string{"text_encoders/qwen3vl_4b_fp8_scaled.safetensors"}, SHA256: map[string]string{"text_encoders/qwen3vl_4b_fp8_scaled.safetensors": "54bd5144df0bbc25dd6ccadfcb826b521445a1b06ae5a42570bdd2974ca87094"}},
			modelAsset{Repo: "NicoLab28/ClipProj-MiniMax-H3", Revision: "2ebdbcdc27a29a9607efdb221a9afcb9a0cdd808", Path: "/hf/hub/models--NicoLab28--ClipProj-MiniMax-H3/snapshots/2ebdbcdc27a29a9607efdb221a9afcb9a0cdd808", HubCache: true, Files: []string{"mmh3-4b-ClipProj-v3.1.safetensors"}, SHA256: map[string]string{"mmh3-4b-ClipProj-v3.1.safetensors": "0184e5c8d666a131962506d21949c2d8a8c6f33445b7b5e347e9a7e0a5baa819"}},
			modelAsset{Repo: "Comfy-Org/MiniMax-H3", Revision: "e5eb578a89295337b8ff433a035929ce0279e0b6", Path: "/hf/hub/models--Comfy-Org--MiniMax-H3/snapshots/e5eb578a89295337b8ff433a035929ce0279e0b6", HubCache: true, Files: []string{"vae/minimax_h3_video_vae_int8_convrot.safetensors"}, SHA256: map[string]string{"vae/minimax_h3_video_vae_int8_convrot.safetensors": "52a2c8c73583c86e4f41cdcce3a6ad0ea562987bc0bf3d60a0cef5f5c8e60c0e"}},
			modelAsset{Repo: "Comfy-Org/MiniMax-H3", Revision: "e5eb578a89295337b8ff433a035929ce0279e0b6", Path: "/hf/hub/models--Comfy-Org--MiniMax-H3/snapshots/e5eb578a89295337b8ff433a035929ce0279e0b6", HubCache: true, Files: []string{"vae/minimax_h3_audio_vae_fp32.safetensors"}, SHA256: map[string]string{"vae/minimax_h3_audio_vae_fp32.safetensors": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48"}},
		)
	case "compose.qwen-image21.yaml":
		return []modelAsset{
			{Repo: "abenzerps/Qwen-Image-2.1-Uncensored-GGUF", Revision: "6b34e59458d3eb7ba6a6f86a116aed5253dc02c3", Path: "/hf/hub/models--abenzerps--Qwen-Image-2.1-Uncensored-GGUF/snapshots/6b34e59458d3eb7ba6a6f86a116aed5253dc02c3", HubCache: true, Files: []string{"qwen-image-2.1-UC-NVFP4.safetensors"}, SHA256: map[string]string{"qwen-image-2.1-UC-NVFP4.safetensors": "7cdf87660e84aecf740b5b0f65c848549ba8a74d88ac201976931ca4c780d121"}},
			{Repo: "Comfy-Org/Qwen-Image-2.1", Revision: "cb504a4090723e43f17ad01cec0359490e2de613", Path: "/hf/hub/models--Comfy-Org--Qwen-Image-2.1/snapshots/cb504a4090723e43f17ad01cec0359490e2de613", HubCache: true, Files: []string{"text_encoders/qwen3vl_8b_w4a8.safetensors", "vae/qwen_image_2.1_vae_bf16.safetensors"}, SHA256: map[string]string{"text_encoders/qwen3vl_8b_w4a8.safetensors": "7754425e55e7bea2bfde4dde59a4cc236cb44e5ee9c215ea66ef8d47012824eb", "vae/qwen_image_2.1_vae_bf16.safetensors": "bb21f7473051e1ac368515dd3f2e15cd44d7a11748ee8823e1ddca3e4876b7c9"}},
		}
	case "compose.flash-next.yaml":
		repo, rev, err := QwenQADCheckpoint(c.qwenQADVariant())
		if err != nil {
			return nil
		}
		if c.qwenQADVariant() == "huihui_lil" {
			release := qwenQADHuihuiRelease
			release.Path = "/hf/" + repo
			return []modelAsset{release}
		}
		if c.qwenQADVariant() == "radixark" {
			release := qwenRadixArkRelease
			release.Path = "/hf/" + repo
			return []modelAsset{release}
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
	case "compose.qwen3-tts.yaml":
		return []modelAsset{{Repo: "Serveurperso/Qwen3-TTS-GGUF", Revision: "b7ee2e8c7459c3bea99da23e3d178125a7d1713c", Path: "/tmp/models", Files: []string{"qwen-talker-0.6b-customvoice-Q8_0.gguf", "qwen-tokenizer-12hz-Q8_0.gguf"}, SHA256: map[string]string{"qwen-talker-0.6b-customvoice-Q8_0.gguf": "4eb38675c736ed6ac72012846ac8d6ef80e5af8bc05726870f0b3a6569588519", "qwen-tokenizer-12hz-Q8_0.gguf": "1883beeed99348fc35e23dd225e9082f93f6f8c109330a33d935baa8acdbfd94"}}}
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
	if component.ComposeAsset == "compose.qwen3-tts.yaml" {
		modelDir = filepath.Join(filepath.Dir(cache), "qwen3-tts")
	}
	if component.ComposeAsset == "compose.nemotron-asr.yaml" &&
		!hostModelMatches(ctx, host, filepath.Join(modelDir, nemotronQ5File), nemotronQ5SHA256) &&
		!hostModelMatches(ctx, host, filepath.Join(modelDir, nemotronF16File), nemotronF16SHA256) {
		items = append(items, nemotronSourceAsset())
	}
	if _, err = executeHost(ctx, host, nil, "mkdir", "-p", cache, modelDir); err != nil {
		return err
	}
	image, err := modelPreparationImage(ctx, host, data)
	if err != nil {
		return err
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
	args := []string{"docker", "run", "--rm", "-i", "--user", user, "-e", "HOME=/tmp", "--memory", "2g", "--memory-swap", "2g", "-e", "HF_HOME=/hf", "-e", "HF_HUB_DISABLE_PROGRESS_BARS=1", "-e", "HF_XET_HIGH_PERFORMANCE=0", "-e", "HF_XET_NUM_CONCURRENT_RANGE_GETS=1", "-v", cacheMount + ":/hf", "-v", modelDir + ":/tmp/models"}
	offline := len(items) > 0
	for _, item := range items {
		offline = offline && item.LocalOnly
	}
	if offline {
		args = append(args, "--network", "none", "-e", "HF_HUB_OFFLINE=1")
	} else {
		args = append(args, "-e", "HF_HUB_OFFLINE=0")
	}
	args = append(args, image)
	out, err := executeHost(ctx, host, payload, args...)
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
	if component.ComposeAsset == "compose.nemotron-asr.yaml" {
		return c.prepareNemotronQ5Weights(ctx, component, host, data, modelDir)
	}
	return nil
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

// Both asset preparation and closed-file cache advice use the small CPU image.
// Neither path depends on an image-generation service or allocates CUDA memory.
func modelPreparationImage(ctx context.Context, host Host, dataDir string) (string, error) {
	build := filepath.Join(dataDir, "runtime", "model-prepare")
	if err := materializeBuildAssets(ctx, host, "model-prepare", build); err != nil {
		return "", err
	}
	source, _ := assets.ReadFile("assets/model-prepare/prepare.py")
	dockerfile, _ := assets.ReadFile("assets/model-prepare/Dockerfile")
	digest := sha256.Sum256(append(source, dockerfile...))
	image := fmt.Sprintf("sparktalk-model-prepare:1-%x", digest[:8])
	if _, err := executeHost(ctx, host, nil, "docker", "image", "inspect", image); err != nil {
		if _, err := executeHost(ctx, host, nil, "docker", "build", "-t", image, build); err != nil {
			return "", err
		}
	}
	return image, nil
}
