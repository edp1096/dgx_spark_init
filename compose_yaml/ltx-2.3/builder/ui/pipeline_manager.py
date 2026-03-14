"""Pipeline lifecycle management and model configuration."""

import dataclasses
import gc
import logging
import os
import time
import weakref
from pathlib import Path

import torch
from tqdm import tqdm
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.quantization import QuantizationPolicy
from ltx_pipelines.utils.constants import LTX_2_3_PARAMS

FP8_QUANTIZATION = QuantizationPolicy.fp8_cast()

logger = logging.getLogger("ltx2-ui")

# ---------------------------------------------------------------------------
# q8_kernels FP8 matmul — native FP8 GEMM (replaces BF16 upcast approach)
# ---------------------------------------------------------------------------
_Q8_AVAILABLE = False
_fp8_mm = None
try:
    from q8_kernels.modules.linear import FP8Linear as _Q8FP8Linear
    from q8_kernels.functional.linear import fp8_mm as _fp8_mm
    _Q8_AVAILABLE = True
except ImportError:
    pass


def _fp8_forward_no_hadamard(self, x, x_scales=None, out_dtype=None):
    """FP8 forward WITHOUT Hadamard rotation on input.

    q8_kernels' default FP8LinearFunc always applies Hadamard rotation to
    16-bit inputs before quantizing to FP8, but fp8_gemm does NOT apply the
    inverse transform — producing ~500% relative error. This forward simply
    casts the input to FP8 and calls fp8_gemm directly.
    """
    orig_shape = x.shape
    x_fp8 = x.to(torch.float8_e4m3fn).view(-1, orig_shape[-1])
    bias = self.bias.data if self.bias is not None else None
    o = _fp8_mm(x_fp8, self.weight.data, bias, False)
    return o.view(*orig_shape[:-1], self.weight.shape[0])


def _patch_transformer_q8(model):
    """Replace FP8 nn.Linear layers with q8_kernels FP8Linear for native FP8 GEMM.

    Called after transformer is loaded with fp8_cast. Swaps the upcast-forward
    Linear layers for FP8Linear which runs matmul directly in FP8.
    Uses a custom forward that skips the Hadamard rotation (see above).
    """
    count = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, torch.nn.Linear) or isinstance(module, _Q8FP8Linear):
            continue
        if module.weight.dtype != torch.float8_e4m3fn:
            continue
        fp8 = _Q8FP8Linear(
            module.in_features, module.out_features,
            bias=module.bias is not None, device=module.weight.device,
        )
        fp8.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            # fp8_gemm requires fp32 bias when IS_FP8_FAST_ACC_AVAILABLE
            fp8.bias = torch.nn.Parameter(
                module.bias.data.to(torch.float32), requires_grad=False,
            )
        # Override forward to skip Hadamard rotation
        import types
        fp8.forward = types.MethodType(_fp8_forward_no_hadamard, fp8)
        # Replace in parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], fp8)
        count += 1
    if count > 0:
        torch.cuda.empty_cache()
        logger.info("q8_kernels: replaced %d Linear → FP8Linear (native FP8 GEMM, no Hadamard)", count)


# ---------------------------------------------------------------------------
# Model loading progress — gr.Markdown 폴링 방식
# ---------------------------------------------------------------------------
# model_ledger에서 래핑할 메서드 이름 목록
_WRAPPABLE_METHODS = [
    "text_encoder", "gemma_embeddings_processor", "video_encoder",
    "transformer", "spatial_upsampler", "video_decoder",
    "audio_encoder", "audio_decoder", "vocoder",
]

# 파이프라인별 모델 로딩 순서 (짧은 표시 이름)
_PIPELINE_LOAD_ORDER = {
    "distilled": [
        "TextEnc", "Embed", "VAEEnc", "Transformer",
        "VAEDec✱", "AudioDec✱", "Vocoder✱",
        "SpatialUp", "VAEDec", "AudioDec", "Vocoder",
    ],
    "ti2vid": [
        "[S1]TextEnc", "[S1]Embed", "[S1]VAEEnc", "[S1]Transformer",
        "[S1]VAEDec✱", "[S1]AudioDec✱", "[S1]Vocoder✱",
        "[S1]VAEEnc", "[S2]SpatialUp", "[S2]Transformer",
        "[S2]VAEDec", "[S2]AudioDec", "[S2]Vocoder",
    ],
    "ti2vid_hq": [
        "[S1]TextEnc", "[S1]Embed", "[S1]VAEEnc", "[S1]Transformer",
        "[S1]VAEDec✱", "[S1]AudioDec✱", "[S1]Vocoder✱",
        "[S1]VAEEnc", "[S2]SpatialUp", "[S2]Transformer",
        "[S2]VAEDec", "[S2]AudioDec", "[S2]Vocoder",
    ],
    "retake": [
        "VAEEnc", "AudioEnc", "TextEnc", "Embed",
        "Transformer", "VAEDec", "AudioDec", "Vocoder",
    ],
    "retake_distilled": [
        "VAEEnc", "AudioEnc", "TextEnc", "Embed",
        "Transformer", "VAEDec", "AudioDec", "Vocoder",
    ],
    "keyframe": [
        "[S1]TextEnc", "[S1]Embed", "[S1]VAEEnc", "[S1]Transformer",
        "[S1]VAEDec✱", "[S1]AudioDec✱", "[S1]Vocoder✱",
        "[S2]SpatialUp", "[S2]Transformer",
        "[S2]VAEDec", "[S2]AudioDec", "[S2]Vocoder",
    ],
    "a2vid": [
        "[S1]TextEnc", "[S1]Embed", "[S1]AudioEnc", "[S1]VAEEnc", "[S1]Transformer",
        "[S1]VAEDec✱",
        "[S1]VAEEnc", "[S2]SpatialUp", "[S2]Transformer",
        "[S2]VAEDec",
    ],
    "iclora": [
        "[S1]TextEnc", "[S1]Embed", "[S1]VAEEnc", "[S1]Transformer",
        "[S1]VAEDec✱", "[S1]AudioDec✱", "[S1]Vocoder✱",
        "[S2]SpatialUp", "[S2]Transformer",
        "[S2]VAEDec", "[S2]AudioDec", "[S2]Vocoder",
    ],
}


def _wrap_model_ledger(mgr, ledger, stage_prefix: str = ""):
    """Wrap model_ledger methods — 진행 상태 추적 + 해제 감지 (캐싱 없음, 파이프라인의 메모리 관리 존중)."""
    for method_name in _WRAPPABLE_METHODS:
        if not hasattr(ledger, method_name):
            continue
        original = getattr(ledger, method_name)
        if getattr(original, "_wrapped", False):
            continue

        def _make_wrapper(orig, mname):
            def wrapper(*args, **kwargs):
                load_order = _PIPELINE_LOAD_ORDER.get(mgr.current_type, [])
                total = len(load_order) if load_order else 1
                idx = len(mgr._loaded_names)
                current = load_order[idx] if idx < len(load_order) else f"model_{idx + 1}"

                mgr._current_loading = current
                logger.info("Loading %s (%d/%d)...", current, idx + 1, total)

                # Send "loading_start" to main process via IPC
                if mgr.progress_queue is not None and mgr._current_task_id is not None:
                    try:
                        mgr.progress_queue.put_nowait({
                            "task_id": mgr._current_task_id,
                            "type": "loading_start",
                            "data": {"name": current, "index": idx + 1, "total": total},
                        })
                    except Exception:
                        pass

                # tqdm 바 — gr.Progress(track_tqdm=True)가 캡처
                desc = f"Loading {current} ({idx + 1}/{total})"
                if mgr._loading_bar is not None:
                    mgr._loading_bar.close()
                mgr._loading_bar = tqdm(total=total, initial=idx, desc=desc, unit="model", leave=False)

                t0 = time.time()
                result = orig(*args, **kwargs)

                # Apply q8_kernels FP8 matmul optimization for transformer
                if mname == "transformer" and _Q8_AVAILABLE:
                    _patch_transformer_q8(result)

                elapsed = time.time() - t0

                mgr._loading_bar.update(1)

                mgr._loaded_names.append(current)
                mgr._load_times.append(elapsed)
                mgr._current_loading = None
                logger.info("%s loaded in %.0fs (%d/%d)", current, elapsed, idx + 1, total)

                # Send "loading_done" to main process via IPC queue
                if mgr.progress_queue is not None and mgr._current_task_id is not None:
                    try:
                        mgr.progress_queue.put_nowait({
                            "task_id": mgr._current_task_id,
                            "type": "loading_done",
                            "data": {"name": current, "index": idx + 1,
                                     "total": total, "elapsed": elapsed},
                        })
                    except Exception:
                        pass

                # 모델 해제 감지 — del 시 상태 업데이트
                model_name = current
                def _on_unload():
                    mgr._mark_unloaded(model_name)
                    logger.info("%s unloaded", model_name)
                weakref.finalize(result, _on_unload)

                return result
            wrapper._wrapped = True
            return wrapper

        setattr(ledger, method_name, _make_wrapper(original, method_name))


# ---------------------------------------------------------------------------
# Defaults & Constants
# ---------------------------------------------------------------------------
DEFAULTS = dataclasses.replace(
    LTX_2_3_PARAMS,
    video_guider_params=dataclasses.replace(
        LTX_2_3_PARAMS.video_guider_params,
        cfg_scale=2.0,
        stg_scale=0.4,
        rescale_scale=0.85,
    ),
    num_inference_steps=25,
)
from config import MODEL_DIR as DEFAULT_MODEL_DIR, OUTPUT_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(str(DEFAULT_MODEL_DIR), exist_ok=True)

RESOLUTION_CHOICES = [
    "1920x1080", "1080x1920",
    "1280x720", "720x1280",
    "1024x1536", "1536x1024",
    "1024x1024",
    "768x1024", "1024x768",
    "512x768", "768x512",
    "544x960", "960x544",
]

FRAME_CHOICES = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121, 161, 193]

SAMPLE_PROMPTS = [
    "A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable parsing. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and the background is softly blurred, suggesting an indoor setting.",
    "A golden retriever puppy runs through a sunlit meadow, tall grass swaying gently in the breeze. The camera follows low to the ground, capturing the puppy's joyful expression as it bounds forward. Warm afternoon light creates a soft golden glow across the scene.",
    "Aerial drone shot of ocean waves crashing against rocky cliffs at sunset. The camera slowly pulls back to reveal the dramatic coastline. Mist rises from the impact of waves, catching orange and pink light from the setting sun.",
    "A steaming cup of coffee sits on a wooden table by a rain-streaked window. Water droplets slowly trace paths down the glass. The camera slowly racks focus from the cup to the blurred city lights outside. Warm interior lighting contrasts with the cool blue tones of the rainy evening.",
    "Time-lapse of a flower blooming in a dark studio. A single white lily unfurls its petals under soft directional lighting. The background is completely black, making the flower appear to glow. The camera is static, capturing every subtle movement of the petals opening.",
]

REQUIRED_MODELS = {
    "ti2vid": ["ltx-2.3-22b-dev-fp8.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
               "ltx-2.3-22b-distilled-lora-384.safetensors", "gemma-3-12b-it-qat-q4_0-unquantized",
               "Huihui-Qwen3.5-4B-abliterated"],
    "distilled": ["ltx-2.3-22b-distilled-fp8.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
                  "gemma-3-12b-it-qat-q4_0-unquantized", "Huihui-Qwen3.5-4B-abliterated"],
    "iclora": ["ltx-2.3-22b-distilled-fp8.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
               "gemma-3-12b-it-qat-q4_0-unquantized", "Huihui-Qwen3.5-4B-abliterated"],
    "keyframe": ["ltx-2.3-22b-dev-fp8.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
                 "ltx-2.3-22b-distilled-lora-384.safetensors", "gemma-3-12b-it-qat-q4_0-unquantized",
                 "Huihui-Qwen3.5-4B-abliterated"],
    "a2vid": ["ltx-2.3-22b-dev-fp8.safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
              "ltx-2.3-22b-distilled-lora-384.safetensors", "gemma-3-12b-it-qat-q4_0-unquantized",
              "Huihui-Qwen3.5-4B-abliterated"],
    "retake": ["ltx-2.3-22b-dev-fp8.safetensors", "gemma-3-12b-it-qat-q4_0-unquantized",
               "Huihui-Qwen3.5-4B-abliterated"],
}

IC_LORA_MAP = {
    "Union Control": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
    "Motion Track": "ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors",
}

LORAS_DIR = Path(DEFAULT_MODEL_DIR) / "loras"
os.makedirs(str(LORAS_DIR), exist_ok=True)


def scan_lora_files() -> list[str]:
    """Scan {model_dir}/loras/ for .safetensors files. Returns sorted filenames."""
    loras_dir = Path(DEFAULT_MODEL_DIR) / "loras"
    if not loras_dir.exists():
        return []
    return sorted(f.name for f in loras_dir.glob("*.safetensors"))


# ---------------------------------------------------------------------------
# Pipeline Manager
# ---------------------------------------------------------------------------
class PipelineManager:
    """Manages pipeline lifecycle — one active pipeline at a time."""

    def __init__(self, progress_queue=None) -> None:
        self.current_pipeline = None
        self.current_type: str | None = None
        self.model_dir: str = str(DEFAULT_MODEL_DIR)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._iclora_path: str | None = None
        self._lora_strength: float = 0.8
        self._custom_loras: tuple = ()
        # Loading progress state
        self._loaded_names: list[str] = []
        self._load_times: list[float] = []
        self._current_loading: str | None = None
        self._is_generating: bool = False
        self._unloaded_names: set[str] = set()
        self._loading_bar = None
        # IPC progress queue (set by worker process)
        self.progress_queue = progress_queue
        self._current_task_id: str | None = None

    def start_loading_bar(self):
        """Reset loading state before pipeline call."""
        self._loaded_names = []
        self._load_times = []
        self._current_loading = None
        self._unloaded_names = set()
        self._is_generating = True
        self._loading_bar = None

        # Send loading plan to main process via IPC
        load_order = _PIPELINE_LOAD_ORDER.get(self.current_type, [])
        if self.progress_queue is not None and self._current_task_id is not None and load_order:
            try:
                self.progress_queue.put_nowait({
                    "task_id": self._current_task_id,
                    "type": "loading_plan",
                    "data": {"plan": load_order, "total": len(load_order)},
                })
            except Exception:
                pass

    def stop_loading_bar(self):
        """Clear loading state after pipeline call."""
        if self._loading_bar is not None:
            self._loading_bar.close()
            self._loading_bar = None
        self._current_loading = None
        self._is_generating = False

    def _mark_unloaded(self, model_name: str):
        """Called by weakref finalizer when a model is garbage collected."""
        self._unloaded_names.add(model_name)

    def get_loading_status(self) -> str:
        """Return current model loading status as Markdown text (polled by gr.Markdown every=1).

        Note: During generation, Gradio queue may block this polling.
        Real-time progress is shown via tqdm bars captured by gr.Progress(track_tqdm=True).
        This Markdown shows the final summary after generation completes.
        """
        if not self._is_generating and not self._loaded_names:
            return ""
        load_order = _PIPELINE_LOAD_ORDER.get(self.current_type, [])
        if not load_order:
            return ""
        total = len(load_order)
        lines = []
        for i, name in enumerate(load_order):
            if i < len(self._loaded_names):
                loaded_name = self._loaded_names[i]
                t_val = self._load_times[i] if i < len(self._load_times) else -1
                if loaded_name in self._unloaded_names:
                    lines.append(f"- ~~{loaded_name}~~: freed")
                elif t_val == 0:
                    lines.append(f"- **{loaded_name}**: cached")
                else:
                    lines.append(f"- **{loaded_name}**: ok ({t_val:.0f}s)")
            elif name == self._current_loading:
                lines.append(f"- **{name}**: loading...")
            else:
                lines.append(f"- {name}: wait")
        loaded = len(self._loaded_names)
        header = f"**Model Loading [{loaded}/{total}]**"
        if loaded == total and self._load_times:
            total_time = sum(self._load_times)
            if total_time > 0:
                header += f" — done in {total_time:.0f}s"
            else:
                header += " — all cached"
        return header + "\n" + "\n".join(lines)

    def _cleanup(self) -> None:
        if self.current_pipeline is not None:
            logger.info("Cleaning up pipeline: %s", self.current_type)
            del self.current_pipeline
            self.current_pipeline = None
            self.current_type = None
            self._iclora_path = None
            self._custom_loras = ()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def check_models(self, pipeline_type: str) -> list[str]:
        """Return list of missing model files for a pipeline type."""
        required = REQUIRED_MODELS.get(pipeline_type, [])
        missing = []
        for f in required:
            path = Path(self.model_dir) / f
            if not path.exists():
                missing.append(f)
        return missing

    def _model_path(self, filename: str) -> str:
        return str(Path(self.model_dir) / filename)

    def _gemma_root(self) -> str:
        return self._model_path("gemma-3-12b-it")

    def _distilled_lora(self, strength: float = 0.8) -> list[LoraPathStrengthAndSDOps]:
        return [LoraPathStrengthAndSDOps(
            self._model_path("ltx-2.3-22b-distilled-lora-384.safetensors"),
            strength,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )]

    def _build_custom_lora_list(self, custom_loras: list[dict]) -> list[LoraPathStrengthAndSDOps]:
        """Convert UI custom LoRA dicts to pipeline LoRA specs."""
        result = []
        loras_dir = Path(self.model_dir) / "loras"
        for entry in custom_loras:
            path = loras_dir / entry["filename"]
            if path.exists():
                result.append(LoraPathStrengthAndSDOps(
                    str(path), entry["strength"], LTXV_LORA_COMFY_RENAMING_MAP,
                ))
            else:
                logger.warning("Custom LoRA not found: %s", path)
        return result

    @staticmethod
    def _lora_cache_key(custom_loras: list[dict]) -> tuple:
        return tuple((d["filename"], d["strength"]) for d in custom_loras) if custom_loras else ()

    def get_ti2vid(self, sampler: str = "euler", lora_strength: float = 0.8,
                   custom_loras: list[dict] | None = None, quantization=None):
        custom_loras = custom_loras or []
        lora_key = self._lora_cache_key(custom_loras)
        target = "ti2vid_hq" if sampler == "res_2s" else "ti2vid"
        if (self.current_type == target and self._lora_strength == lora_strength
                and self._custom_loras == lora_key):
            return self.current_pipeline
        self._cleanup()
        self._lora_strength = lora_strength
        self._custom_loras = lora_key
        custom_lora_list = self._build_custom_lora_list(custom_loras)
        if sampler == "res_2s":
            from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
            self.current_pipeline = TI2VidTwoStagesHQPipeline(
                checkpoint_path=self._model_path("ltx-2.3-22b-dev-fp8.safetensors"),
                distilled_lora=self._distilled_lora(lora_strength),
                distilled_lora_strength_stage_1=lora_strength * 0.375,
                distilled_lora_strength_stage_2=lora_strength,
                spatial_upsampler_path=self._model_path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
                gemma_root=self._gemma_root(),
                loras=custom_lora_list,
                device=self.device,
                quantization=FP8_QUANTIZATION,
            )
        else:
            from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
            self.current_pipeline = TI2VidTwoStagesPipeline(
                checkpoint_path=self._model_path("ltx-2.3-22b-dev-fp8.safetensors"),
                distilled_lora=self._distilled_lora(lora_strength),
                spatial_upsampler_path=self._model_path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
                gemma_root=self._gemma_root(),
                loras=custom_lora_list,
                device=self.device,
                quantization=FP8_QUANTIZATION,
            )
        self.current_type = target
        self._wrap_current_pipeline()
        return self.current_pipeline

    def get_distilled(self, quantization=None):
        if self.current_type == "distilled":
            return self.current_pipeline
        self._cleanup()
        from ltx_pipelines.distilled import DistilledPipeline
        self.current_pipeline = DistilledPipeline(
            distilled_checkpoint_path=self._model_path("ltx-2.3-22b-distilled-fp8.safetensors"),
            gemma_root=self._gemma_root(),
            spatial_upsampler_path=self._model_path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            loras=[],
            device=self.device,
            quantization=FP8_QUANTIZATION,
        )
        self.current_type = "distilled"
        self._wrap_current_pipeline()
        return self.current_pipeline

    def get_retake(self, distilled: bool = False, quantization=None):
        target = "retake_distilled" if distilled else "retake"
        if self.current_type == target:
            return self.current_pipeline
        self._cleanup()
        from ltx_pipelines.retake import RetakePipeline
        ckpt = "ltx-2.3-22b-distilled-fp8.safetensors" if distilled else "ltx-2.3-22b-dev-fp8.safetensors"
        self.current_pipeline = RetakePipeline(
            checkpoint_path=self._model_path(ckpt),
            gemma_root=self._gemma_root(),
            loras=[],
            device=self.device,
            quantization=FP8_QUANTIZATION,
        )
        self.current_type = target
        self._wrap_current_pipeline()
        return self.current_pipeline

    def get_keyframe(self, lora_strength: float = 0.8,
                     custom_loras: list[dict] | None = None, quantization=None):
        custom_loras = custom_loras or []
        lora_key = self._lora_cache_key(custom_loras)
        if (self.current_type == "keyframe" and self._lora_strength == lora_strength
                and self._custom_loras == lora_key):
            return self.current_pipeline
        self._cleanup()
        self._lora_strength = lora_strength
        self._custom_loras = lora_key
        custom_lora_list = self._build_custom_lora_list(custom_loras)
        from ltx_pipelines.keyframe_interpolation import KeyframeInterpolationPipeline
        self.current_pipeline = KeyframeInterpolationPipeline(
            checkpoint_path=self._model_path("ltx-2.3-22b-dev-fp8.safetensors"),
            distilled_lora=self._distilled_lora(lora_strength),
            spatial_upsampler_path=self._model_path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            gemma_root=self._gemma_root(),
            loras=custom_lora_list,
            device=self.device,
            quantization=FP8_QUANTIZATION,
        )
        self.current_type = "keyframe"
        self._wrap_current_pipeline()
        return self.current_pipeline

    def get_a2vid(self, lora_strength: float = 0.8,
                  custom_loras: list[dict] | None = None, quantization=None):
        custom_loras = custom_loras or []
        lora_key = self._lora_cache_key(custom_loras)
        if (self.current_type == "a2vid" and self._lora_strength == lora_strength
                and self._custom_loras == lora_key):
            return self.current_pipeline
        self._cleanup()
        self._lora_strength = lora_strength
        self._custom_loras = lora_key
        custom_lora_list = self._build_custom_lora_list(custom_loras)
        from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
        self.current_pipeline = A2VidPipelineTwoStage(
            checkpoint_path=self._model_path("ltx-2.3-22b-dev-fp8.safetensors"),
            distilled_lora=self._distilled_lora(lora_strength),
            spatial_upsampler_path=self._model_path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            gemma_root=self._gemma_root(),
            loras=custom_lora_list,
            device=self.device,
            quantization=FP8_QUANTIZATION,
        )
        self.current_type = "a2vid"
        self._wrap_current_pipeline()
        return self.current_pipeline

    def get_iclora(self, lora_paths: list[str], lora_strength: float = 1.0,
                   distilled_lora_strength: float = 0.8,
                   custom_loras: list[dict] | None = None, quantization=None):
        custom_loras = custom_loras or []
        lora_key = self._lora_cache_key(custom_loras)
        cache_key = (tuple(lora_paths), distilled_lora_strength, lora_key)
        if self.current_type == "iclora" and self._iclora_cache_key == cache_key:
            return self.current_pipeline
        self._cleanup()
        from ltx_pipelines.ic_lora import ICLoraPipeline
        ic_loras = [LoraPathStrengthAndSDOps(p, lora_strength, LTXV_LORA_COMFY_RENAMING_MAP)
                    for p in lora_paths]
        custom_lora_list = self._build_custom_lora_list(custom_loras)
        all_loras = ic_loras + custom_lora_list
        self.current_pipeline = ICLoraPipeline(
            distilled_checkpoint_path=self._model_path("ltx-2.3-22b-distilled-fp8.safetensors"),
            spatial_upsampler_path=self._model_path("ltx-2.3-spatial-upscaler-x2-1.0.safetensors"),
            gemma_root=self._gemma_root(),
            loras=all_loras,
            device=self.device,
            quantization=FP8_QUANTIZATION,
            distilled_lora=tuple(self._distilled_lora(distilled_lora_strength)),
        )
        self.current_type = "iclora"
        self._iclora_cache_key = cache_key
        self._wrap_current_pipeline()
        return self.current_pipeline

    def _wrap_current_pipeline(self):
        """Wrap all model_ledger instances in the current pipeline with progress notifications."""
        p = self.current_pipeline
        if p is None:
            return
        if hasattr(p, "stage_1_model_ledger"):
            _wrap_model_ledger(self, p.stage_1_model_ledger, "Stage 1")
        if hasattr(p, "stage_2_model_ledger"):
            _wrap_model_ledger(self, p.stage_2_model_ledger, "Stage 2")
        if hasattr(p, "model_ledger"):
            _wrap_model_ledger(self, p.model_ledger)


pipeline_mgr = PipelineManager()
