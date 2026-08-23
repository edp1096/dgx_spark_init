#!/usr/bin/env python3
"""OpenAI-compatible facade for Krea 2 Turbo NVFP4 on ComfyUI."""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageFilter, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field


COMFY_URL = "http://127.0.0.1:8188"
MODEL_ID = "krea2-turbo-nvfp4"
MODEL_ALIASES = {MODEL_ID, "krea/Krea-2-Turbo"}
DIFFUSION_MODEL = "krea2_turbo_nvfp4.safetensors"
STYLE_REFERENCE_MODEL = "krea2_turbo_int8_convrot.safetensors"
TEXT_ENCODER = "qwen3vl_4b_fp8_scaled.safetensors"
VISION_TEXT_ENCODER = "qwen3vl_4b_bf16.safetensors"
VISION_INSTRUCT_SYSTEM = (
    "Describe the key features of the reference image (color, shape, size, texture, objects, "
    "background), then explain how the user's instruction should combine with or alter it, and "
    "generate a new image meeting the instruction while staying consistent with the reference "
    "where appropriate:"
)
VAE = "qwen_image_vae.safetensors"
REAL_VAE = "krea2RealVae_v10.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
DEPTH_CONTROL_LORA = "krea2-depth-control-lora.safetensors"
IDENTITY_EDIT_LORA = "krea2_identity_edit_v1_2.safetensors"
FILTER_BYPASS_BALANCED = "fedor_bypass.safetensors"
FILTER_BYPASS_STRONG = "krea2filterbypass3.safetensors"
FILTER_BYPASS_ADHERENCE = "user/skc3vo.safetensors"
DETAIL_ENHANCER_LORA = "krea-detail-enhancer-exp.safetensors"
NK2E_EDIT_LORA = "NK2E-v0.3.safetensors"
NK2E_CANNY_LORA = "NK2E-canny-v0.1.safetensors"
ANYPAINT_LORA = "krea2_anypaint_rank32.safetensors"
STYLE_REFERENCE_LORA = "krea2_style_reference.safetensors"
STYLE_TRIGGERS = {
    "darkbrush": "monochrome ink wash style",
    "dotmatrix": "monochrome stippling style",
    "kidsdrawing": "naive expressive sketch style",
    "neondrip": "textured abstract style",
    "rainywindow": "rainy window style",
    "retroanime": "purple retro anime style",
    "softwatercolor": "art deco watercolor style",
    "sunsetblur": "ethereal motion blur style",
    "vintagetarot": "vintage tarot style",
}
STYLE_LORAS = {name: f"krea2_{name}.safetensors" for name in STYLE_TRIGGERS}
USER_LORA_ROOT = (Path("/opt/ComfyUI/models/loras/user")).resolve()
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
OUTPUT_ROOT = Path("/opt/ComfyUI/output").resolve()
INPUT_ROOT = Path("/opt/ComfyUI/input").resolve()
generation_lock = asyncio.Lock()
depth_processor: Any | None = None
depth_model: Any | None = None


class StyleSelection(BaseModel):
    name: str
    strength: float = 1.0


class UserLoRASelection(BaseModel):
    filename: str
    strength: float = 1.0


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    prompt: str
    model: str = MODEL_ID
    n: int = 1
    size: str = "1024x1024"
    seed: int | None = None
    response_format: str = "b64_json"
    control_image: str | None = None
    control_strength: float = 1.0
    source_image: str | None = None
    reference_image: str | None = None
    identity_mask: str | None = None
    strict_mask: str | None = None
    strict_mask_grow: int = 0
    strict_mask_feather: float = 0.0
    vae_mode: str = "default"
    identity_fit_mode: str = "fit"
    identity_strength: float = 1.0
    ref_boost: float = 4.0
    grounding_px: int = 768
    steps: int | None = None
    sampler_name: str | None = None
    scheduler: str | None = None
    style: str | None = None
    style_strength: float = 1.0
    styles: list[StyleSelection] = Field(default_factory=list)
    user_loras: list[UserLoRASelection] = Field(default_factory=list)
    vision_images: list[str] = Field(default_factory=list)
    vision_mode: str = "descriptor"
    vision_megapixels: float = 1.0
    style_reference_images: list[str] = Field(default_factory=list)
    style_reference_strength: float = 1.0
    nk2e_image: str | None = None
    nk2e_mode: str = "edit"
    nk2e_strength: float = 0.7
    nk2e_preprocessed: bool = False
    anypaint_image: str | None = None
    anypaint_mask: str | None = None
    outpaint_left: int = 0
    outpaint_top: int = 0
    outpaint_right: int = 0
    outpaint_bottom: int = 0
    anypaint_strength: float = 1.0
    anypaint_reference_max_edge: int = 384
    anypaint_boundary_redraw_px: int = 32
    anypaint_vlm_reference: bool = True
    filter_mode: str = "balanced"
    filter_strength: float | None = None
    prompt_enhancer: bool = False
    prompt_enhancer_strength: float = 1.0
    prompt_text_scale: float = 1.75
    detail_enhance_image: str | None = None
    detail_strength: float = 1.0
    detail_vae: str = "wan"


app = FastAPI(title="Krea 2 Turbo NVFP4 API")


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="size must be WIDTHxHEIGHT") from exc
    if not (512 <= width <= 2048 and 512 <= height <= 2048):
        raise HTTPException(status_code=400, detail="width and height must be between 512 and 2048")
    if width % 16 or height % 16:
        raise HTTPException(status_code=400, detail="width and height must be multiples of 16")
    return width, height


def workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    steps: int = 8,
    styles: list[StyleSelection] | None = None,
    user_loras: list[UserLoRASelection] | None = None,
) -> dict[str, Any]:
    graph = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }
    model_input: list[Any] = ["1", 0]
    for offset, style in enumerate(styles or []):
        node_id = str(20 + offset)
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": STYLE_LORAS[style.name],
                "strength_model": style.strength,
            },
        }
        model_input = [node_id, 0]
    next_id = 20 + len(styles or [])
    for selection in user_loras or []:
        node_id = str(next_id)
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": f"user/{selection.filename}",
                "strength_model": selection.strength,
            },
        }
        model_input = [node_id, 0]
        next_id += 1
    graph["7"]["inputs"]["model"] = model_input
    return graph


def apply_sampling(graph: dict[str, Any], sampler_name: str, scheduler: str) -> dict[str, Any]:
    """Apply the validated sampling pair to both basic and advanced ComfyUI graphs."""
    for node in graph.values():
        class_type = node.get("class_type")
        inputs = node.get("inputs", {})
        if class_type == "KSampler":
            inputs["sampler_name"] = sampler_name
            inputs["scheduler"] = scheduler
        elif class_type == "KSamplerSelect":
            inputs["sampler_name"] = sampler_name
        elif class_type == "BasicScheduler":
            inputs["scheduler"] = scheduler
    return graph


def apply_filter_bypass(
    graph: dict[str, Any], mode: str, strength: float | None
) -> dict[str, Any]:
    """Apply a selectable filter-relaxation vector before every model adapter."""
    if mode == "off":
        return graph
    lora_name = {
        "adherence": FILTER_BYPASS_ADHERENCE,
        "balanced": FILTER_BYPASS_BALANCED,
        "strong": FILTER_BYPASS_STRONG,
    }[mode]
    default_strength = 0.05 if mode == "adherence" else 1.0
    resolved_strength = strength if strength is not None else default_strength
    patch_id = "900"
    graph[patch_id] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["1", 0],
            "lora_name": lora_name,
            "strength_model": resolved_strength,
        },
    }
    for node_id, node in graph.items():
        if node_id == patch_id:
            continue
        inputs = node.get("inputs", {})
        if inputs.get("model") == ["1", 0]:
            inputs["model"] = [patch_id, 0]
    return graph


def apply_prompt_enhancer(
    graph: dict[str, Any], enabled: bool, strength: float, text_scale: float
) -> dict[str, Any]:
    """Wrap each sampler's final model input with the Krea2T prompt enhancer."""
    if not enabled:
        return graph
    next_id = 910
    for node in list(graph.values()):
        if node.get("class_type") not in {"KSampler", "CFGGuider"}:
            continue
        model_input = node.get("inputs", {}).get("model")
        if model_input is None:
            continue
        while str(next_id) in graph:
            next_id += 1
        node_id = str(next_id)
        graph[node_id] = {
            "class_type": "Krea2T-Enhancer-Advanced",
            "inputs": {
                "model": model_input,
                "enabled": True,
                "strength": strength,
                "text_scale": text_scale,
                "debug": False,
            },
        }
        node["inputs"]["model"] = [node_id, 0]
        next_id += 1
    return graph


def apply_vision_conditioning(
    graph: dict[str, Any],
    prompt: str,
    image_names: list[str],
    mode: str,
    vision_megapixels: float,
) -> dict[str, Any]:
    graph["2"]["inputs"]["clip_name"] = VISION_TEXT_ENCODER
    encoder_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": prompt,
        "vision_megapixels": vision_megapixels,
        "mask_padding": 0.0,
        "vision_position": "before prompt",
        "print_prompt": False,
    }
    for index, image_name in enumerate(image_names, start=1):
        node_id = str(79 + index)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        encoder_inputs[f"image{index}"] = [node_id, 0]
    if mode == "instruct":
        graph["70"] = {
            "class_type": "Krea2SystemPrompt",
            "inputs": {"text": VISION_INSTRUCT_SYSTEM},
        }
        encoder_inputs["system_prompt"] = ["70", 0]
    graph["4"] = {"class_type": "TextEncodeKrea2", "inputs": encoder_inputs}
    return graph


def style_reference_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    image_names: list[str],
    strength: float,
    steps: int,
) -> dict[str, Any]:
    """Official Ostris/ComfyUI Krea 2 style-reference graph."""
    encode_inputs: dict[str, Any] = {
        "clip": ["2", 0],
        "prompt": prompt,
        "vae": ["3", 0],
    }
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": STYLE_REFERENCE_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "4": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": encode_inputs},
        "5": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["4", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "6": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "7": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": STYLE_REFERENCE_LORA, "strength_model": strength},
        },
        "8": {
            "class_type": "ModelSamplingFlux",
            "inputs": {"model": ["7", 0], "max_shift": 1.15, "base_shift": 0.5, "width": width, "height": height},
        },
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "11": {"class_type": "CFGGuider", "inputs": {"model": ["8", 0], "positive": ["5", 0], "negative": ["6", 0], "cfg": 1.0}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "13": {"class_type": "BasicScheduler", "inputs": {"model": ["8", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["9", 0]}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["3", 0]}},
        "16": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["15", 0]}},
    }
    for index, image_name in enumerate(image_names, start=1):
        node_id = str(79 + index)
        graph[node_id] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        encode_inputs[f"image{index}"] = [node_id, 0]
    return graph


def detail_enhance_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    source_name: str,
    strength: float,
    steps: int,
    vae_name: str,
) -> dict[str, Any]:
    """Ostris edit graph used by the experimental Krea detail-enhancer LoRA."""
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": VISION_TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}},
        "4": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "5": {
            "class_type": "TextEncodeKrea2OstrisEdit",
            "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image1": ["4", 0]},
        },
        "6": {
            "class_type": "TextEncodeKrea2OstrisEdit",
            "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["4", 0]},
        },
        "7": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["5", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "8": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "9": {
            "class_type": "Krea2OstrisEditModelPatch",
            "inputs": {"model": ["1", 0], "kv_cache": False},
        },
        "10": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["9", 0], "lora_name": DETAIL_ENHANCER_LORA, "strength_model": strength},
        },
        "11": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "12": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["10", 0],
                "positive": ["7", 0],
                "negative": ["8", 0],
                "latent_image": ["11", 0],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "er_sde",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["13", 0]}},
    }
    return graph


def nk2e_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    reference_name: str,
    mode: str,
    strength: float,
    steps: int,
) -> dict[str, Any]:
    graph = workflow(prompt, width, height, seed, prefix, steps)
    graph.update(
        {
            "10": {"class_type": "LoadImage", "inputs": {"image": reference_name}},
            "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
            "12": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["1", 0],
                    "lora_name": NK2E_CANNY_LORA if mode == "canny" else NK2E_EDIT_LORA,
                    "strength_model": strength,
                },
            },
            "13": {"class_type": "NK2EInContextModelNode", "inputs": {"model": ["12", 0]}},
            "14": {
                "class_type": "NK2ESetReferenceNode",
                "inputs": {"conditioning": ["4", 0], "reference": ["11", 0]},
            },
        }
    )
    graph["7"]["inputs"].update({"model": ["13", 0], "positive": ["14", 0]})
    return graph


def anypaint_workflow(
    prompt: str,
    seed: int,
    prefix: str,
    source_name: str,
    mask_name: str | None,
    left: int,
    top: int,
    right: int,
    bottom: int,
    strength: float,
    reference_max_edge: int,
    boundary_redraw_px: int,
    vlm_reference: bool,
    steps: int,
) -> dict[str, Any]:
    prepare_inputs: dict[str, Any] = {
        "source": ["10", 0],
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "reference_max_edge": reference_max_edge,
        "boundary_redraw_px": boundary_redraw_px,
    }
    graph: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": DIFFUSION_MODEL, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "10": {"class_type": "LoadImage", "inputs": {"image": source_name}},
        "20": {"class_type": "Krea2AnyPaintPrepare", "inputs": prepare_inputs},
        "21": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": ANYPAINT_LORA, "strength_model": strength},
        },
        "22": {"class_type": "Krea2AnyPaintModelPatch", "inputs": {"model": ["21", 0], "kv_cache": True}},
        "23": {
            "class_type": "Krea2AnyPaintEncode",
            "inputs": {
                "clip": ["2", 0],
                "prompt": prompt,
                "vae": ["3", 0],
                "semantic_reference": ["20", 0],
                "known_image": ["20", 1],
                "keep_mask": ["20", 3],
                "vlm_reference": vlm_reference,
            },
        },
        "24": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "25": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["22", 0],
                "positive": ["23", 0],
                "negative": ["24", 0],
                "latent_image": ["23", 1],
                "seed": seed,
                "steps": steps,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
            },
        },
        "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["3", 0]}},
        "27": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["26", 0]}},
    }
    if mask_name is not None:
        graph["11"] = {"class_type": "LoadImage", "inputs": {"image": mask_name}}
        graph["12"] = {"class_type": "ImageToMask", "inputs": {"image": ["11", 0], "channel": "red"}}
        prepare_inputs["generated_mask"] = ["12", 0]
    return graph


def depth_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    depth_name: str,
    control_strength: float,
    steps: int,
    styles: list[StyleSelection],
    user_loras: list[UserLoRASelection],
) -> dict[str, Any]:
    graph = workflow(prompt, width, height, seed, prefix, steps, styles, user_loras)
    model_input = graph["7"]["inputs"]["model"]
    graph.update(
        {
            "10": {"class_type": "LoadImage", "inputs": {"image": depth_name}},
            "11": {
                "class_type": "Krea2ControlLoRALoader",
                "inputs": {
                    "model": model_input,
                    "lora_name": DEPTH_CONTROL_LORA,
                    "strength": control_strength,
                },
            },
            "12": {
                "class_type": "Krea2ControlImageEncode",
                "inputs": {
                    "control_image": ["10", 0],
                    "vae": ["3", 0],
                    "resize": "match_latent_size",
                    "upscale_method": "lanczos",
                    "crop": "center",
                    "channel_mode": "grayscale",
                    "normalize": "per_image_minmax",
                    "invert": False,
                    "batch_mode": "independent_images",
                    "latent": ["6", 0],
                },
            },
            "13": {
                "class_type": "Krea2ControlApply",
                "inputs": {"model": ["11", 0], "control_latent": ["12", 0]},
            },
        }
    )
    graph["7"]["inputs"]["model"] = ["13", 0]
    return graph


def identity_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    prefix: str,
    source_name: str,
    reference_name: str | None,
    identity_mask_name: str | None,
    identity_strength: float,
    ref_boost: float,
    grounding_px: int,
    steps: int,
    styles: list[StyleSelection],
    user_loras: list[UserLoRASelection],
    depth_name: str | None,
    control_strength: float,
    fit_mode: str,
) -> dict[str, Any]:
    graph = workflow(prompt, width, height, seed, prefix, steps)
    graph.update(
        {
            "10": {"class_type": "LoadImage", "inputs": {"image": source_name}},
            "11": {"class_type": "VAEEncode", "inputs": {"pixels": ["10", 0], "vae": ["3", 0]}},
            "12": {
                "class_type": "EmptySD3LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1},
            },
            "13": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["1", 0],
                    "lora_name": IDENTITY_EDIT_LORA,
                    "strength_model": identity_strength,
                },
            },
            "15": {
                "class_type": "Krea2EditModelPatch",
                "inputs": {
                    "model": ["13", 0],
                    "source_latent": ["11", 0],
                    "ref_boost": ref_boost,
                    "ref_boost_a": 1.0,
                    "fit_mode": "crop (legacy)" if fit_mode == "crop" else "fit",
                    "vae": ["3", 0],
                    "source_image": ["10", 0],
                    "target_latent": ["12", 0],
                },
            },
            "16": {
                "class_type": "Krea2EditGroundedEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "prompt": prompt,
                    "grounding_px": grounding_px,
                    "image": ["10", 0],
                },
            },
            "17": {
                "class_type": "Krea2EditGroundedEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "prompt": "",
                    "grounding_px": grounding_px,
                    "image": ["10", 0],
                },
            },
        }
    )
    model_input: list[Any] = ["13", 0]
    next_id = 20
    for style in styles:
        graph[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": STYLE_LORAS[style.name],
                "strength_model": style.strength,
            },
        }
        model_input = [str(next_id), 0]
        next_id += 1
    for selection in user_loras:
        graph[str(next_id)] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_input,
                "lora_name": f"user/{selection.filename}",
                "strength_model": selection.strength,
            },
        }
        model_input = [str(next_id), 0]
        next_id += 1
    if depth_name is not None:
        graph[str(next_id)] = {"class_type": "LoadImage", "inputs": {"image": depth_name}}
        depth_load_id = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "Krea2ControlLoRALoader",
            "inputs": {
                "model": model_input,
                "lora_name": DEPTH_CONTROL_LORA,
                "strength": control_strength,
            },
        }
        depth_lora_id = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "Krea2ControlImageEncode",
            "inputs": {
                "control_image": [depth_load_id, 0],
                "vae": ["3", 0],
                "resize": "match_latent_size",
                "upscale_method": "lanczos",
                "crop": "center",
                "channel_mode": "grayscale",
                "normalize": "per_image_minmax",
                "invert": False,
                "batch_mode": "independent_images",
                "latent": ["12", 0],
            },
        }
        depth_encode_id = str(next_id)
        next_id += 1
        graph[str(next_id)] = {
            "class_type": "Krea2ControlApply",
            "inputs": {"model": [depth_lora_id, 0], "control_latent": [depth_encode_id, 0]},
        }
        model_input = [str(next_id), 0]
    graph["15"]["inputs"]["model"] = model_input

    if reference_name is not None:
        graph["18"] = {"class_type": "LoadImage", "inputs": {"image": reference_name}}
        graph["19"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["18", 0], "vae": ["3", 0]}}
        graph["15"]["inputs"].update(
            {"source_latent_b": ["19", 0], "source_image_b": ["18", 0]}
        )
        graph["16"]["inputs"]["image_b"] = ["18", 0]
        graph["17"]["inputs"]["image_b"] = ["18", 0]

    if identity_mask_name is not None:
        graph["14"] = {
            "class_type": "LoadImageMask",
            "inputs": {"image": identity_mask_name, "channel": "red"},
        }
        graph["15"]["inputs"]["ref_boost_mask"] = ["14", 0]

    graph["7"]["inputs"].update(
        {
            "model": ["15", 0],
            "positive": ["16", 0],
            "negative": ["17", 0],
            "latent_image": ["12", 0],
        }
    )
    return graph


def decode_image(encoded: str) -> Image.Image:
    if encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[-1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image must be valid base64") from exc
    if len(raw) > 32 << 20:
        raise HTTPException(status_code=400, detail="image exceeds 32 MiB")
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="image is not a valid image") from exc


def make_depth_image(source: Image.Image) -> Image.Image:
    global depth_processor, depth_model

    import torch
    import torch.nn.functional as functional
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    if depth_processor is None or depth_model is None:
        depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).eval()

    inputs = depth_processor(images=source, return_tensors="pt")
    with torch.inference_mode():
        prediction = depth_model(**inputs).predicted_depth
    prediction = functional.interpolate(
        prediction.unsqueeze(1),
        size=(source.height, source.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    prediction -= prediction.min()
    prediction /= prediction.max().clamp_min(1e-6)
    gray = (prediction.mul(255).byte().cpu().numpy())
    return Image.fromarray(gray, mode="L").convert("RGB")


def save_depth_input(encoded: str) -> tuple[str, Path, str]:
    source = decode_image(encoded)
    depth = make_depth_image(source)
    relative = f"krea-depth/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    depth.save(path, format="PNG")
    preview = io.BytesIO()
    depth.save(preview, format="PNG")
    return relative, path, base64.b64encode(preview.getvalue()).decode("ascii")


def save_nk2e_input(encoded: str, mode: str, preprocessed: bool = False) -> tuple[str, Path, str | None]:
    image = decode_image(encoded)
    preview_encoded: str | None = None
    if mode == "canny" and not preprocessed:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        image = Image.fromarray(edges, mode="L").convert("RGB")
        preview = io.BytesIO()
        image.save(preview, format="PNG")
        preview_encoded = base64.b64encode(preview.getvalue()).decode("ascii")
    relative = f"nk2e-{mode}/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    if mode == "canny" and preprocessed:
        preview = io.BytesIO()
        image.save(preview, format="PNG")
        preview_encoded = base64.b64encode(preview.getvalue()).decode("ascii")
    return relative, path, preview_encoded


def composite_strict_mask(encoded: str, source_path: Path, mask_path: Path, grow: int, feather: float) -> str:
    generated = decode_image(encoded)
    source = Image.open(source_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if generated.size != source.size or mask.size != source.size:
        raise HTTPException(status_code=400, detail="strict mask, source, and generated image dimensions must match")
    if grow:
        mask = mask.filter(ImageFilter.MaxFilter(grow * 2 + 1))
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    result = Image.composite(generated, source, mask)
    output = io.BytesIO()
    result.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def save_input(encoded: str, folder: str) -> tuple[str, Path]:
    image = decode_image(encoded)
    relative = f"{folder}/{uuid.uuid4().hex}.png"
    path = INPUT_ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return relative, path


async def comfy_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2) as client:
            response = await client.get(f"{COMFY_URL}/object_info")
        return response.is_success
    except httpx.HTTPError:
        return False


async def wait_for_output(client: httpx.AsyncClient, prompt_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 30 * 60
    while time.monotonic() < deadline:
        response = await client.get(f"{COMFY_URL}/history/{prompt_id}")
        response.raise_for_status()
        history = response.json().get(prompt_id)
        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {status.get('messages', [])}")
            for output in history.get("outputs", {}).values():
                images = output.get("images", [])
                if images:
                    return images[0]
        await asyncio.sleep(0.25)
    raise TimeoutError("image generation timed out")


async def execute_workflow(graph: dict[str, Any]) -> str:
    image: dict[str, Any] | None = None
    try:
        async with httpx.AsyncClient(timeout=30 * 60) as client:
            submitted = await client.post(f"{COMFY_URL}/prompt", json={"prompt": graph})
            submitted.raise_for_status()
            body = submitted.json()
            if body.get("node_errors"):
                raise RuntimeError(f"invalid workflow: {body['node_errors']}")
            image = await wait_for_output(client, body["prompt_id"])
            viewed = await client.get(f"{COMFY_URL}/view", params=image)
            viewed.raise_for_status()
            return base64.b64encode(viewed.content).decode("ascii")
    finally:
        if image is not None:
            candidate = (OUTPUT_ROOT / image["subfolder"] / image["filename"]).resolve()
            if candidate.is_relative_to(OUTPUT_ROOT):
                candidate.unlink(missing_ok=True)


@app.get("/health")
async def health() -> dict[str, str]:
    if not await comfy_ready():
        raise HTTPException(status_code=503, detail="NVFP4 runtime is starting")
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict[str, Any]:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.post("/v1/images/generations")
async def generate(request: ImageRequest) -> dict[str, Any]:
    if request.model not in MODEL_ALIASES:
        raise HTTPException(status_code=400, detail=f"model mismatch: {request.model}")
    if request.n != 1:
        raise HTTPException(status_code=400, detail="only n=1 is supported")
    if request.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only b64_json is supported")
    if request.sampler_name not in {None, "euler", "er_sde"}:
        raise HTTPException(status_code=400, detail="sampler_name must be euler or er_sde")
    if request.scheduler not in {None, "simple"}:
        raise HTTPException(status_code=400, detail="scheduler must be simple")
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    width, height = parse_size(request.size)
    if not 0 <= request.control_strength <= 2:
        raise HTTPException(status_code=400, detail="control_strength must be between 0 and 2")
    if not 0 <= request.identity_strength <= 2:
        raise HTTPException(status_code=400, detail="identity_strength must be between 0 and 2")
    if not 0 <= request.ref_boost <= 20:
        raise HTTPException(status_code=400, detail="ref_boost must be between 0 and 20")
    if not 384 <= request.grounding_px <= 1024:
        raise HTTPException(status_code=400, detail="grounding_px must be between 384 and 1024")
    if request.style not in {None, *STYLE_LORAS}:
        raise HTTPException(status_code=400, detail=f"unsupported style: {request.style}")
    if not 0 <= request.style_strength <= 2:
        raise HTTPException(status_code=400, detail="style_strength must be between 0 and 2")
    styles = list(request.styles)
    if request.style is not None and not any(style.name == request.style for style in styles):
        styles.append(StyleSelection(name=request.style, strength=request.style_strength))
    if len(styles) > len(STYLE_LORAS):
        raise HTTPException(status_code=400, detail="too many style LoRAs")
    if len({style.name for style in styles}) != len(styles):
        raise HTTPException(status_code=400, detail="duplicate style LoRAs are not supported")
    for style in styles:
        if style.name not in STYLE_LORAS:
            raise HTTPException(status_code=400, detail=f"unsupported style: {style.name}")
        if not 0 <= style.strength <= 2:
            raise HTTPException(status_code=400, detail="style strength must be between 0 and 2")
    user_loras = list(request.user_loras)
    if len(user_loras) > 5:
        raise HTTPException(status_code=400, detail="at most five user LoRAs may be stacked")
    if len({selection.filename for selection in user_loras}) != len(user_loras):
        raise HTTPException(status_code=400, detail="duplicate user LoRAs are not supported")
    for selection in user_loras:
        if selection.filename == "skc3vo.safetensors":
            raise HTTPException(status_code=400, detail="skc3vo is a filter mode; use filter_mode=adherence")
        filename = Path(selection.filename).name
        if filename != selection.filename or not filename.endswith(".safetensors"):
            raise HTTPException(status_code=400, detail="invalid user LoRA filename")
        if not (USER_LORA_ROOT / filename).is_file():
            raise HTTPException(status_code=400, detail=f"user LoRA not found: {filename}")
        if not 0 <= selection.strength <= 2:
            raise HTTPException(status_code=400, detail="user LoRA strength must be between 0 and 2")
    if len(request.vision_images) > 4:
        raise HTTPException(status_code=400, detail="at most four vision reference images are supported")
    if len(request.style_reference_images) > 2:
        raise HTTPException(status_code=400, detail="at most two style reference images are supported")
    if request.vision_mode not in {"descriptor", "instruct"}:
        raise HTTPException(status_code=400, detail="vision_mode must be descriptor or instruct")
    if not 0.1 <= request.vision_megapixels <= 4:
        raise HTTPException(status_code=400, detail="vision_megapixels must be between 0.1 and 4")
    if request.vision_images and request.source_image:
        raise HTTPException(status_code=400, detail="vision references cannot yet be combined with identity edit")
    if not 0 <= request.style_reference_strength <= 2:
        raise HTTPException(status_code=400, detail="style_reference_strength must be between 0 and 2")
    if request.style_reference_images and (
        request.vision_images or request.source_image or request.control_image or styles or user_loras
    ):
        raise HTTPException(
            status_code=400,
            detail="style references cannot yet be combined with vision, identity, depth, or style presets",
        )
    if request.nk2e_mode not in {"edit", "canny"}:
        raise HTTPException(status_code=400, detail="nk2e_mode must be edit or canny")
    if not 0 <= request.nk2e_strength <= 2:
        raise HTTPException(status_code=400, detail="nk2e_strength must be between 0 and 2")
    if (request.identity_mask or request.strict_mask) and not request.source_image:
        raise HTTPException(status_code=400, detail="identity masks require source_image")
    if not 0 <= request.strict_mask_grow <= 128 or not 0 <= request.strict_mask_feather <= 128:
        raise HTTPException(status_code=400, detail="strict mask grow and feather must be between 0 and 128")
    if request.vae_mode not in {"default", "wan", "real"}:
        raise HTTPException(status_code=400, detail="vae_mode must be default, wan, or real")
    if request.identity_fit_mode not in {"fit", "crop"}:
        raise HTTPException(status_code=400, detail="identity_fit_mode must be fit or crop")
    if request.filter_mode not in {"off", "adherence", "balanced", "strong"}:
        raise HTTPException(status_code=400, detail="filter_mode must be off, adherence, balanced, or strong")
    if request.filter_strength is not None and not 0 <= request.filter_strength <= 10:
        raise HTTPException(status_code=400, detail="filter_strength must be between 0 and 10")
    if not 0 <= request.prompt_enhancer_strength <= 2:
        raise HTTPException(status_code=400, detail="prompt_enhancer_strength must be between 0 and 2")
    if not 0.25 <= request.prompt_text_scale <= 4:
        raise HTTPException(status_code=400, detail="prompt_text_scale must be between 0.25 and 4")
    if not 0 <= request.detail_strength <= 2:
        raise HTTPException(status_code=400, detail="detail_strength must be between 0 and 2")
    if request.detail_vae not in {"wan", "qwen"}:
        raise HTTPException(status_code=400, detail="detail_vae must be wan or qwen")
    if request.nk2e_image and (
        request.style_reference_images
        or request.vision_images
        or request.source_image
        or request.control_image
        or styles
        or user_loras
    ):
        raise HTTPException(status_code=400, detail="NK2E experiments cannot be combined with other Krea modules")
    if request.anypaint_mask and not request.anypaint_image:
        raise HTTPException(status_code=400, detail="anypaint_mask requires anypaint_image")
    if request.anypaint_image and (
        request.style_reference_images
        or request.vision_images
        or request.source_image
        or request.reference_image
        or request.control_image
        or styles
        or user_loras
        or request.nk2e_image
    ):
        raise HTTPException(status_code=400, detail="AnyPaint cannot be combined with other Krea modules")
    if request.detail_enhance_image and (
        request.style_reference_images
        or request.vision_images
        or request.source_image
        or request.reference_image
        or request.control_image
        or styles
        or user_loras
        or request.nk2e_image
        or request.anypaint_image
    ):
        raise HTTPException(status_code=400, detail="detail enhancement cannot be combined with other Krea modules")
    pads = (
        request.outpaint_left,
        request.outpaint_top,
        request.outpaint_right,
        request.outpaint_bottom,
    )
    if any(value < 0 or value > 1536 or value % 16 for value in pads):
        raise HTTPException(status_code=400, detail="outpaint padding must be 0..1536 in multiples of 16")
    if not 0 <= request.anypaint_strength <= 2:
        raise HTTPException(status_code=400, detail="anypaint_strength must be between 0 and 2")
    if not 128 <= request.anypaint_reference_max_edge <= 768 or request.anypaint_reference_max_edge % 16:
        raise HTTPException(status_code=400, detail="anypaint_reference_max_edge must be 128..768 in multiples of 16")
    if not 0 <= request.anypaint_boundary_redraw_px <= 256:
        raise HTTPException(status_code=400, detail="anypaint_boundary_redraw_px must be between 0 and 256")
    steps = request.steps if request.steps is not None else (
        10 if request.source_image or request.detail_enhance_image else 8
    )
    if not 1 <= steps <= 20:
        raise HTTPException(status_code=400, detail="steps must be between 1 and 20")
    # Keep random seeds within JavaScript's exact integer range so the web client
    # can clone and reproduce them without rounding.
    seed = request.seed if request.seed is not None and request.seed >= 0 else secrets.randbits(53)
    for style in styles:
        trigger = STYLE_TRIGGERS[style.name]
        if trigger.lower() not in prompt.lower():
            prompt = f"{prompt}. {trigger}"
    prefix = f"nvfp4-api/{uuid.uuid4().hex}"
    depth_path: Path | None = None
    depth_preview: str | None = None
    source_path: Path | None = None
    reference_path: Path | None = None
    identity_mask_path: Path | None = None
    strict_mask_path: Path | None = None
    vision_paths: list[Path] = []
    style_reference_paths: list[Path] = []
    nk2e_path: Path | None = None
    nk2e_preview: str | None = None
    anypaint_path: Path | None = None
    anypaint_mask_path: Path | None = None
    detail_path: Path | None = None

    async with generation_lock:
        try:
            depth_name: str | None = None
            if request.control_image:
                depth_name, depth_path, depth_preview = await asyncio.to_thread(
                    save_depth_input, request.control_image
                )
            vision_names: list[str] = []
            for encoded_image in request.vision_images:
                vision_name, vision_path = await asyncio.to_thread(
                    save_input, encoded_image, "krea-vision"
                )
                vision_names.append(vision_name)
                vision_paths.append(vision_path)
            style_reference_names: list[str] = []
            for encoded_image in request.style_reference_images:
                image_name, image_path = await asyncio.to_thread(
                    save_input, encoded_image, "krea-style-reference"
                )
                style_reference_names.append(image_name)
                style_reference_paths.append(image_path)
            nk2e_name: str | None = None
            if request.nk2e_image:
                nk2e_name, nk2e_path, nk2e_preview = await asyncio.to_thread(
                    save_nk2e_input, request.nk2e_image, request.nk2e_mode, request.nk2e_preprocessed
                )
            anypaint_name: str | None = None
            anypaint_mask_name: str | None = None
            if request.anypaint_image:
                anypaint_name, anypaint_path = await asyncio.to_thread(
                    save_input, request.anypaint_image, "krea-anypaint"
                )
                source_width, source_height = Image.open(anypaint_path).size
                output_width = source_width + request.outpaint_left + request.outpaint_right
                output_height = source_height + request.outpaint_top + request.outpaint_bottom
                if not (512 <= output_width <= 2048 and 512 <= output_height <= 2048):
                    raise HTTPException(status_code=400, detail="AnyPaint output dimensions must be between 512 and 2048")
                if output_width % 16 or output_height % 16:
                    raise HTTPException(status_code=400, detail="AnyPaint source plus padding must be multiples of 16")
                if request.anypaint_mask:
                    anypaint_mask_name, anypaint_mask_path = await asyncio.to_thread(
                        save_input, request.anypaint_mask, "krea-anypaint-mask"
                    )
                    if Image.open(anypaint_mask_path).size != (source_width, source_height):
                        raise HTTPException(status_code=400, detail="AnyPaint mask dimensions must match the source image")
            detail_name: str | None = None
            if request.detail_enhance_image:
                detail_name, detail_path = await asyncio.to_thread(
                    save_input, request.detail_enhance_image, "krea-detail-enhance"
                )
                width, height = Image.open(detail_path).size
                if not (512 <= width <= 2048 and 512 <= height <= 2048):
                    raise HTTPException(status_code=400, detail="detail image dimensions must be between 512 and 2048")
                if width % 16 or height % 16:
                    raise HTTPException(status_code=400, detail="detail image dimensions must be multiples of 16")
            if detail_name is not None:
                graph = detail_enhance_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    detail_name,
                    request.detail_strength,
                    steps,
                    WAN_VAE if request.detail_vae == "wan" else VAE,
                )
            elif anypaint_name is not None:
                graph = anypaint_workflow(
                    prompt,
                    seed,
                    prefix,
                    anypaint_name,
                    anypaint_mask_name,
                    request.outpaint_left,
                    request.outpaint_top,
                    request.outpaint_right,
                    request.outpaint_bottom,
                    request.anypaint_strength,
                    request.anypaint_reference_max_edge,
                    request.anypaint_boundary_redraw_px,
                    request.anypaint_vlm_reference,
                    steps,
                )
            elif nk2e_name is not None:
                graph = nk2e_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    nk2e_name,
                    request.nk2e_mode,
                    request.nk2e_strength,
                    steps,
                )
            elif style_reference_names:
                graph = style_reference_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    style_reference_names,
                    request.style_reference_strength,
                    steps,
                )
            elif request.source_image:
                source_name, source_path = await asyncio.to_thread(
                    save_input, request.source_image, "krea-edit"
                )
                identity_mask_name: str | None = None
                if request.identity_mask:
                    identity_mask_name, identity_mask_path = await asyncio.to_thread(
                        save_input, request.identity_mask, "krea-edit-mask"
                    )
                    if Image.open(identity_mask_path).size != Image.open(source_path).size:
                        raise HTTPException(status_code=400, detail="identity mask dimensions must match source image")
                if request.strict_mask:
                    _, strict_mask_path = await asyncio.to_thread(
                        save_input, request.strict_mask, "krea-strict-mask"
                    )
                    if Image.open(strict_mask_path).size != Image.open(source_path).size:
                        raise HTTPException(status_code=400, detail="strict mask dimensions must match source image")
                reference_name: str | None = None
                if request.reference_image:
                    reference_name, reference_path = await asyncio.to_thread(
                        save_input, request.reference_image, "krea-edit"
                    )
                graph = identity_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    source_name,
                    reference_name,
                    identity_mask_name,
                    request.identity_strength,
                    request.ref_boost,
                    request.grounding_px,
                    steps,
                    styles,
                    user_loras,
                    depth_name,
                    request.control_strength,
                    request.identity_fit_mode,
                )
            elif depth_name is not None:
                graph = depth_workflow(
                    prompt,
                    width,
                    height,
                    seed,
                    prefix,
                    depth_name,
                    request.control_strength,
                    steps,
                    styles,
                    user_loras,
                )
                if vision_names:
                    graph = apply_vision_conditioning(
                        graph,
                        prompt,
                        vision_names,
                        request.vision_mode,
                        request.vision_megapixels,
                    )
            else:
                graph = workflow(prompt, width, height, seed, prefix, steps, styles, user_loras)
                if vision_names:
                    graph = apply_vision_conditioning(
                        graph,
                        prompt,
                        vision_names,
                        request.vision_mode,
                        request.vision_megapixels,
                    )
            graph = apply_filter_bypass(graph, request.filter_mode, request.filter_strength)
            graph = apply_prompt_enhancer(
                graph,
                request.prompt_enhancer,
                request.prompt_enhancer_strength,
                request.prompt_text_scale,
            )
            sampler_name = request.sampler_name or ("er_sde" if detail_name is not None else "euler")
            scheduler = request.scheduler or "simple"
            graph = apply_sampling(graph, sampler_name, scheduler)
            if "3" in graph and detail_name is None:
                if request.vae_mode == "real":
                    graph["3"]["inputs"]["vae_name"] = REAL_VAE
                elif request.vae_mode == "wan":
                    graph["3"]["inputs"]["vae_name"] = WAN_VAE
            encoded = await execute_workflow(graph)
            if strict_mask_path is not None and source_path is not None:
                encoded = await asyncio.to_thread(
                    composite_strict_mask,
                    encoded,
                    source_path,
                    strict_mask_path,
                    request.strict_mask_grow,
                    request.strict_mask_feather,
                )
        except (httpx.HTTPError, KeyError, RuntimeError, TimeoutError) as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            if depth_path is not None:
                depth_path.unlink(missing_ok=True)
            if source_path is not None:
                source_path.unlink(missing_ok=True)
            if reference_path is not None:
                reference_path.unlink(missing_ok=True)
            if identity_mask_path is not None:
                identity_mask_path.unlink(missing_ok=True)
            if strict_mask_path is not None:
                strict_mask_path.unlink(missing_ok=True)
            for vision_path in vision_paths:
                vision_path.unlink(missing_ok=True)
            for style_reference_path in style_reference_paths:
                style_reference_path.unlink(missing_ok=True)
            if nk2e_path is not None:
                nk2e_path.unlink(missing_ok=True)
            if anypaint_path is not None:
                anypaint_path.unlink(missing_ok=True)
            if anypaint_mask_path is not None:
                anypaint_mask_path.unlink(missing_ok=True)
            if detail_path is not None:
                detail_path.unlink(missing_ok=True)

    response = {"created": int(time.time()), "seed": seed, "data": [{"b64_json": encoded}]}
    if depth_preview is not None:
        response["control_b64_json"] = depth_preview
    if nk2e_preview is not None:
        response["control_b64_json"] = nk2e_preview
    return response
