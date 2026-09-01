#!/usr/bin/env python3
"""Submit the verified stage-one H3 character-sheet graph to ComfyUI."""

from __future__ import annotations

import json
import os
import time
import urllib.request


COMFY_URL = "http://127.0.0.1:8190"
NOISE_SEED = int(os.environ.get("H3_SEED", "1"))
WIDTH = int(os.environ.get("H3_WIDTH", "2816"))
HEIGHT = int(os.environ.get("H3_HEIGHT", "2816"))
STEPS = int(os.environ.get("H3_STEPS", "8"))
FACE_IMAGE = os.environ.get("H3_FACE_IMAGE", "h3-face.png")
OUTFIT_IMAGE = os.environ.get("H3_OUTFIT_IMAGE", "h3-outfit.png")
SINGLE_REFERENCE = os.environ.get("H3_SINGLE_REFERENCE", "").lower() in {
    "1", "true", "yes", "on",
}
OUTPUT_PREFIX = os.environ.get(
    "H3_OUTPUT_PREFIX", "h3-smoke/stage1-character-sheet-exact"
)

PROMPT = """Create one static high-resolution character turnaround sheet of the same adult Korean woman.

Use Picture 1 for facial identity, age, skin tone, and natural facial proportions.
Use Picture 2 for the costume design, colors, materials, footwear, and wearable accessories. Apply that costume to the person from Picture 1. Keep the face uncovered; do not wear a helmet or mask.

Arrange exactly three equal vertical photographs from left to right on a neutral light-grey studio background:
Photo 1: complete full-body front view in a relaxed A-pose.
Photo 2: complete full-body strict 90-degree right-side view.
Photo 3: complete full-body back view.

Keep the same identity, body proportions, costume, scale, ground line, lighting, and realistic photographic finish in all three photos. Show the complete head and feet. No extra panel, no action pose, no separate product display, and no text."""

PROMPT_FILE = os.environ.get("H3_PROMPT_FILE", "")
if PROMPT_FILE:
    with open(PROMPT_FILE, "r", encoding="utf-8") as prompt_file:
        PROMPT = prompt_file.read().strip()


def request_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        COMFY_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def graph() -> dict:
    result = {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default",
        }},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0],
            "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            "strength_model": 0.75,
        }},
        "3": {"class_type": "ModelAttentionBackend", "inputs": {
            "model": ["2", 0],
            "attention": "comfy kitchen attention",
        }},
        "4": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            "type": "minimax",
            "device": "default",
        }},
        "5": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_t1_image_vae_step1597.safetensors",
        }},
        "6": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors",
        }},
        "7": {"class_type": "LoadImage", "inputs": {"image": FACE_IMAGE}},
        "8": {"class_type": "LoadImage", "inputs": {"image": OUTFIT_IMAGE}},
        "9": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0],
            "vae": ["5", 0],
            "audio_vae": ["6", 0],
            "prompt": PROMPT,
            "width": WIDTH,
            "height": HEIGHT,
            "length": 5,
            "ref_image_size": "max",
            "ref_images.ref_image_0": ["7", 0],
        }},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": NOISE_SEED}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "12": {"class_type": "BasicScheduler", "inputs": {
            "model": ["3", 0],
            "scheduler": "sgm_uniform",
            "steps": STEPS,
            "denoise": 1.0,
        }},
        "13": {"class_type": "BasicGuider", "inputs": {
            "model": ["3", 0],
            "conditioning": ["9", 0],
        }},
        "14": {"class_type": "ToobusyMiniMaxH3ImageLatent", "inputs": {
            "width": WIDTH,
            "height": HEIGHT,
        }},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["10", 0],
            "guider": ["13", 0],
            "sampler": ["11", 0],
            "sigmas": ["12", 0],
            "latent_image": ["14", 0],
        }},
        "16": {"class_type": "VAEDecode", "inputs": {
            "samples": ["15", 0],
            "vae": ["5", 0],
        }},
        "17": {"class_type": "SaveImage", "inputs": {
            "images": ["16", 0],
            "filename_prefix": OUTPUT_PREFIX,
        }},
    }
    if not SINGLE_REFERENCE:
        result["9"]["inputs"]["ref_images.ref_image_1"] = ["8", 0]
    return result


def main() -> None:
    started = time.monotonic()
    response = request_json("/prompt", {"prompt": graph(), "client_id": "h3-smoke-test"})
    prompt_id = response["prompt_id"]
    print(f"prompt_id={prompt_id}", flush=True)
    while True:
        history = request_json(f"/history/{prompt_id}")
        item = history.get(prompt_id)
        if item is not None:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
            outputs = item.get("outputs", {}).get("17", {}).get("images", [])
            if outputs:
                elapsed = time.monotonic() - started
                print(json.dumps({"elapsed_seconds": elapsed, "images": outputs}, ensure_ascii=False))
                return
        time.sleep(2)


if __name__ == "__main__":
    main()
