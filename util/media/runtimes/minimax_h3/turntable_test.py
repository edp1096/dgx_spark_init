#!/usr/bin/env python3
"""Generate a real multi-frame H3 turntable video from one character image."""

from __future__ import annotations

import json
import os
import time
import urllib.request


COMFY_URL = "http://127.0.0.1:8190"
SEED = int(os.environ.get("H3_SEED", "61"))
WIDTH = int(os.environ.get("H3_WIDTH", "768"))
HEIGHT = int(os.environ.get("H3_HEIGHT", "1344"))
LENGTH = int(os.environ.get("H3_LENGTH", "124"))
STEPS = int(os.environ.get("H3_STEPS", "4"))
REFERENCE = os.environ.get("H3_REFERENCE", "dynamic-h3-front-square.png")
OUTPUT_PREFIX = os.environ.get("H3_OUTPUT_PREFIX", "h3-turntable/character-360")

PROMPT = """<Picture 1> is the only character identity and wardrobe reference.
Create one continuous five-second photorealistic studio turntable shot of exactly the same adult Korean woman. She remains perfectly still in a neutral relaxed A-pose at the exact center of a plain warm light-grey studio. Preserve her exact face, facial proportions, age, skin tone, body proportions, shoulder-length side-parted dark hair, short-sleeved yellow V-neck button-front belted midi dress, and white lace-up sneakers throughout the entire shot.

The camera performs one smooth constant-speed clockwise 360-degree orbit around her at eye level, beginning at a complete front view, passing through front-right three-quarter, strict right profile, back-right three-quarter, complete back view, back-left three-quarter, strict left profile, front-left three-quarter, and ending at the exact front view. Keep the camera distance, focal length, framing, horizon, exposure, lighting, background, character scale, and ground position completely locked. Keep her complete head and feet visible at all times.

Single unbroken shot. No cuts, no zoom, no camera height change, no subject rotation relative to the studio, no body motion, no walking, no speaking, no expression change, no hair movement, no cloth movement, no extra person, no extra limb, no anatomy change, no wardrobe change, no identity drift, no text, no labels, no watermark, and no sound."""


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
    return {
        "1": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            "weight_dtype": "default",
        }},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["1", 0],
            "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
            "strength_model": 1.0,
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
            "vae_name": "minimax_h3_video_vae_fp16.safetensors",
        }},
        "6": {"class_type": "VAELoader", "inputs": {
            "vae_name": "minimax_h3_audio_vae_fp32.safetensors",
        }},
        "7": {"class_type": "LoadImage", "inputs": {"image": REFERENCE}},
        "8": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["4", 0],
            "vae": ["5", 0],
            "audio_vae": ["6", 0],
            "prompt": PROMPT,
            "width": WIDTH,
            "height": HEIGHT,
            "length": LENGTH,
            "ref_image_size": "max",
            "ref_images.ref_image_0": ["7", 0],
        }},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "11": {"class_type": "BasicScheduler", "inputs": {
            "model": ["3", 0],
            "scheduler": "simple",
            "steps": STEPS,
            "denoise": 1.0,
        }},
        "12": {"class_type": "BasicGuider", "inputs": {
            "model": ["3", 0],
            "conditioning": ["8", 0],
        }},
        "13": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0],
            "guider": ["12", 0],
            "sampler": ["10", 0],
            "sigmas": ["11", 0],
            "latent_image": ["8", 1],
        }},
        "14": {"class_type": "VAEDecode", "inputs": {
            "samples": ["13", 0],
            "vae": ["5", 0],
        }},
        "15": {"class_type": "CreateVideo", "inputs": {
            "images": ["14", 0],
            "fps": 24.0,
            "bit_depth": 8,
        }},
        "16": {"class_type": "SaveVideo", "inputs": {
            "video": ["15", 0],
            "filename_prefix": OUTPUT_PREFIX,
            "format": "mp4",
            "codec": "h264",
        }},
    }


def main() -> None:
    started = time.monotonic()
    response = request_json("/prompt", {"prompt": graph(), "client_id": "h3-turntable-test"})
    prompt_id = response["prompt_id"]
    print(f"prompt_id={prompt_id}", flush=True)
    while True:
        history = request_json(f"/history/{prompt_id}")
        item = history.get(prompt_id)
        if item is not None:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, ensure_ascii=False))
            outputs = item.get("outputs", {}).get("16", {})
            if outputs:
                print(json.dumps({
                    "elapsed_seconds": time.monotonic() - started,
                    "output": outputs,
                }, ensure_ascii=False), flush=True)
                return
        time.sleep(2)


if __name__ == "__main__":
    main()
