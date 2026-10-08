#!/usr/bin/env python3
"""Run a small, reproducible Krea 2 ReID experiment through ComfyUI's API."""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from pathlib import Path

import requests


def memory_used_gib() -> float:
    values: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
    return (values["MemTotal"] - values["MemAvailable"]) / 1024 / 1024


def upload_image(base_url: str, path: Path) -> str:
    with path.open("rb") as handle:
        response = requests.post(
            f"{base_url}/upload/image",
            files={"image": (path.name, handle, "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    return "/".join(part for part in (payload.get("subfolder"), payload["name"]) if part)


def reid_graph(image_name: str, prompt: str, seed: int, prefix: str) -> dict:
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "krea2_turbo_int8_convrot.safetensors", "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen3vl_4b_bf16.safetensors", "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {
            "class_type": "ImageScaleToTotalPixels",
            "inputs": {"image": ["4", 0], "upscale_method": "area", "megapixels": 0.140625, "resolution_steps": 16},
        },
        "6": {
            "class_type": "TextEncodeKrea2OstrisEdit",
            "inputs": {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0], "image1": ["5", 0]},
        },
        "7": {
            "class_type": "TextEncodeKrea2OstrisEdit",
            "inputs": {"clip": ["2", 0], "prompt": "", "vae": ["3", 0], "image1": ["5", 0]},
        },
        "8": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "9": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {"conditioning": ["7", 0], "reference_latents_method": "index_timestep_zero"},
        },
        "10": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": "krea2_reid_rank32.safetensors", "strength_model": 1.0},
        },
        "11": {
            "class_type": "Krea2OstrisEditModelPatch",
            "inputs": {"model": ["10", 0], "kv_cache": True},
        },
        "12": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "13": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["11", 0], "positive": ["8", 0], "negative": ["9", 0],
                "latent_image": ["12", 0], "seed": seed, "steps": 8, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            },
        },
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["3", 0]}},
        "15": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["14", 0]}},
    }


def quadview_graph(image_name: str, seed: int, prefix: str) -> dict:
    prompt = (
        "Convert the character in the image to a character sheet showing a face close-up, "
        "front full body, side full body and back full body views"
    )
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "krea2_turbo_int8_convrot.safetensors", "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"},
        },
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}},
        "6": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": ["1", 0], "lora_name": "QuadView_krea2_v1.safetensors", "strength_model": 1.0},
        },
        "7": {
            "class_type": "Krea2EditModelPatch",
            "inputs": {
                "model": ["6", 0], "source_latent": ["5", 0], "ref_boost": 1.0,
                "ref_boost_a": 1.0, "fit_mode": "fit", "vae": ["3", 0], "source_image": ["4", 0],
            },
        },
        "8": {
            "class_type": "Krea2EditGroundedEncode",
            "inputs": {"clip": ["2", 0], "prompt": prompt, "grounding_px": 0, "system_prompt": "", "image": ["4", 0]},
        },
        "9": {
            "class_type": "Krea2EditGroundedEncode",
            "inputs": {"clip": ["2", 0], "prompt": "", "grounding_px": 768, "system_prompt": "", "image": ["4", 0]},
        },
        "10": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1536, "height": 1024, "batch_size": 1}},
        "11": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["7", 0], "positive": ["8", 0], "negative": ["9", 0],
                "latent_image": ["10", 0], "seed": seed, "steps": 10, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
            },
        },
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["12", 0]}},
    }


def run(base_url: str, graph: dict, output_dir: Path) -> dict:
    client_id = str(uuid.uuid4())
    response = requests.post(f"{base_url}/prompt", json={"prompt": graph, "client_id": client_id}, timeout=120)
    response.raise_for_status()
    prompt_id = response.json()["prompt_id"]

    stop = threading.Event()
    samples: list[float] = []

    def sample_memory() -> None:
        while not stop.wait(0.25):
            samples.append(memory_used_gib())

    started = time.monotonic()
    thread = threading.Thread(target=sample_memory, daemon=True)
    thread.start()
    try:
        while True:
            history_response = requests.get(f"{base_url}/history/{prompt_id}", timeout=30)
            history_response.raise_for_status()
            history = history_response.json().get(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(json.dumps(status, ensure_ascii=False))
                outputs = history.get("outputs", {})
                images = [item for value in outputs.values() for item in value.get("images", [])]
                if images:
                    break
            time.sleep(1)
    finally:
        stop.set()
        thread.join(timeout=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for item in images:
        view = requests.get(f"{base_url}/view", params=item, timeout=120)
        view.raise_for_status()
        destination = output_dir / Path(item["filename"]).name
        destination.write_bytes(view.content)
        saved.append(str(destination.resolve()))
    return {
        "prompt_id": prompt_id,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "memory_start_gib": round(samples[0], 2) if samples else None,
        "memory_peak_gib": round(max(samples), 2) if samples else None,
        "outputs": saved,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mode", choices=("reid", "quadview"), default="reid")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--seed", type=int, default=4242424242)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/krea-reid"))
    args = parser.parse_args()

    if args.mode == "reid" and not args.prompt.strip():
        parser.error("--prompt is required in reid mode")

    image_name = upload_image(args.base_url, args.image)
    prefix = f"character_{args.mode}/{int(time.time())}"
    graph = (
        reid_graph(image_name, args.prompt, args.seed, prefix)
        if args.mode == "reid"
        else quadview_graph(image_name, args.seed, prefix)
    )
    result = run(args.base_url, graph, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
