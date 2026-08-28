#!/usr/bin/env python3
"""Validate Krea ReID across visibly distinct human identities and text-only controls."""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import requests


IDENTITIES = {
    "elderly_man": (
        "A realistic waist-up studio portrait of a 74-year-old Korean man with a long narrow face, "
        "deep forehead and eye wrinkles, thick straight silver eyebrows, dark brown eyes, a broad slightly bulbous nose, "
        "thin lips, a small dark mole high on his left cheek, short swept-back silver hair with a receding hairline, "
        "and a lean build. He wears a plain charcoal henley shirt. Neutral gray background, soft even daylight, "
        "front-facing, calm neutral expression, highly detailed natural skin."
    ),
    "young_man": (
        "A realistic waist-up studio portrait of a 24-year-old Black man with deep brown skin, a smooth oval face, "
        "high cheekbones, almond-shaped dark eyes, a straight medium-width nose, full lips, a clean jawline, "
        "a very short precise low fade haircut and no facial hair. He has an athletic slim build and wears a plain cobalt blue crew-neck shirt. "
        "Neutral gray background, soft even daylight, front-facing, calm neutral expression, highly detailed natural skin."
    ),
    "rugged_man": (
        "A realistic waist-up studio portrait of a rugged 46-year-old white man with weathered tan skin, a wide angular face, "
        "a visibly crooked previously broken nose, a short pale scar cutting through his right eyebrow, steel gray eyes, "
        "rough dark brown hair with gray at the temples, and a dense uneven salt-and-pepper stubble beard. "
        "He has a broad muscular build and wears a faded olive work shirt. Neutral gray background, soft even daylight, "
        "front-facing, stern neutral expression, highly detailed pores and scars."
    ),
    "western_woman": (
        "A realistic waist-up studio portrait of a 32-year-old Irish woman with very fair skin, a heart-shaped face, "
        "dense freckles across her cheeks and nose, large green eyes, a narrow upturned nose, softly defined coral lips, "
        "and shoulder-length naturally curly copper-red hair parted on the left. She has a slender build and wears a plain forest-green blouse, "
        "with no jewelry. Neutral gray background, soft even daylight, front-facing, calm neutral expression, highly detailed natural skin."
    ),
    "east_asian_woman": (
        "A realistic waist-up studio portrait of a 29-year-old Korean woman with light warm-beige skin, a compact oval face, "
        "straight softly arched eyebrows, narrow dark brown monolid eyes, a small straight nose, muted rose lips, "
        "and a distinct tiny beauty mark below the outer corner of her right eye. Her glossy black hair is cut into a blunt chin-length bob with straight bangs. "
        "She has a petite build and wears a plain burgundy mock-neck top, with no jewelry. Neutral gray background, soft even daylight, "
        "front-facing, calm neutral expression, highly detailed natural skin."
    ),
}

SCENES = {
    "market": "The person selects vegetables at a busy outdoor market, candid waist-up photograph, soft morning sunlight.",
    "seashore": "The person stands beside a windy rocky seashore under an overcast sky, cinematic medium shot.",
}


def generate(endpoint: str, prompt: str, seed: int, reference: bytes | None = None) -> tuple[bytes, float]:
    payload = {
        "model": "krea2-turbo-nvfp4", "prompt": prompt, "checkpoint": "official",
        "size": "768x768", "steps": 8, "seed": seed, "response_format": "b64_json",
        "output_format": "png", "filter_mode": "balanced", "filter_strength": 1,
        "prompt_enhancer": False, "sampler_name": "euler", "scheduler": "simple",
    }
    if reference is not None:
        payload["reid_image"] = base64.b64encode(reference).decode("ascii")
    started = time.monotonic()
    response = requests.post(f"{endpoint.rstrip('/')}/v1/images/generations", json=payload, timeout=1800)
    response.raise_for_status()
    data = base64.b64decode(response.json()["data"][0]["b64_json"])
    return data, round(time.monotonic() - started, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8691")
    parser.add_argument("--output", type=Path, default=Path("data/experiments/krea-character-validation/diversity"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"identities": {}, "scenes": SCENES}

    references: dict[str, bytes] = {}
    for index, (name, prompt) in enumerate(IDENTITIES.items()):
        data, elapsed = generate(args.endpoint, prompt, 810000 + index)
        references[name] = data
        (args.output / f"{name}_reference.png").write_bytes(data)
        report["identities"][name] = {"reference_prompt": prompt, "reference_seconds": elapsed, "outputs": {}}
        print(f"reference {name}: {elapsed}s", flush=True)

    for identity_index, name in enumerate(IDENTITIES):
        for scene_index, (scene, prompt) in enumerate(SCENES.items()):
            seed = 820000 + identity_index * 10 + scene_index
            data, elapsed = generate(args.endpoint, prompt, seed, references[name])
            path = args.output / f"{name}_{scene}_reid.png"
            path.write_bytes(data)
            report["identities"][name]["outputs"][f"{scene}_reid"] = {"path": str(path), "seed": seed, "seconds": elapsed}
            print(f"reid {name}/{scene}: {elapsed}s", flush=True)

    for identity_index, name in enumerate(IDENTITIES):
        seed = 820000 + identity_index * 10
        data, elapsed = generate(args.endpoint, SCENES["market"], seed)
        path = args.output / f"{name}_market_control.png"
        path.write_bytes(data)
        report["identities"][name]["outputs"]["market_control"] = {"path": str(path), "seed": seed, "seconds": elapsed}
        print(f"control {name}/market: {elapsed}s", flush=True)

    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
