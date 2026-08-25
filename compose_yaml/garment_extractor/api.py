import base64
import io
import os
import threading
from typing import Annotated

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageFilter
from scipy import ndimage
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


MODEL_ID = os.getenv("GARMENT_MODEL", "fashn-ai/fashn-human-parser")
MODEL_REVISION = os.getenv("GARMENT_MODEL_REVISION", "1f80c34dbab321c5730dda5c3fea279fd3e97498")
MAX_IMAGES = 5
MAX_IMAGE_BYTES = 32 << 20

# FASHN Human Parser labels: background, face, hair, top, dress, skirt,
# pants, belt, bag, hat, scarf, glasses, arms, hands, legs, feet, torso,
# jewelry.  Targets intentionally exclude body pixels.
TARGETS = {
    "all": (3, 4, 5, 6, 7, 9, 10, 15, 17),
    "upper": (3,),
    "lower": (5, 6, 7),
    "dress": (4,),
    "outer": (3,),
    "shoes": (15,),
    "accessories": (7, 9, 10, 17),
}

app = FastAPI(title="Spark Media Garment Extractor", version="1.0.0")
model = None
processor = None
model_lock = threading.Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    global model, processor
    if model is not None:
        return
    with model_lock:
        if model is not None:
            return
        processor = SegformerImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        model.eval().to(device)


def png_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def sharpness_score(rgb: np.ndarray, mask: np.ndarray) -> float:
    gray = rgb.astype(np.float32).mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    pixels = mask > 127
    if not pixels.any():
        return 0.0
    return float(np.mean((gx + gy)[pixels]) / 255.0)


def segment(image: Image.Image, class_ids: tuple[int, ...], feather: float):
    rgb_image = image.convert("RGB")
    inputs = processor(images=rgb_image, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits
        logits = F.interpolate(logits, size=(rgb_image.height, rgb_image.width), mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)[0]
        selected = torch.zeros_like(prediction, dtype=torch.bool)
        for class_id in class_ids:
            selected |= prediction == class_id
    prediction_np = prediction.cpu().numpy()
    selected_np = selected.cpu().numpy()

    # Semantic parsing does not provide person instances. Group all non-background
    # body parts into people and keep the dominant, near-centre subject so a
    # background crowd does not leak into one garment cutout.
    human = prediction_np != 0
    joined = ndimage.binary_closing(human, structure=np.ones((7, 7), dtype=bool), iterations=2)
    labels, count = ndimage.label(joined)
    if count:
        height, width = selected_np.shape
        best_label = 0
        best_score = -1.0
        for label_id in range(1, count + 1):
            component = labels == label_id
            area = int(component.sum())
            if area < 64:
                continue
            ys, xs = np.nonzero(component)
            centre_distance = ((float(xs.mean()) / max(width - 1, 1) - 0.5) ** 2 + (float(ys.mean()) / max(height - 1, 1) - 0.5) ** 2) ** 0.5
            score = area * max(0.35, 1.0 - centre_distance)
            if score > best_score:
                best_score, best_label = score, label_id
        if best_label:
            person = ndimage.binary_dilation(labels == best_label, iterations=5)
            selected_np &= person

    hard = selected_np.astype(np.uint8) * 255
    if not np.any(hard):
        raise ValueError("선택한 의상 영역을 찾지 못했습니다")

    # Close one-pixel holes while preserving the semantic boundary. A small
    # alpha feather removes blocky decoder edges without inventing garment area.
    mask_image = Image.fromarray(hard, mode="L").filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    if feather > 0:
        mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=feather))
    mask = np.asarray(mask_image)
    source = np.asarray(rgb_image)
    rgba = np.dstack((source, mask)).astype(np.uint8)

    binary = hard > 0
    coverage = float(binary.mean())
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    border_penalty = float(border.mean())
    sharpness = sharpness_score(source, hard)
    # Prefer a large, complete and clear view. Border contact usually means the
    # garment is cropped; sharpness is only a light tie-breaker.
    score = coverage * (1.0 - 0.65 * border_penalty) + min(sharpness, 0.25) * 0.08
    return Image.fromarray(rgba, mode="RGBA"), mask_image, coverage, border_penalty, score


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": model is not None, "device": device}


@app.post("/v1/garments/extract")
async def extract_garment(
    images: Annotated[list[UploadFile], File()],
    target: Annotated[str, Form()] = "all",
    feather: Annotated[float, Form()] = 1.0,
):
    target_keys = list(dict.fromkeys(item.strip() for item in target.split(",") if item.strip()))
    if not target_keys:
        target_keys = ["all"]
    if any(item not in TARGETS for item in target_keys):
        raise HTTPException(400, "unsupported garment target")
    if "all" in target_keys:
        target_keys = ["all"]
    class_ids = tuple(sorted({class_id for item in target_keys for class_id in TARGETS[item]}))
    if not images or len(images) > MAX_IMAGES:
        raise HTTPException(400, f"1..{MAX_IMAGES} images are required")
    if feather < 0 or feather > 8:
        raise HTTPException(400, "feather must be between 0 and 8")
    try:
        load_model()
    except Exception as error:
        raise HTTPException(503, f"model loading failed: {error}") from error

    candidates = []
    failures = []
    for index, upload in enumerate(images):
        data = await upload.read(MAX_IMAGE_BYTES + 1)
        if not data or len(data) > MAX_IMAGE_BYTES:
            failures.append({"index": index, "error": "image must be 1 byte..32 MiB"})
            continue
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            cutout, mask, coverage, border_penalty, score = segment(image, class_ids, feather)
            candidates.append({
                "index": index,
                "cutout": cutout,
                "mask": mask,
                "coverage": coverage,
                "border_penalty": border_penalty,
                "score": score,
                "width": cutout.width,
                "height": cutout.height,
            })
        except Exception as error:
            failures.append({"index": index, "error": str(error)})
    if not candidates:
        raise HTTPException(422, {"message": "선택한 의상을 찾지 못했습니다", "failures": failures})

    chosen = max(candidates, key=lambda item: item["score"])
    return {
        "model": MODEL_ID,
        "target": ",".join(target_keys),
        "selected_index": chosen["index"],
        "width": chosen["width"],
        "height": chosen["height"],
        "coverage": chosen["coverage"],
        "cutout_b64": png_base64(chosen["cutout"]),
        "mask_b64": png_base64(chosen["mask"]),
        "candidates": [
            {
                "index": item["index"],
                "coverage": item["coverage"],
                "border_penalty": item["border_penalty"],
                "score": item["score"],
            }
            for item in candidates
        ],
        "failures": failures,
    }
