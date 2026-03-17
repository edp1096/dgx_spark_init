"""Auto-captioning for LoRA training datasets using Qwen3.5-2B (native multimodal).

Reuses the same Qwen3.5-2B model already cached for translation.
Loads with Qwen3_5ForConditionalGeneration + Qwen3VLProcessor for image input.
"""

import gc
import logging
from pathlib import Path

import torch
from PIL import Image

from zit_config import TRANSLATOR_DIR, TRANSLATOR_REPO, MODEL_DIR

logger = logging.getLogger("zit-ui")

_model = None
_processor = None

# ---------------------------------------------------------------------------
# Captioning prompt — detailed enough for LoRA training quality
# ---------------------------------------------------------------------------
CAPTION_SYSTEM = """\
You are an expert image captioning assistant for AI image generation training datasets.
Your job is to produce a single, highly detailed, comma-separated caption describing everything visible in the photograph."""

CAPTION_USER = """\
Describe this image in exhaustive detail for AI image generation model training.

Include ALL of the following in order:
1. **Subject**: gender, estimated age range, ethnicity if apparent, facial features (face shape, eye shape/color, eyebrows, nose, lips, skin tone, freckles/moles), facial expression, gaze direction
2. **Hair**: color, length, texture (straight/wavy/curly), style (up/down/braided/bangs), parting
3. **Clothing & accessories**: garment type, color, pattern, fabric texture, neckline, sleeves, jewelry, glasses, hats
4. **Pose & body**: head tilt, shoulder orientation, hand position, body posture, cropping (headshot/bust/half-body/full-body)
5. **Background & environment**: setting (indoor/outdoor/studio), background elements, depth of field (blurred/sharp), colors
6. **Lighting**: direction (front/side/back/rim), quality (soft/hard/diffused), color temperature (warm/cool/neutral), shadows, highlights, catchlights in eyes
7. **Photography**: apparent camera angle (eye-level/low/high), lens type (wide/normal/telephoto), bokeh quality, overall mood/atmosphere
8. **Image quality**: resolution feel (sharp/soft), color grading, contrast level

Output a single paragraph of comma-separated descriptive phrases. No bullet points, no numbering, no line breaks.
Do NOT start with "This image shows" or "A photo of". Start directly with the subject description.
Do NOT include any speculative or uncertain language like "possibly", "might be", "appears to".
Be specific and concrete."""


def _load():
    global _model, _processor
    if _model is not None:
        return
    from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

    local_path = MODEL_DIR / TRANSLATOR_DIR
    model_src = str(local_path) if local_path.exists() else TRANSLATOR_REPO
    logger.info("Loading captioner: %s ...", model_src)
    _processor = AutoProcessor.from_pretrained(model_src)
    _model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_src,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    logger.info("Captioner loaded.")


def _unload():
    global _model, _processor
    if _model is None:
        return
    del _model, _processor
    _model = _processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Captioner unloaded.")


def caption_image(image_path: str) -> str:
    """Generate a detailed caption for a single image."""
    _load()
    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": CAPTION_SYSTEM + "\n\n" + CAPTION_USER},
            ],
        },
    ]

    inputs = _processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Trim input tokens to get only the generated part
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    caption = _processor.decode(generated, skip_special_tokens=True).strip()

    # Remove thinking tags if present (Qwen3.5 thinking mode)
    if "<think>" in caption:
        import re
        caption = re.sub(r"<think>.*?</think>\s*", "", caption, flags=re.DOTALL).strip()

    return caption


def auto_caption_dataset(dataset_path: str, overwrite: bool = False,
                         trigger_word: str = ""):
    """Generator: caption all images, yielding status strings.

    Loads model on first call, unloads after completion.
    If trigger_word is set, it is prepended to every caption.
    """
    ds = Path(dataset_path)
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = sorted(f for f in ds.iterdir() if f.suffix.lower() in img_exts)

    if not images:
        yield "No images found in dataset"
        return

    to_caption = images if overwrite else [f for f in images if not f.with_suffix(".txt").exists()]
    if not to_caption:
        yield "All images already have captions"
        return

    try:
        yield f"Loading captioner (Qwen3.5-2B)..."
        _load()

        trigger = trigger_word.strip().rstrip(",").strip()

        for i, img_path in enumerate(to_caption, 1):
            try:
                cap = caption_image(str(img_path))
                if trigger:
                    cap = f"{trigger}, {cap}"
                img_path.with_suffix(".txt").write_text(cap, encoding="utf-8")
                yield f"[{i}/{len(to_caption)}] {img_path.name}: {cap[:80]}"
            except Exception as e:
                logger.error("Caption failed for %s: %s", img_path.name, e)
                yield f"[{i}/{len(to_caption)}] {img_path.name}: ERROR - {e}"

        yield f"Done! Captioned {len(to_caption)} images."
    except Exception as e:
        yield f"Error: {e}"
    finally:
        _unload()
