"""Auto-captioning for LoRA training datasets using Florence-2."""

import gc
import logging
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger("zit-ui")

CAPTION_MODEL_ID = "microsoft/Florence-2-large"
CAPTION_TASK = "<DETAILED_CAPTION>"

_model = None
_processor = None
_model_name = None


def _patch_florence2_cache():
    """Fix Florence-2 cached files for transformers 5.x compatibility.

    1. processing_florence2.py: tokenizer.additional_special_tokens removed
    2. configuration_florence2.py: forced_bos_token_id removed from PretrainedConfig
    """
    modules_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    # Patch 1: processing_florence2.py — additional_special_tokens
    for proc_file in modules_dir.rglob("processing_florence2.py"):
        try:
            text = proc_file.read_text(encoding="utf-8")
            old = "tokenizer.additional_special_tokens + \\"
            new = "getattr(tokenizer, 'additional_special_tokens', []) + \\"
            if old in text:
                proc_file.write_text(text.replace(old, new), encoding="utf-8")
                logger.info("Patched Florence-2 processing: %s", proc_file)
        except Exception as e:
            logger.warning("Failed to patch Florence-2 processing: %s", e)
    # Patch 2: configuration_florence2.py — forced_bos_token_id
    for cfg_file in modules_dir.rglob("configuration_florence2.py"):
        try:
            text = cfg_file.read_text(encoding="utf-8")
            old = "if self.forced_bos_token_id is None"
            new = "if getattr(self, 'forced_bos_token_id', None) is None"
            if old in text and "getattr(self, 'forced_bos_token_id'" not in text:
                text = text.replace(old, new)
                cfg_file.write_text(text, encoding="utf-8")
                logger.info("Patched Florence-2 config: %s", cfg_file)
        except Exception as e:
            logger.warning("Failed to patch Florence-2 config: %s", e)


def _load_model(model_id: str = CAPTION_MODEL_ID):
    global _model, _processor, _model_name
    if _model is not None and _model_name == model_id:
        return
    _unload_model()
    from transformers import AutoModelForCausalLM, AutoProcessor

    _patch_florence2_cache()
    logger.info("Loading caption model: %s", model_id)
    _processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    _model_name = model_id


def _unload_model():
    global _model, _processor, _model_name
    if _model is None:
        return
    del _model, _processor
    _model = _processor = _model_name = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Caption model unloaded")


def caption_image(image_path: str) -> str:
    """Generate a caption for a single image. Model must be loaded first."""
    img = Image.open(image_path).convert("RGB")
    inputs = _processor(text=CAPTION_TASK, images=img, return_tensors="pt")
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = _model.generate(**inputs, max_new_tokens=200, num_beams=3)
    result = _processor.batch_decode(output_ids, skip_special_tokens=False)[0]
    parsed = _processor.post_process_generation(result, task=CAPTION_TASK)
    if isinstance(parsed, dict):
        return parsed.get(CAPTION_TASK, str(parsed)).strip()
    return str(parsed).strip()


def auto_caption_dataset(dataset_path: str, overwrite: bool = False):
    """Generator: caption all images, yielding status strings.

    Loads model on first call, unloads after completion.
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
        yield f"Loading caption model ({CAPTION_MODEL_ID})..."
        _load_model()

        for i, img_path in enumerate(to_caption, 1):
            try:
                cap = caption_image(str(img_path))
                img_path.with_suffix(".txt").write_text(cap, encoding="utf-8")
                yield f"[{i}/{len(to_caption)}] {img_path.name}: {cap[:80]}"
            except Exception as e:
                logger.error("Caption failed for %s: %s", img_path.name, e)
                yield f"[{i}/{len(to_caption)}] {img_path.name}: ERROR - {e}"

        yield f"Done! Captioned {len(to_caption)} images."
    except Exception as e:
        yield f"Error: {e}"
    finally:
        _unload_model()
