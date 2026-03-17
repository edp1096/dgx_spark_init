"""Lightweight translation module using MADLAD-400-3B-MT (INT8)."""

import logging
import torch

from zit_config import MADLAD_DIR, MADLAD_REPO, MODEL_DIR

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    local_path = MODEL_DIR / MADLAD_DIR
    model_src = str(local_path) if local_path.exists() else MADLAD_REPO
    logger.info("Loading translator: %s (INT8)...", model_src)
    _tokenizer = T5Tokenizer.from_pretrained(model_src)
    _model = T5ForConditionalGeneration.from_pretrained(
        model_src,
        device_map="auto",
        load_in_8bit=True,
    )
    _model.eval()
    logger.info("Translator loaded.")


def translate_to_en(text: str) -> str:
    """Translate text to English. Returns original if already English or empty."""
    if not text or not text.strip():
        return ""
    _load()
    input_text = f"<2en> {text}"
    inputs = _tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=512)
    result = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


def unload():
    """Unload translator model to free VRAM."""
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    torch.cuda.empty_cache()
    logger.info("Translator unloaded.")
