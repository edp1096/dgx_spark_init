"""Lightweight translation module using MADLAD-400-3B-MT (INT8)."""

import logging
import torch

from zit_config import MADLAD_DIR, MADLAD_REPO, MODEL_DIR

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

# MADLAD-400 major 30 languages: (code, display_name)
# Order: English first, then CJK, then alphabetical
LANGUAGES = [
    ("en", "English"),
    ("zh", "Chinese (中文)"),
    ("ja", "Japanese (日本語)"),
    ("ko", "Korean (한국어)"),
    ("ar", "Arabic (العربية)"),
    ("bn", "Bengali (বাংলা)"),
    ("cs", "Czech (Čeština)"),
    ("da", "Danish (Dansk)"),
    ("de", "German (Deutsch)"),
    ("el", "Greek (Ελληνικά)"),
    ("es", "Spanish (Español)"),
    ("fi", "Finnish (Suomi)"),
    ("fr", "French (Français)"),
    ("he", "Hebrew (עברית)"),
    ("hi", "Hindi (हिन्दी)"),
    ("hu", "Hungarian (Magyar)"),
    ("id", "Indonesian (Bahasa)"),
    ("it", "Italian (Italiano)"),
    ("ms", "Malay (Melayu)"),
    ("nl", "Dutch (Nederlands)"),
    ("no", "Norwegian (Norsk)"),
    ("pl", "Polish (Polski)"),
    ("pt", "Portuguese (Português)"),
    ("ro", "Romanian (Română)"),
    ("ru", "Russian (Русский)"),
    ("sv", "Swedish (Svenska)"),
    ("th", "Thai (ไทย)"),
    ("tr", "Turkish (Türkçe)"),
    ("uk", "Ukrainian (Українська)"),
    ("vi", "Vietnamese (Tiếng Việt)"),
]

LANG_CHOICES = [f"{name} [{code}]" for code, name in LANGUAGES]
DEFAULT_LANG = LANG_CHOICES[0]  # English


def _parse_lang_code(choice: str) -> str:
    """Extract language code from dropdown choice like 'English [en]'."""
    if "[" in choice and "]" in choice:
        return choice.split("[")[-1].rstrip("]")
    return "en"


def _load():
    global _model, _tokenizer
    if _model is not None:
        return
    from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
    local_path = MODEL_DIR / MADLAD_DIR
    model_src = str(local_path) if local_path.exists() else MADLAD_REPO
    logger.info("Loading translator: %s (INT8)...", model_src)
    _tokenizer = T5Tokenizer.from_pretrained(model_src)
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    _model = T5ForConditionalGeneration.from_pretrained(
        model_src,
        device_map="auto",
        quantization_config=quantization_config,
    )
    _model.eval()
    logger.info("Translator loaded.")


def translate(text: str, target_lang: str = "English [en]") -> str:
    """Translate text to target language."""
    if not text or not text.strip():
        return ""
    _load()
    lang_code = _parse_lang_code(target_lang)
    input_text = f"<2{lang_code}> {text}"
    inputs = _tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model.generate(**inputs, max_new_tokens=512)
    result = _tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result


# Backward compatibility
def translate_to_en(text: str) -> str:
    return translate(text, "English [en]")


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
