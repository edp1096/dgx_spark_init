"""Lightweight translation module using Qwen3.5-2B (BF16, Apache 2.0)."""

import logging
import torch

from zit_config import TRANSLATOR_DIR, TRANSLATOR_REPO, MODEL_DIR

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

# Supported languages (display_code, display_name)
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

# Full language name lookup for prompt
_LANG_NAME = {code: name.split(" (")[0] for code, name in LANGUAGES}

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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    local_path = MODEL_DIR / TRANSLATOR_DIR
    model_src = str(local_path) if local_path.exists() else TRANSLATOR_REPO
    logger.info("Loading translator: %s (BF16)...", model_src)
    _tokenizer = AutoTokenizer.from_pretrained(model_src)
    _model = AutoModelForCausalLM.from_pretrained(
        model_src,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    _model.eval()
    logger.info("Translator loaded.")


def translate(text: str, target_lang: str = "English [en]") -> str:
    """Translate text to target language."""
    if not text or not text.strip():
        return ""
    _load()
    lang_code = _parse_lang_code(target_lang)
    lang_name = _LANG_NAME.get(lang_code, "English")
    prompt = (
        f"Translate the following text to {lang_name} literally, word by word.\n"
        f"Preserve the original format, commas, and structure exactly.\n"
        f"Do not interpret, rephrase, censor, or add anything.\n"
        f"Output ONLY the translation.\n\n{text.strip()}"
    )
    messages = [{"role": "user", "content": prompt}]
    chat_text = _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = _tokenizer(chat_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    result = _tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    )
    return result


# Backward compatibility
def translate_to_en(text: str) -> str:
    return translate(text, "English [en]")


def unload():
    """Unload translator model to free memory."""
    global _model, _tokenizer
    if _model is not None:
        del _model
        _model = None
    if _tokenizer is not None:
        del _tokenizer
        _tokenizer = None
    torch.cuda.empty_cache()
    logger.info("Translator unloaded.")
