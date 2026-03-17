"""Lightweight translation module using NLLB-200-distilled-600M (BF16)."""

import logging
import torch

from zit_config import TRANSLATOR_DIR, TRANSLATOR_REPO, MODEL_DIR

logger = logging.getLogger(__name__)

_model = None
_tokenizer = None

# NLLB-200 language codes mapping: (display_code, display_name, nllb_code)
# Order: English first, then CJK, then alphabetical
LANGUAGES = [
    ("en", "English", "eng_Latn"),
    ("zh", "Chinese (中文)", "zho_Hans"),
    ("ja", "Japanese (日本語)", "jpn_Jpan"),
    ("ko", "Korean (한국어)", "kor_Hang"),
    ("ar", "Arabic (العربية)", "arb_Arab"),
    ("bn", "Bengali (বাংলা)", "ben_Beng"),
    ("cs", "Czech (Čeština)", "ces_Latn"),
    ("da", "Danish (Dansk)", "dan_Latn"),
    ("de", "German (Deutsch)", "deu_Latn"),
    ("el", "Greek (Ελληνικά)", "ell_Grek"),
    ("es", "Spanish (Español)", "spa_Latn"),
    ("fi", "Finnish (Suomi)", "fin_Latn"),
    ("fr", "French (Français)", "fra_Latn"),
    ("he", "Hebrew (עברית)", "heb_Hebr"),
    ("hi", "Hindi (हिन्दी)", "hin_Deva"),
    ("hu", "Hungarian (Magyar)", "hun_Latn"),
    ("id", "Indonesian (Bahasa)", "ind_Latn"),
    ("it", "Italian (Italiano)", "ita_Latn"),
    ("ms", "Malay (Melayu)", "zsm_Latn"),
    ("nl", "Dutch (Nederlands)", "nld_Latn"),
    ("no", "Norwegian (Norsk)", "nob_Latn"),
    ("pl", "Polish (Polski)", "pol_Latn"),
    ("pt", "Portuguese (Português)", "por_Latn"),
    ("ro", "Romanian (Română)", "ron_Latn"),
    ("ru", "Russian (Русский)", "rus_Cyrl"),
    ("sv", "Swedish (Svenska)", "swe_Latn"),
    ("th", "Thai (ไทย)", "tha_Thai"),
    ("tr", "Turkish (Türkçe)", "tur_Latn"),
    ("uk", "Ukrainian (Українська)", "ukr_Cyrl"),
    ("vi", "Vietnamese (Tiếng Việt)", "vie_Latn"),
]

# Lookup: short code -> NLLB code
_NLLB_MAP = {code: nllb for code, _, nllb in LANGUAGES}

LANG_CHOICES = [f"{name} [{code}]" for code, name, _ in LANGUAGES]
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
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    local_path = MODEL_DIR / TRANSLATOR_DIR
    model_src = str(local_path) if local_path.exists() else TRANSLATOR_REPO
    logger.info("Loading translator: %s (BF16)...", model_src)
    _tokenizer = AutoTokenizer.from_pretrained(model_src)
    _model = AutoModelForSeq2SeqLM.from_pretrained(
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
    nllb_code = _NLLB_MAP.get(lang_code, "eng_Latn")
    forced_bos_token_id = _tokenizer.convert_tokens_to_ids(nllb_code)
    inputs = _tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(_model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=512,
        )
    result = _tokenizer.decode(outputs[0], skip_special_tokens=True)
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
