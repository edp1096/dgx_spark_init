"""Internationalization — key-based multi-language support for ZIT Gradio UI.

To add a new language:
  1. Add a new column to STRINGS (e.g. "ja": "日本語テキスト")
  2. Add the language name to LANGUAGES dict
  3. That's it — JS handles the rest.
"""

# Language code → display name (shown in Settings radio)
LANGUAGES = {
    "en": "English",
    "ko": "한국어",
}

# Translation table: key → {lang: text}
# "en" values are also the default labels used in Gradio components.
STRINGS = {
    # Header
    "header_title":              {"en": "ZIT",                             "ko": "ZIT"},

    # Tab names
    "tab_generate":              {"en": "Generate",                        "ko": "생성"},
    "tab_controlnet":            {"en": "ControlNet",                      "ko": "컨트롤넷"},
    "tab_inpaint":               {"en": "Inpaint",                         "ko": "인페인트"},
    "tab_faceswap":              {"en": "FaceSwap",                        "ko": "페이스스왑"},
    "tab_settings":              {"en": "Settings",                        "ko": "설정"},
    "tab_history":               {"en": "History",                         "ko": "히스토리"},

    # Common controls
    "prompt":                    {"en": "Prompt",                          "ko": "프롬프트"},
    "prompt_placeholder":        {"en": "Describe your image...",          "ko": "이미지를 설명하세요..."},
    "negative_prompt":           {"en": "Negative Prompt",                 "ko": "네거티브 프롬프트"},
    "resolution":                {"en": "Resolution (WxH)",                "ko": "해상도 (가로x세로)"},
    "seed":                      {"en": "Seed (-1=random)",                "ko": "시드 (-1=랜덤)"},
    "num_images":                {"en": "Num Images",                      "ko": "이미지 수"},
    "generate":                  {"en": "Generate",                        "ko": "생성"},
    "generated_image":           {"en": "Generated Image",                 "ko": "생성 이미지"},
    "info":                      {"en": "Info",                            "ko": "정보"},

    # Generate tab parameters
    "steps":                     {"en": "Steps",                           "ko": "스텝 수"},
    "time_shift":                {"en": "Time Shift",                      "ko": "타임 시프트"},
    "guidance_scale":            {"en": "Guidance Scale",                   "ko": "가이던스 스케일"},
    "cfg_normalization":         {"en": "CFG Normalization",                "ko": "CFG 정규화"},
    "cfg_truncation":            {"en": "CFG Truncation",                   "ko": "CFG 절삭"},
    "max_sequence_length":       {"en": "Max Sequence Length",              "ko": "최대 시퀀스 길이"},
    "attention_backend":         {"en": "Attention Backend",                "ko": "어텐션 백엔드"},

    # ControlNet tab
    "control_mode":              {"en": "Control Mode",                    "ko": "제어 모드"},
    "input_image":               {"en": "Input Image",                     "ko": "입력 이미지"},
    "control_preview":           {"en": "Control Preview",                 "ko": "제어 미리보기"},
    "preview_preprocessor":      {"en": "Preview Preprocessor",            "ko": "전처리 미리보기"},
    "control_scale":             {"en": "Control Scale",                   "ko": "제어 강도"},
    "match_image_size":          {"en": "Match Image Size",                "ko": "이미지 크기 맞추기"},

    # Inpaint tab
    "mode":                      {"en": "Mode",                            "ko": "모드"},
    "inpaint":                   {"en": "Inpaint",                         "ko": "인페인트"},
    "outpaint":                  {"en": "Outpaint",                        "ko": "아웃페인트"},
    "draw_mask":                 {"en": "Draw Mask",                       "ko": "마스크 그리기"},
    "expand_direction":          {"en": "Expand Direction",                "ko": "확장 방향"},
    "expand_size":               {"en": "Expand Size",                     "ko": "확장 크기"},

    # FaceSwap tab
    "target_image":              {"en": "Target Image",                    "ko": "대상 이미지"},
    "source_face":               {"en": "Source Face",                     "ko": "원본 얼굴"},
    "swap_face":                 {"en": "Swap Face",                       "ko": "얼굴 교체"},
    "result":                    {"en": "Result",                          "ko": "결과"},

    # Sample prompts
    "sample_1":                  {"en": "Sample 1",                        "ko": "샘플 1"},
    "sample_2":                  {"en": "Sample 2",                        "ko": "샘플 2"},
    "sample_3":                  {"en": "Sample 3",                        "ko": "샘플 3"},

    # Settings tab
    "model_settings":            {"en": "Model Settings",                  "ko": "모델 설정"},
    "model_directory":           {"en": "Model Directory",                 "ko": "모델 디렉토리"},
    "apply":                     {"en": "Apply",                           "ko": "적용"},
    "status":                    {"en": "Status",                          "ko": "상태"},
    "check_models":              {"en": "Check Models",                    "ko": "모델 확인"},
    "model_status":              {"en": "Model Status",                    "ko": "모델 상태"},
    "lora_download":             {"en": "LoRA Download",                   "ko": "LoRA 다운로드"},
    "hf_repo_or_url":            {"en": "HuggingFace Repo ID or URL",     "ko": "HuggingFace 레포 ID 또는 URL"},
    "filename_in_repo":          {"en": "Filename in Repo (e.g. model.safetensors)",
                                  "ko": "레포 내 파일명 (예: model.safetensors)"},
    "save_as":                   {"en": "Save As (optional)",              "ko": "저장 이름 (선택)"},
    "download":                  {"en": "Download",                        "ko": "다운로드"},
    "download_status":           {"en": "Download Status",                 "ko": "다운로드 상태"},
    "installed_loras":           {"en": "Installed LoRAs",                 "ko": "설치된 LoRA"},
    "language":                  {"en": "Language",                        "ko": "언어"},

    # History tab
    "generation_history":        {"en": "Generation History",              "ko": "생성 히스토리"},
    "refresh":                   {"en": "Refresh",                         "ko": "새로고침"},
    "delete_selected":           {"en": "Delete Selected",                 "ko": "선택 삭제"},
    "delete_all":                {"en": "Delete All",                      "ko": "전체 삭제"},
    "clear_cache":               {"en": "Clear Cache",                     "ko": "캐시 정리"},
    "generated_images":          {"en": "Generated Images",                "ko": "생성 이미지 목록"},
    "selected_file":             {"en": "Selected File",                   "ko": "선택된 파일"},
    "file_info":                 {"en": "File Info",                       "ko": "파일 정보"},
    "confirm_delete_all":        {"en": "Delete ALL generated images?",    "ko": "생성된 이미지를 모두 삭제하시겠습니까?"},

    # Kill button
    "kill":                      {"en": "Kill (emergency stop)",           "ko": "중지 (긴급 정지)"},
}


def get_i18n_js() -> str:
    """Generate the client-side i18n JavaScript."""
    import json

    lang_strings = {}
    all_to_en = {}

    for _key, texts in STRINGS.items():
        en_text = texts["en"]
        for lang, translated in texts.items():
            if lang == "en":
                continue
            lang_strings.setdefault(lang, {})[en_text] = translated
            all_to_en[translated] = en_text

    lang_strings_json = json.dumps(lang_strings, ensure_ascii=False)
    all_to_en_json = json.dumps(all_to_en, ensure_ascii=False)
    languages_json = json.dumps(LANGUAGES, ensure_ascii=False)

    return """
(function() {
    const LANG_STRINGS = """ + lang_strings_json + """;
    const ALL_TO_EN = """ + all_to_en_json + """;
    const LANGUAGES = """ + languages_json + """;

    let currentLang = 'en';
    let applying = false;

    function detectLang() {
        const saved = localStorage.getItem('zit-lang');
        if (saved && (saved === 'en' || LANG_STRINGS[saved])) return saved;
        const nav = (navigator.language || navigator.userLanguage || 'en').toLowerCase();
        for (const code of Object.keys(LANG_STRINGS)) {
            if (nav.startsWith(code)) return code;
        }
        return 'en';
    }

    function translateNodes(lang, root) {
        const targetMap = LANG_STRINGS[lang] || {};

        root.querySelectorAll(
            'label, button, span, h1, h2, h3, p, em'
        ).forEach(el => {
            if (el.closest('#lang-selector')) return;
            el.childNodes.forEach(node => {
                if (node.nodeType !== Node.TEXT_NODE) return;
                const t = node.textContent.trim();
                if (!t) return;
                const enKey = ALL_TO_EN[t] || t;
                const target = lang === 'en' ? enKey : (targetMap[enKey] || enKey);
                if (target !== t) {
                    node.textContent = node.textContent.replace(t, target);
                }
            });
        });

        root.querySelectorAll('textarea[placeholder], input[placeholder]').forEach(el => {
            const ph = el.getAttribute('placeholder');
            if (!ph) return;
            const enKey = ALL_TO_EN[ph] || ph;
            const target = lang === 'en' ? enKey : (targetMap[enKey] || enKey);
            if (target !== ph) el.setAttribute('placeholder', target);
        });
    }

    function translateDOM(lang) { translateNodes(lang, document); }

    function applyLang(lang) {
        if (applying) return;
        applying = true;
        currentLang = lang;
        localStorage.setItem('zit-lang', lang);
        translateDOM(lang);
        const confirmMap = LANG_STRINGS[lang] || {};
        window._zit_confirm_msg = confirmMap['Delete ALL generated images?'] || 'Delete ALL generated images?';
        applying = false;
    }

    window._zit_setLang = function(lang) {
        if (lang !== currentLang) applyLang(lang);
    };

    let debounceTimer = null;
    let radioSynced = false;

    function syncRadio(lang) {
        const sel = document.getElementById('lang-selector');
        if (!sel) return false;
        const radios = sel.querySelectorAll('input[type="radio"]');
        if (radios.length === 0) return false;
        const target = LANGUAGES[lang] || 'English';
        radios.forEach(r => {
            const lbl = r.closest('label');
            if (lbl && lbl.textContent.trim() === target && !r.checked) {
                lbl.click();
            }
        });
        return true;
    }

    const observer = new MutationObserver((mutations) => {
        if (applying || currentLang === 'en') return;
        if (!radioSynced) {
            radioSynced = syncRadio(currentLang);
        }
        let hasNew = false;
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    hasNew = true;
                    translateNodes(currentLang, node);
                }
            }
        }
        if (!hasNew) {
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => { applyLang(currentLang); }, 100);
        }
    });

    function init() {
        const lang = detectLang();
        observer.observe(document.body, {childList: true, subtree: true});
        if (lang === 'en') return;
        applyLang(lang);
        radioSynced = syncRadio(lang);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
"""
