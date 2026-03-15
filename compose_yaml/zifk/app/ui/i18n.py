"""Internationalization — key-based multi-language support for ZIFK Gradio UI.

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
    "header_title":              {"en": "ZIFK",                          "ko": "ZIFK"},

    # Tab names
    "tab_generate":              {"en": "Generate",                      "ko": "생성"},
    "tab_edit":                  {"en": "Edit",                          "ko": "편집"},
    "tab_compare":               {"en": "Compare",                      "ko": "비교"},
    "tab_settings":              {"en": "Settings",                      "ko": "설정"},
    "tab_history":               {"en": "History",                       "ko": "히스토리"},

    # Generate tab — model selector
    "model":                     {"en": "Model",                         "ko": "모델"},
    "zit_fast":                  {"en": "ZIT (Fast)",                    "ko": "ZIT (빠른)"},
    "zib_creative":              {"en": "ZIB (Creative)",                "ko": "ZIB (창의적)"},
    "klein_distilled":           {"en": "Klein (Distilled)",             "ko": "Klein (증류)"},
    "klein_base":                {"en": "Klein Base",                    "ko": "Klein 베이스"},

    # Common controls
    "prompt":                    {"en": "Prompt",                        "ko": "프롬프트"},
    "prompt_placeholder":        {"en": "Describe your image...",        "ko": "이미지를 설명하세요..."},
    "negative_prompt":           {"en": "Negative Prompt",               "ko": "네거티브 프롬프트"},
    "resolution":                {"en": "Resolution (WxH)",              "ko": "해상도 (가로x세로)"},
    "seed":                      {"en": "Seed (-1=random)",              "ko": "시드 (-1=랜덤)"},
    "num_images":                {"en": "Num Images",                    "ko": "이미지 수"},
    "generate":                  {"en": "Generate",                      "ko": "생성"},
    "generated_image":           {"en": "Generated Image",               "ko": "생성 이미지"},
    "info":                      {"en": "Info",                          "ko": "정보"},
    "send_to_edit":              {"en": "Send to Edit",                  "ko": "편집으로 보내기"},

    # Z-Image advanced
    "zimage_advanced":           {"en": "Z-Image Advanced",              "ko": "Z-Image 고급 설정"},
    "steps":                     {"en": "Steps",                         "ko": "스텝 수"},
    "guidance_scale":            {"en": "Guidance Scale",                "ko": "가이던스 스케일"},
    "cfg_normalization":         {"en": "CFG Normalization",             "ko": "CFG 정규화"},
    "cfg_truncation":            {"en": "CFG Truncation",                "ko": "CFG 절삭"},
    "max_sequence_length":       {"en": "Max Sequence Length",           "ko": "최대 시퀀스 길이"},

    # Klein advanced
    "klein_advanced":            {"en": "Klein Advanced",                "ko": "Klein 고급 설정"},
    "guidance":                  {"en": "Guidance",                      "ko": "가이던스"},

    # Sample prompts
    "sample_1":                  {"en": "Sample 1",                      "ko": "샘플 1"},
    "sample_2":                  {"en": "Sample 2",                      "ko": "샘플 2"},
    "sample_3":                  {"en": "Sample 3",                      "ko": "샘플 3"},

    # Edit tab
    "mode":                      {"en": "Mode",                          "ko": "모드"},
    "edit_single_ref":           {"en": "Edit (Single Ref)",             "ko": "편집 (단일 참조)"},
    "multi_reference":           {"en": "Multi-Reference",               "ko": "다중 참조"},
    "input_image":               {"en": "Input Image",                   "ko": "입력 이미지"},
    "reference_images":          {"en": "Reference Images (Multi-Ref)",  "ko": "참조 이미지 (다중 참조)"},
    "add_reference":             {"en": "+ Add Reference",               "ko": "+ 참조 추가"},
    "clear_references":          {"en": "Clear References",              "ko": "참조 초기화"},
    "match_image_size":          {"en": "Match Image Size",              "ko": "이미지 크기 맞추기"},
    "klein_parameters":          {"en": "Klein Parameters",              "ko": "Klein 파라미터"},
    "klein_variant":             {"en": "Klein Variant",                 "ko": "Klein 변형"},
    "distilled":                 {"en": "Distilled",                     "ko": "증류"},
    "base":                      {"en": "Base",                          "ko": "베이스"},
    "edit_prompt_placeholder":   {"en": "Describe the edit or generation...", "ko": "편집 또는 생성 내용을 설명하세요..."},

    # Compare tab
    "compare_prompt_placeholder": {"en": "Same prompt across models...", "ko": "모든 모델에 동일한 프롬프트..."},
    "seed_fixed":                {"en": "Seed (fixed recommended)",      "ko": "시드 (고정 권장)"},
    "comparison_results":        {"en": "Comparison Results",            "ko": "비교 결과"},
    "model_parameters":          {"en": "Model Parameters",              "ko": "모델 파라미터"},
    "zib_negative_prompt":       {"en": "ZIB Negative Prompt",           "ko": "ZIB 네거티브 프롬프트"},
    "zib_steps":                 {"en": "ZIB Steps",                     "ko": "ZIB 스텝"},
    "zib_cfg":                   {"en": "ZIB CFG",                       "ko": "ZIB CFG"},
    "klein_steps":               {"en": "Klein Steps",                   "ko": "Klein 스텝"},
    "klein_guidance":            {"en": "Klein Guidance",                "ko": "Klein 가이던스"},
    "klein_base_steps":          {"en": "Klein Base Steps",              "ko": "Klein Base 스텝"},
    "klein_base_guidance":       {"en": "Klein Base Guidance",           "ko": "Klein Base 가이던스"},
    "compare":                   {"en": "Compare",                       "ko": "비교"},

    # Settings tab
    "model_settings":            {"en": "Model Settings",                "ko": "모델 설정"},
    "model_directory":           {"en": "Model Directory",               "ko": "모델 디렉토리"},
    "apply":                     {"en": "Apply",                         "ko": "적용"},
    "status":                    {"en": "Status",                        "ko": "상태"},
    "check_models":              {"en": "Check Models",                  "ko": "모델 확인"},
    "model_status":              {"en": "Model Status",                  "ko": "모델 상태"},
    "attention_backend":         {"en": "Z-Image Attention Backend",     "ko": "Z-Image 어텐션 백엔드"},
    "attention_info":            {"en": "native=SDPA(auto FA2), flash=FA2, _native_flash=force SDPA flash",
                                  "ko": "native=SDPA(자동 FA2), flash=FA2, _native_flash=강제 SDPA flash"},
    "lora_download":             {"en": "LoRA Download",                 "ko": "LoRA 다운로드"},
    "model_family":              {"en": "Model Family",                  "ko": "모델 패밀리"},
    "hf_repo_or_url":            {"en": "HuggingFace Repo ID or URL",   "ko": "HuggingFace 레포 ID 또는 URL"},
    "filename_in_repo":          {"en": "Filename in Repo (e.g. model.safetensors)",
                                  "ko": "레포 내 파일명 (예: model.safetensors)"},
    "save_as":                   {"en": "Save As (optional)",            "ko": "저장 이름 (선택)"},
    "download":                  {"en": "Download",                      "ko": "다운로드"},
    "download_status":           {"en": "Download Status",               "ko": "다운로드 상태"},
    "installed_loras":           {"en": "Installed LoRAs",               "ko": "설치된 LoRA"},
    "zimage_loras":              {"en": "Z-Image LoRAs",                 "ko": "Z-Image LoRA 목록"},
    "klein_loras":               {"en": "Klein LoRAs",                   "ko": "Klein LoRA 목록"},
    "language":                  {"en": "Language",                      "ko": "언어"},

    # History tab
    "generation_history":        {"en": "Generation History",            "ko": "생성 히스토리"},
    "refresh":                   {"en": "Refresh",                       "ko": "새로고침"},
    "delete_selected":           {"en": "Delete Selected",               "ko": "선택 삭제"},
    "delete_all":                {"en": "Delete All",                    "ko": "전체 삭제"},
    "clear_cache":               {"en": "Clear Cache",                   "ko": "캐시 정리"},
    "generated_images":          {"en": "Generated Images",              "ko": "생성 이미지 목록"},
    "selected_file":             {"en": "Selected File",                 "ko": "선택된 파일"},
    "file_info":                 {"en": "File Info",                     "ko": "파일 정보"},
    "confirm_delete_all":        {"en": "Delete ALL generated images?",  "ko": "생성된 이미지를 모두 삭제하시겠습니까?"},

    # Kill button
    "kill":                      {"en": "Kill (emergency stop)",         "ko": "중지 (긴급 정지)"},
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
        const saved = localStorage.getItem('zifk-lang');
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
        localStorage.setItem('zifk-lang', lang);
        translateDOM(lang);
        const confirmMap = LANG_STRINGS[lang] || {};
        window._zifk_confirm_msg = confirmMap['Delete ALL generated images?'] || 'Delete ALL generated images?';
        applying = false;
    }

    window._zifk_setLang = function(lang) {
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
