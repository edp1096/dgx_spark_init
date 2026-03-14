"""Internationalization — key-based multi-language support for Gradio UI.

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
    "header_title":              {"en": "LTX-2 Gradio UI",             "ko": "LTX-2 Gradio UI"},

    # Tab names
    "tab_distilled":             {"en": "ti2vid (distilled)",          "ko": "ti2vid (distilled)"},
    "tab_ti2vid":                {"en": "ti2vid",                      "ko": "ti2vid"},
    "tab_iclora":                {"en": "IC-LoRA",                     "ko": "IC-LoRA"},
    "tab_keyframe":              {"en": "Keyframe Interpolation",      "ko": "키프레임 보간"},
    "tab_a2vid":                 {"en": "Audio → Video",               "ko": "오디오 → 영상"},
    "tab_retake":                {"en": "Retake",                      "ko": "리테이크"},
    "tab_settings":              {"en": "Settings",                    "ko": "설정"},
    "tab_history":               {"en": "History",                     "ko": "히스토리"},

    # Tab descriptions
    "desc_ti2vid":               {"en": "2-stage generation (dev model). Supports negative prompt and guidance parameters.",
                                  "ko": "2단계 생성 (dev 모델). 네거티브 프롬프트 및 가이던스 파라미터 지원."},
    "desc_distilled":            {"en": "Fast 8-step generation. Negative prompt via NAG guidance (2x slower when enabled).",
                                  "ko": "빠른 8스텝 생성. NAG 가이던스로 네거티브 프롬프트 지원 (사용 시 2배 느림)."},
    "desc_iclora":               {"en": "Reference video conditioned generation (distilled model). Negative prompt via NAG guidance.",
                                  "ko": "참조 영상 기반 생성 (distilled 모델). NAG 가이던스로 네거티브 프롬프트 지원."},
    "desc_keyframe":             {"en": "Interpolate between keyframe images (dev model). Supports negative prompt.",
                                  "ko": "키프레임 이미지 간 보간 (dev 모델). 네거티브 프롬프트 지원."},
    "desc_a2vid":                {"en": "Audio-conditioned video generation (dev model). Supports negative prompt.",
                                  "ko": "오디오 기반 영상 생성 (dev 모델). 네거티브 프롬프트 지원."},
    "desc_retake":               {"en": "Re-generate sections of existing video. Distilled mode disables negative prompt.",
                                  "ko": "기존 영상 구간 재생성. Distilled 모드에서는 네거티브 프롬프트 비활성."},

    # Common controls
    "prompt":                    {"en": "Prompt",                      "ko": "프롬프트"},
    "negative_prompt":           {"en": "Negative Prompt",             "ko": "네거티브 프롬프트"},
    "resolution":                {"en": "Resolution (WxH)",            "ko": "해상도 (가로x세로)"},
    "length":                    {"en": "Length",                      "ko": "길이 설정"},
    "frames_radio":              {"en": "Frames",                      "ko": "프레임"},
    "duration_radio":            {"en": "Duration (sec)",              "ko": "길이 (초)"},
    "frames_label":              {"en": "Frames (8k+1)",               "ko": "프레임 수 (8k+1)"},
    "duration_label":            {"en": "Duration (sec)",              "ko": "길이 (초)"},
    "fps":                       {"en": "FPS",                         "ko": "FPS"},
    "steps":                     {"en": "Steps",                       "ko": "스텝 수"},
    "seed":                      {"en": "Seed (-1=random)",            "ko": "시드 (-1=랜덤)"},
    "sampler":                   {"en": "Sampler",                     "ko": "샘플러"},
    "enhance_prompt":            {"en": "Enhance Prompt",              "ko": "프롬프트 보정"},
    "fp8":                       {"en": "FP8 Quantization",            "ko": "FP8 양자화"},
    "generate":                  {"en": "Generate",                    "ko": "생성"},
    "generated_video":           {"en": "Generated Video",             "ko": "생성 영상"},
    "info":                      {"en": "Info",                        "ko": "정보"},

    # Conditioning image
    "conditioning_images":       {"en": "Conditioning Images",         "ko": "이미지 컨디셔닝"},
    "conditioning_image":        {"en": "Conditioning Image",          "ko": "이미지 컨디셔닝"},
    "primary_image":             {"en": "Primary Image (Frame 0)",     "ko": "기본 이미지 (프레임 0)"},
    "primary_strength":          {"en": "Primary Strength",            "ko": "기본 강도"},
    "image_optional":            {"en": "Image (optional)",            "ko": "이미지 (선택)"},
    "image_strength":            {"en": "Image Strength",              "ko": "이미지 강도"},
    "additional_images":         {"en": "Additional Conditioning Images", "ko": "추가 이미지 컨디셔닝"},
    "add_cond_image":            {"en": "+ Add Conditioning Image",    "ko": "+ 이미지 컨디셔닝 추가"},
    "extra_image_n":             {"en": "Image {n}",                   "ko": "이미지 {n}"},
    "extra_frame_index":         {"en": "Frame Index",                 "ko": "프레임 인덱스"},
    "extra_strength":            {"en": "Strength",                    "ko": "강도"},

    # Guidance
    "guidance":                  {"en": "Guidance (advanced)",          "ko": "가이던스 (고급)"},
    "video_guidance":            {"en": "Video Guidance",               "ko": "비디오 가이던스"},
    "audio_guidance":            {"en": "Audio Guidance",               "ko": "오디오 가이던스"},
    "cfg_scale":                 {"en": "CFG Scale",                    "ko": "CFG 스케일"},
    "stg_scale":                 {"en": "STG Scale",                    "ko": "STG 스케일"},
    "rescale":                   {"en": "Rescale",                      "ko": "리스케일"},
    "modality_scale":            {"en": "Modality Scale",               "ko": "모달리티 스케일"},
    "stg_blocks":                {"en": "STG Blocks",                   "ko": "STG 블록"},
    "skip_step":                 {"en": "Skip Step",                    "ko": "스킵 스텝"},
    "skip_step_info":            {"en": "Skip STG guidance every N steps (0=none, higher=faster)",
                                  "ko": "N 스텝마다 STG 가이던스 스킵 (0=없음, 높을수록 빠름)"},
    "image_crf":                 {"en": "Image CRF",                    "ko": "이미지 CRF"},
    "image_crf_info":            {"en": "Compression quality (0=lossless, 51=worst)",
                                  "ko": "압축 품질 (0=무손실, 51=최저)"},

    # Prompt Constructor
    "prompt_constructor":        {"en": "Prompt Constructor",            "ko": "프롬프트 구성기"},
    "pc_style":                  {"en": "Style",                         "ko": "스타일"},
    "pc_shot":                   {"en": "Shot Type",                     "ko": "샷 타입"},
    "pc_subject":                {"en": "Subject & Action",              "ko": "주제 & 동작"},
    "pc_environment":            {"en": "Environment",                   "ko": "환경"},
    "pc_lighting":               {"en": "Lighting",                      "ko": "조명"},
    "pc_camera":                 {"en": "Camera Movement",               "ko": "카메라 움직임"},
    "pc_insert":                 {"en": "Insert into Prompt",            "ko": "프롬프트에 삽입"},

    # Disable Audio
    "disable_audio":             {"en": "Disable Audio",                 "ko": "오디오 비활성화"},

    # Placeholders
    "ph_video":                  {"en": "Describe your video...",                  "ko": "영상을 설명하세요..."},
    "ph_transform":              {"en": "Describe the transformation...",          "ko": "변환 내용을 설명하세요..."},
    "ph_interpolation":          {"en": "Describe the interpolation...",           "ko": "보간 내용을 설명하세요..."},
    "ph_audio_video":            {"en": "Describe the video for this audio...",    "ko": "오디오에 맞는 영상을 설명하세요..."},
    "ph_retake":                 {"en": "Describe the regenerated section...",     "ko": "재생성할 구간을 설명하세요..."},

    # Sample buttons
    "sample_1":                  {"en": "Sample 1",                    "ko": "샘플 1"},
    "sample_2":                  {"en": "Sample 2",                    "ko": "샘플 2"},
    "sample_3":                  {"en": "Sample 3",                    "ko": "샘플 3"},
    "sample_4":                  {"en": "Sample 4",                    "ko": "샘플 4"},
    "sample_5":                  {"en": "Sample 5",                    "ko": "샘플 5"},

    # Distilled tab
    "distilled_note":            {"en": "Fixed 8-step distilled schedule. NAG guidance available.",
                                  "ko": "고정 8스텝 distilled 스케줄. NAG 가이던스 사용 가능."},

    # NAG
    "nag_scale":                 {"en": "NAG Scale",                    "ko": "NAG 스케일"},
    "nag_scale_info":            {"en": "Guidance strength (1.0=off, higher=stronger, doubles inference time)",
                                  "ko": "가이던스 강도 (1.0=비활성, 높을수록 강함, 추론 시간 2배)"},
    "nag_alpha":                 {"en": "NAG Alpha (Rescale)",           "ko": "NAG 알파 (리스케일)"},
    "nag_alpha_info":            {"en": "CFG rescale factor (0=off, higher=reduce artifacts)",
                                  "ko": "CFG 리스케일 계수 (0=비활성, 높을수록 아티팩트 감소)"},
    "lora_strength":             {"en": "Distilled LoRA Strength",       "ko": "Distilled LoRA 강도"},
    "lora_strength_info":        {"en": "Stage 2 distilled LoRA strength (lower=less distilled artifacts)",
                                  "ko": "Stage 2 distilled LoRA 강도 (낮을수록 잡티 감소)"},
    "custom_lora":               {"en": "Custom LoRA",                  "ko": "커스텀 LoRA"},
    "add_lora":                  {"en": "+ Add LoRA",                   "ko": "+ LoRA 추가"},
    "refresh_loras":             {"en": "Refresh",                      "ko": "새로고침"},
    "lora_file":                 {"en": "LoRA File",                    "ko": "LoRA 파일"},
    "lora_strength_custom":      {"en": "Strength",                     "ko": "강도"},

    # IC-LoRA tab
    "reference_video":           {"en": "Reference Video",             "ko": "참조 영상"},
    "reference_strength":        {"en": "Reference Strength",          "ko": "참조 강도"},
    "iclora_type":               {"en": "IC-LoRA Type",                "ko": "IC-LoRA 유형"},
    "iclora_type_info":          {"en": "Union Control: Preserve overall structure of reference video | Motion Track: Follow motion trajectory",
                                  "ko": "Union Control: 참조 영상의 전체 구조 유지 | Motion Track: 모션 궤적 추적"},
    "attention_strength":        {"en": "Attention Strength",          "ko": "어텐션 강도"},
    "cond_preprocess":           {"en": "Conditioning Preprocess",    "ko": "컨디셔닝 전처리"},
    "canny_low":                 {"en": "Canny Low",                  "ko": "Canny 하한"},
    "canny_high":                {"en": "Canny High",                 "ko": "Canny 상한"},
    "preprocess_preview":        {"en": "Preprocessing Preview",      "ko": "전처리 미리보기"},
    "suggest_prompt_image":      {"en": "Suggest Prompt from Image",  "ko": "이미지에서 프롬프트 제안"},
    "suggest_prompt_ref":        {"en": "Suggest Prompt from Reference", "ko": "레퍼런스에서 프롬프트 제안"},
    "skip_upscale":              {"en": "Skip Upscale (half res)",     "ko": "업스케일 건너뛰기 (절반 해상도)"},

    # Keyframe tab
    "keyframe_images":           {"en": "Keyframe Images",             "ko": "키프레임 이미지"},
    "frame_indices":             {"en": "Frame Indices (comma-separated)", "ko": "프레임 인덱스 (쉼표 구분)"},
    "keyframe_strength":         {"en": "Keyframe Strength",           "ko": "키프레임 강도"},

    # Audio tab
    "audio_file":                {"en": "Audio File",                  "ko": "오디오 파일"},
    "audio_start":               {"en": "Audio Start (sec)",           "ko": "오디오 시작 (초)"},
    "audio_max_duration":        {"en": "Max Duration (0=all)",        "ko": "최대 길이 (0=전체)"},

    # Retake tab
    "source_video":              {"en": "Source Video",                "ko": "원본 영상"},
    "start_time":                {"en": "Start Time (sec)",            "ko": "시작 시간 (초)"},
    "end_time":                  {"en": "End Time (sec)",              "ko": "종료 시간 (초)"},
    "regen_video":               {"en": "Regenerate Video",            "ko": "영상 재생성"},
    "regen_audio":               {"en": "Regenerate Audio",            "ko": "오디오 재생성"},
    "distilled_mode":            {"en": "Distilled Mode",              "ko": "Distilled 모드"},
    "result_video":              {"en": "Result Video",                "ko": "결과 영상"},

    # Settings tab
    "model_paths":               {"en": "Model Paths",                 "ko": "모델 경로"},
    "model_directory":           {"en": "Model Directory",             "ko": "모델 디렉토리"},
    "apply":                     {"en": "Apply",                       "ko": "적용"},
    "check_models":              {"en": "Check Models",                "ko": "모델 확인"},
    "status":                    {"en": "Status",                      "ko": "상태"},
    "language":                  {"en": "Language",                    "ko": "언어"},
    "ui_language":               {"en": "UI Language",                 "ko": "UI 언어"},

    # Custom LoRA download
    "custom_lora_download":      {"en": "Custom LoRA Download",        "ko": "커스텀 LoRA 다운로드"},
    "lora_download_source":      {"en": "HuggingFace Repo ID or Direct URL (.safetensors)",
                                  "ko": "HuggingFace Repo ID 또는 직접 URL (.safetensors)"},
    "lora_download_filename":    {"en": "Filename in Repo (for HF repo, e.g. model.safetensors)",
                                  "ko": "레포 내 파일명 (HF repo인 경우, 예: model.safetensors)"},
    "lora_download_savename":    {"en": "Save As (optional, auto-detected if empty)",
                                  "ko": "저장 파일명 (선택, 비우면 자동 감지)"},
    "lora_download_btn":         {"en": "Download",                    "ko": "다운로드"},
    "lora_download_status":      {"en": "Download Status",             "ko": "다운로드 상태"},
    "lora_installed_list":       {"en": "Installed Custom LoRAs",      "ko": "설치된 커스텀 LoRA 목록"},
    "lora_delete_btn":           {"en": "Delete Selected",             "ko": "선택 삭제"},
    "lora_refresh_list":         {"en": "Refresh List",                "ko": "목록 새로고침"},

    # Presets
    "presets":                   {"en": "Presets",                      "ko": "프리셋"},
    "export":                    {"en": "Export",                       "ko": "내보내기"},
    "import_json":               {"en": "Import (.json)",               "ko": "가져오기 (.json)"},
    "preset_loaded":             {"en": "Preset loaded.",               "ko": "프리셋을 불러왔습니다."},
    "preset_tab_mismatch":       {"en": "This preset is for '{tab}' tab. Please import it in the correct tab.",
                                  "ko": "이 프리셋은 '{tab}' 탭용입니다. 해당 탭에서 가져와 주세요."},
    "preset_tab_mismatch_short": {"en": "Tab mismatch: '{tab}' → use '{tab}' tab",
                                  "ko": "탭 불일치: '{tab}' → '{tab}' 탭에서 사용"},

    # History tab
    "generation_history":        {"en": "Generation History",          "ko": "생성 히스토리"},
    "refresh":                   {"en": "Refresh",                     "ko": "새로고침"},
    "delete_selected":           {"en": "Delete Selected",             "ko": "선택 삭제"},
    "delete_all":                {"en": "Delete All",                  "ko": "전체 삭제"},
    "generated_videos":          {"en": "Generated Videos",            "ko": "생성 영상 목록"},
    "preview":                   {"en": "Preview",                     "ko": "미리보기"},
    "file_info":                 {"en": "File Info",                   "ko": "파일 정보"},
    "confirm_delete_all":        {"en": "Delete ALL generated videos?","ko": "생성된 영상을 모두 삭제하시겠습니까?"},
}


def get_i18n_js() -> str:
    """Generate the client-side i18n JavaScript."""
    import json

    # Build per-language maps: { "en_text": "target_text" } for each lang
    # The JS always translates FROM whatever is currently displayed TO the target lang.
    # We build: LANG_STRINGS[lang] = { en_text: lang_text } for all langs
    # and      EN_TEXTS = { any_lang_text: en_text } as reverse lookup
    lang_strings = {}  # lang → {en_text: translated_text}
    all_to_en = {}     # any_text → en_text (for reverse lookup)

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
        const saved = localStorage.getItem('ltx2-lang');
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
            // Skip the language selector radio labels
            if (el.closest('#lang-selector')) return;
            el.childNodes.forEach(node => {
                if (node.nodeType !== Node.TEXT_NODE) return;
                const t = node.textContent.trim();
                if (!t) return;
                // Current text → English key
                const enKey = ALL_TO_EN[t] || t;
                // English key → target lang
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
        localStorage.setItem('ltx2-lang', lang);
        translateDOM(lang);
        // Confirm dialog
        const confirmMap = LANG_STRINGS[lang] || {};
        window._ltx2_confirm_msg = confirmMap['Delete ALL generated videos?'] || 'Delete ALL generated videos?';
        applying = false;
    }

    window._ltx2_setLang = function(lang) {
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
        // Try to sync radio once when lang-selector appears in DOM
        if (!radioSynced) {
            radioSynced = syncRadio(currentLang);
        }
        // Translate only added nodes instead of full DOM scan
        let hasNew = false;
        for (const m of mutations) {
            for (const node of m.addedNodes) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    hasNew = true;
                    translateNodes(currentLang, node);
                }
            }
        }
        // Fallback: if no element nodes were added but DOM changed, debounce full scan
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
        // Try immediately in case Settings tab is already rendered
        radioSynced = syncRadio(lang);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
"""
