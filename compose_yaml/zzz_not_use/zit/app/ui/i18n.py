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
    "header_title":              {"en": "ZIT Gradio",                      "ko": "ZIT Gradio"},

    # Tab names
    "tab_generate":              {"en": "Generate",                        "ko": "생성"},
    "tab_inpaint":               {"en": "Inpaint",                         "ko": "인페인트"},
    "tab_inpaint_header":        {"en": "Inpaint / Outpaint",             "ko": "인페인트 / 아웃페인트"},
    "tab_train":                 {"en": "Train",                           "ko": "학습"},
    "tab_train_header":          {"en": "LoRA Training",                   "ko": "LoRA 학습"},
    "tab_settings":              {"en": "Settings",                        "ko": "설정"},
    "tab_history":               {"en": "History",                         "ko": "히스토리"},

    # Common controls
    "prompt":                    {"en": "Prompt",                          "ko": "프롬프트"},
    "negative":                  {"en": "Negative",                        "ko": "네거티브"},
    "prompt_placeholder":        {"en": "Describe your image...",          "ko": "이미지를 설명하세요..."},
    "inpaint_placeholder":       {"en": "Describe what to fill...",        "ko": "채울 내용을 설명하세요..."},
    "negative_prompt":           {"en": "Negative Prompt",                 "ko": "네거티브 프롬프트"},
    "resolution":                {"en": "Resolution (WxH)",                "ko": "해상도 (가로x세로)"},
    "seed":                      {"en": "Seed (-1=random)",                "ko": "시드 (-1=랜덤)"},
    "num_images":                {"en": "Num Images",                      "ko": "이미지 수"},
    "generate":                  {"en": "Generate",                        "ko": "생성"},
    "info":                      {"en": "Info",                            "ko": "정보"},
    "result":                    {"en": "Result",                          "ko": "결과"},

    # Translate section
    "translate":                 {"en": "Translate",                       "ko": "번역"},
    "source":                    {"en": "Source",                          "ko": "원문"},
    "target":                    {"en": "Target",                          "ko": "대상 언어"},
    "translation":               {"en": "Translation",                     "ko": "번역 결과"},
    "use":                       {"en": "Use",                             "ko": "사용"},

    # Generate tab parameters
    "steps":                     {"en": "Steps",                           "ko": "스텝 수"},
    "time_shift":                {"en": "Time Shift",                      "ko": "타임 시프트"},
    "guidance_scale":            {"en": "Guidance Scale",                   "ko": "Guidance Scale"},
    "cfg_normalization":         {"en": "CFG Normalization",                "ko": "CFG 정규화"},
    "cfg_truncation":            {"en": "CFG Truncation",                   "ko": "CFG Truncation"},
    "max_sequence_length":       {"en": "Max Sequence Length",              "ko": "최대 시퀀스 길이"},
    "attention_backend":         {"en": "Attention Backend",                "ko": "어텐션 백엔드"},
    "attn_info":                 {"en": "native=SDPA(auto FA2), flash=FA2, _native_flash=force SDPA flash",
                                  "ko": "native=SDPA(자동 FA2), flash=FA2, _native_flash=강제 SDPA flash"},

    # LoRA section (shared across tabs)
    "lora_accordion":            {"en": "LoRA",                            "ko": "LoRA"},
    "enable_lora":               {"en": "Enable LoRA",                     "ko": "LoRA 활성화"},
    "scale":                     {"en": "Scale",                           "ko": "강도"},
    "add_lora":                  {"en": "+ Add LoRA",                      "ko": "+ LoRA 추가"},
    "trigger_words":             {"en": "Trigger Words",                   "ko": "트리거 워드"},
    "lora_description":          {"en": "Description",                     "ko": "설명"},
    "lora_source":               {"en": "Source URL",                      "ko": "소스 URL"},
    "lora_notes":                {"en": "Notes",                           "ko": "메모"},
    "save_metadata":             {"en": "Save Metadata",                   "ko": "메타데이터 저장"},
    "lora_detail":               {"en": "LoRA Detail",                     "ko": "LoRA 상세"},

    # Presets section
    "presets":                   {"en": "Presets",                         "ko": "프리셋"},
    "click_to_load_preset":      {"en": "Click to load preset",           "ko": "프리셋 클릭하여 로드"},
    "save_as_preset":            {"en": "Save as Preset",                  "ko": "프리셋 저장"},
    "delete_selected_preset":    {"en": "Delete Selected Preset",          "ko": "선택 프리셋 삭제"},
    "export_json":               {"en": "Export JSON",                     "ko": "JSON 내보내기"},
    "import_json":               {"en": "Import JSON",                     "ko": "JSON 가져오기"},
    "expand":                    {"en": "Expand",                          "ko": "확장"},
    "collapse":                  {"en": "Collapse",                        "ko": "축소"},

    # ControlNet section
    "control_mode":              {"en": "Control Mode",                    "ko": "제어 모드"},
    "control_image":             {"en": "Control Image",                   "ko": "제어 이미지"},
    "control_preview":           {"en": "Control Preview",                 "ko": "제어 미리보기"},
    "preview_preprocessor":      {"en": "Preview Preprocessor",            "ko": "전처리 미리보기"},
    "control_scale":             {"en": "Control Scale",                   "ko": "Control Scale"},
    "match_image_size":          {"en": "Match Image Size",                "ko": "이미지 크기 맞추기"},
    "enable_controlnet":         {"en": "Enable ControlNet",               "ko": "ControlNet 활성화"},
    "cn_enable_info":            {"en": "ON: load ControlNet adapter (pose/depth control) / OFF: pure T2I (better face quality)",
                                  "ko": "ON: ControlNet 어댑터 로드 (포즈/뎁스 제어) / OFF: 순수 T2I (얼굴 품질 향상)"},

    # Inpaint tab
    "mode":                      {"en": "Mode",                            "ko": "모드"},
    "inpaint":                   {"en": "Inpaint",                         "ko": "인페인트"},
    "outpaint":                  {"en": "Outpaint",                        "ko": "아웃페인트"},
    "draw_mask":                 {"en": "Draw Mask (white = regenerate)",   "ko": "마스크 그리기 (흰색 = 재생성)"},
    "image":                     {"en": "Image",                           "ko": "이미지"},
    "expand_direction":          {"en": "Expand Direction",                "ko": "확장 방향"},
    "expand_size":               {"en": "Expand Size (px)",                "ko": "확장 크기 (px)"},
    "left":                      {"en": "Left",                            "ko": "왼쪽"},
    "right":                     {"en": "Right",                           "ko": "오른쪽"},
    "up":                        {"en": "Up",                              "ko": "위"},
    "down":                      {"en": "Down",                            "ko": "아래"},

    # Train tab — dataset
    "dataset":                   {"en": "Dataset",                         "ko": "데이터셋"},
    "dataset_info":              {"en": "Select a dataset or create a new one below",
                                  "ko": "데이터셋을 선택하거나 아래에서 새로 만드세요"},
    "data_images":               {"en": "Data Images",                     "ko": "데이터 이미지"},
    "dataset_images":            {"en": "Dataset Images (click to edit caption)",
                                  "ko": "데이터셋 이미지 (클릭하여 캡션 편집)"},
    "caption_editor":            {"en": "Caption Editor",                  "ko": "캡션 편집기"},
    "caption":                   {"en": "Caption",                         "ko": "캡션"},
    "caption_placeholder":       {"en": "Click an image above to edit its caption",
                                  "ko": "위의 이미지를 클릭하여 캡션을 편집하세요"},
    "save_caption":              {"en": "Save Caption",                    "ko": "캡션 저장"},
    "delete_image":              {"en": "Delete Image",                    "ko": "이미지 삭제"},
    "caption_tools":             {"en": "Caption Tools",                   "ko": "캡션 도구"},
    "keywords_to_add":           {"en": "Keywords to add",                 "ko": "추가할 키워드"},
    "prepend_to_all":            {"en": "Prepend to All",                  "ko": "전체 앞에 추가"},
    "append_to_all":             {"en": "Append to All",                   "ko": "전체 뒤에 추가"},
    "trigger_word":              {"en": "Trigger Word",                    "ko": "트리거 워드"},
    "trigger_word_info":         {"en": "Auto-prepended to every generated caption",
                                  "ko": "자동 캡션 생성 시 앞에 자동 추가"},
    "auto_caption":              {"en": "Auto-Caption (AI)",               "ko": "자동 캡션 (AI)"},
    "overwrite_existing":        {"en": "Overwrite existing",              "ko": "기존 덮어쓰기"},
    "progress":                  {"en": "Progress",                        "ko": "진행 상황"},
    "delete_all_captions":       {"en": "Delete All Captions",             "ko": "전체 캡션 삭제"},
    "manage_dataset":            {"en": "Manage Dataset",                  "ko": "데이터셋 관리"},
    "new_dataset_name":          {"en": "New Dataset Name",                "ko": "새 데이터셋 이름"},
    "create":                    {"en": "Create",                          "ko": "생성"},
    "drop_images_captions":      {"en": "Drop Images & Captions here (auto-upload)",
                                  "ko": "이미지 & 캡션 파일을 여기에 드롭 (자동 업로드)"},

    # Train tab — parameters
    "lora_name":                 {"en": "LoRA Name",                       "ko": "LoRA 이름"},
    "lora_name_info":            {"en": "Output: loras/<name>.safetensors",
                                  "ko": "출력: loras/<이름>.safetensors"},
    "rank":                      {"en": "Rank",                            "ko": "랭크"},
    "learning_rate":             {"en": "Learning Rate",                   "ko": "학습률"},
    "lora_alpha":                {"en": "LoRA Alpha",                      "ko": "LoRA 알파"},
    "lora_alpha_info":           {"en": "PEFT scaling = alpha/rank (default=rank)",
                                  "ko": "PEFT 스케일링 = alpha/rank (기본값=rank)"},
    "train_resolution":          {"en": "Resolution",                      "ko": "해상도"},
    "batch_size":                {"en": "Batch Size",                      "ko": "배치 크기"},
    "gradient_accumulation":     {"en": "Gradient Accumulation",           "ko": "그래디언트 누적"},
    "save_checkpoint_every":     {"en": "Save Checkpoint Every N Steps",   "ko": "N 스텝마다 체크포인트 저장"},
    "target_modules":            {"en": "Target Modules",                  "ko": "대상 모듈"},
    "target_modules_info":       {"en": "Comma-separated Linear layer names to train",
                                  "ko": "학습할 Linear 레이어 이름 (쉼표 구분)"},
    "start_training":            {"en": "Start Training",                  "ko": "학습 시작"},
    "stop":                      {"en": "Stop",                            "ko": "중지"},
    "training_log":              {"en": "Training Log",                    "ko": "학습 로그"},
    "ready":                     {"en": "Ready",                           "ko": "대기 중"},

    # Train tab — accordions
    "training":                  {"en": "Training",                        "ko": "학습"},
    "lora_architecture":         {"en": "LoRA Architecture",               "ko": "LoRA 구조"},
    "advanced":                  {"en": "Advanced",                        "ko": "고급"},
    "caption_dropout":           {"en": "Caption Dropout",                 "ko": "캡션 드롭아웃"},
    "noise_offset":              {"en": "Noise Offset",                    "ko": "노이즈 오프셋"},
    "diff_guidance":             {"en": "Differential Guidance",           "ko": "디퍼런셜 가이던스"},
    "diff_guidance_info":        {"en": "0=off, 3.0=ostris default",      "ko": "0=끄기, 3.0=ostris 기본값"},
    "module_dropout":            {"en": "Module Dropout",                  "ko": "모듈 드롭아웃"},
    "rank_dropout":              {"en": "Rank Dropout",                    "ko": "랭크 드롭아웃"},
    "timestep_sampling":         {"en": "Timestep Sampling",               "ko": "타임스텝 샘플링"},
    "timestep_sampling_info":    {"en": "sigmoid=focus on middle timesteps (recommended)",
                                  "ko": "sigmoid=중간 타임스텝 집중 (권장)"},
    "prefix_filter":             {"en": "Prefix Filter",                   "ko": "프리픽스 필터"},
    "prefix_filter_info":        {"en": "layers.=main blocks only (recommended), empty=all",
                                  "ko": "layers.=메인 블록만 (권장), 빈값=전체"},

    # Sample prompts
    "sample_1":                  {"en": "Sample 1",                        "ko": "샘플 1"},
    "sample_2":                  {"en": "Sample 2",                        "ko": "샘플 2"},
    "sample_3":                  {"en": "Sample 3",                        "ko": "샘플 3"},

    # Settings tab — general
    "model_settings":            {"en": "Model Settings",                  "ko": "모델 설정"},
    "model_directory":           {"en": "Model Directory",                 "ko": "모델 디렉토리"},
    "apply":                     {"en": "Apply",                           "ko": "적용"},
    "status":                    {"en": "Status",                          "ko": "상태"},
    "check_models":              {"en": "Check Models",                    "ko": "모델 확인"},
    "model_status":              {"en": "Model Status",                    "ko": "모델 상태"},
    "language":                  {"en": "Language",                        "ko": "언어"},

    # Settings tab — LoRA download
    "lora_download":             {"en": "LoRA Download",                   "ko": "LoRA 다운로드"},
    "lora_download_url":         {"en": "URL / HuggingFace Repo ID / CivitAI URL",
                                  "ko": "URL / HuggingFace 레포 ID / CivitAI URL"},
    "filename_in_repo":          {"en": "Filename in Repo",                "ko": "레포 내 파일명"},
    "dl_fname_placeholder":      {"en": "e.g. model.safetensors (HuggingFace only)",
                                  "ko": "예: model.safetensors (HuggingFace 전용)"},
    "save_as":                   {"en": "Save As (optional)",              "ko": "저장 이름 (선택)"},
    "recommend_scale":           {"en": "Recommend Scale",                 "ko": "추천 강도"},
    "recommend_scale_info":      {"en": "LoRA select default strength",    "ko": "LoRA 선택 시 기본 강도"},
    "civitai_api_key":           {"en": "CivitAI API Key (CivitAI only)",  "ko": "CivitAI API 키 (CivitAI 전용)"},
    "civitai_key_placeholder":   {"en": "Required for CivitAI downloads",  "ko": "CivitAI 다운로드에 필요"},
    "dl_trigger_placeholder":    {"en": "e.g. lya, lee young-ae (CivitAI: auto-filled)",
                                  "ko": "예: lya, lee young-ae (CivitAI: 자동 입력)"},
    "download":                  {"en": "Download",                        "ko": "다운로드"},

    # Settings tab — LoRA upload
    "lora_upload":               {"en": "LoRA Upload",                     "ko": "LoRA 업로드"},
    "upload_safetensors":        {"en": "Upload .safetensors file",        "ko": ".safetensors 파일 업로드"},

    # Settings tab — installed LoRAs
    "installed_loras":           {"en": "Installed LoRAs",                 "ko": "설치된 LoRA"},
    "filename":                  {"en": "Filename",                        "ko": "파일명"},
    "size":                      {"en": "Size",                            "ko": "크기"},
    "alpha":                     {"en": "Alpha",                           "ko": "알파"},
    "selected":                  {"en": "Selected",                        "ko": "선택됨"},
    "delete":                    {"en": "Delete",                          "ko": "삭제"},
    "refresh":                   {"en": "Refresh",                         "ko": "새로고침"},

    # History tab
    "generation_history":        {"en": "Generation History",              "ko": "생성 히스토리"},
    "download_all":              {"en": "Download All",                    "ko": "일괄 다운로드"},
    "delete_selected":           {"en": "Delete Selected",                 "ko": "선택 삭제"},
    "delete_all":                {"en": "Delete All",                      "ko": "전체 삭제"},
    "clear_cache":               {"en": "Clear Cache",                     "ko": "캐시 정리"},
    "generated_images":          {"en": "Generated Images",                "ko": "생성 이미지 목록"},
    "selected_file":             {"en": "Selected File",                   "ko": "선택된 파일"},
    "file_info":                 {"en": "File Info",                       "ko": "파일 정보"},
    "confirm_delete_all":        {"en": "Delete ALL generated images?",    "ko": "생성된 이미지를 모두 삭제하시겠습니까?"},

    # Kill button
    "kill":                      {"en": "Kill (emergency stop)",           "ko": "중지 (긴급 정지)"},

    "parameters":                {"en": "Parameters",                      "ko": "파라미터"},

    # Inpaint — Mask & Denoise
    "mask_denoise":              {"en": "Mask & Denoise",                  "ko": "마스크 & 디노이즈"},
    "cn_step_cutoff":            {"en": "CN Step Cutoff",                  "ko": "CN Step Cutoff"},
    "cn_step_cutoff_info":       {"en": "CN applies for this fraction of steps (0.5=first half)",
                                  "ko": "CN이 적용되는 스텝 비율 (0.5=전반부)"},
    "denoise_strength":          {"en": "Denoise Strength",                "ko": "Denoise Strength"},
    "denoise_strength_info":     {"en": "1.0=full regenerate, lower=preserve more original",
                                  "ko": "1.0=완전 재생성, 낮을수록 원본 유지"},
    "mask_grow":                 {"en": "Mask Grow (px)",                  "ko": "Mask Grow (px)"},
    "mask_blur_radius":          {"en": "Mask Blur Radius",                "ko": "Mask Blur Radius"},
    "crop_stitch":               {"en": "Crop & Stitch",                   "ko": "Crop & Stitch"},
    "crop_stitch_info":          {"en": "Crop mask region, inpaint at optimal res, stitch back (community standard)",
                                  "ko": "마스크 영역 크롭, 최적 해상도 인페인트, 다시 합성 (커뮤니티 표준)"},
    "outpaint_mask_grow":        {"en": "Outpaint Mask Grow (px)",         "ko": "Outpaint Mask Grow (px)"},
    "outpaint_mask_grow_info":   {"en": "GrowMaskWithBlur expand",         "ko": "GrowMaskWithBlur 확장"},
    "outpaint_mask_blur":        {"en": "Outpaint Mask Blur",              "ko": "Outpaint Mask Blur"},
    "outpaint_mask_blur_info":   {"en": "GrowMaskWithBlur blur",           "ko": "GrowMaskWithBlur 블러"},

    # Inpaint — Describe & Guide
    "describe_image":            {"en": "Describe Image (AI)",             "ko": "이미지 설명 (AI)"},
    "neg_prompt_warning":        {"en": "Only works when guidance > 1. Doubles generation time, may reduce quality",
                                  "ko": "guidance > 1일 때만 작동. 생성 시간 2배, 품질 저하 주의"},
    "inpaint_guide":             {"en": "**Inpaint Guide** | Mask: paint generously (small mask = new element won't appear) | "
                                        "Control Scale 0.7 default, lower to 0.3\\~0.5 if new element is hard to add | "
                                        "Step Cutoff 0.5\\~0.7",
                                  "ko": "**Inpaint Guide** | Mask: 넉넉하게 칠할 것 (작으면 새 요소 안 생김) | "
                                        "Control Scale 0.7 기본, 새 요소 추가 어려우면 0.3\\~0.5로 낮출 것 | "
                                        "Step Cutoff 0.5\\~0.7"},
    "outpaint_guide":            {"en": "**Param Guide** | Textured BG (grass/indoor/night): defaults OK | "
                                        "Mixed BG (desert/landscape+sky): Control Scale=0.3\\~0.5, Step Cutoff=0.5 | "
                                        "Uniform BG (sky/ocean): Control Scale=0.5, Step Cutoff=0.5, Denoise=0.85",
                                  "ko": "**Param Guide** | 텍스처 배경 (잔디/실내/야경): 기본값 OK | "
                                        "혼합 배경 (사막/풍경+하늘): Control Scale=0.3\\~0.5, Step Cutoff=0.5 | "
                                        "균일 배경 (하늘/바다): Control Scale=0.5, Step Cutoff=0.5, Denoise=0.85"},

    # History — pagination
    "first_page":                {"en": "<<",                              "ko": "<<"},
    "prev_3":                    {"en": "-3",                              "ko": "-3"},
    "prev_page":                 {"en": "<",                               "ko": "<"},
    "next_page":                 {"en": ">",                               "ko": ">"},
    "next_3":                    {"en": "+3",                              "ko": "+3"},
    "last_page":                 {"en": ">>",                              "ko": ">>"},
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

    # Markdown blocks: elem_id → {lang: html}
    # These are translated by replacing innerHTML instead of text node matching
    md_blocks = {}
    md_keys = {"inpaint_guide": "inpaint-guide", "outpaint_guide": "outpaint-guide"}
    for str_key, elem_id in md_keys.items():
        if str_key in STRINGS:
            texts = STRINGS[str_key]
            import re
            block = {}
            for lang, raw in texts.items():
                # Convert markdown bold to HTML
                html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', raw)
                html = html.replace('\\~', '~')
                block[lang] = html
            md_blocks[elem_id] = block

    lang_strings_json = json.dumps(lang_strings, ensure_ascii=False)
    all_to_en_json = json.dumps(all_to_en, ensure_ascii=False)
    languages_json = json.dumps(LANGUAGES, ensure_ascii=False)
    md_blocks_json = json.dumps(md_blocks, ensure_ascii=False)

    return """
(function() {
    const LANG_STRINGS = """ + lang_strings_json + """;
    const ALL_TO_EN = """ + all_to_en_json + """;
    const LANGUAGES = """ + languages_json + """;
    const MD_BLOCKS = """ + md_blocks_json + """;

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

    const SKIP_SELECTORS = '.image_editor, .image-editor, .pixi-canvas, canvas';

    function translateNodes(lang, root) {
        if (root.closest && root.closest(SKIP_SELECTORS)) return;
        const targetMap = LANG_STRINGS[lang] || {};

        root.querySelectorAll(
            'label, button, span, h1, h2, h3, h4, p, em'
        ).forEach(el => {
            if (el.closest(SKIP_SELECTORS)) return;
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

    function translateMdBlocks(lang) {
        for (const [elemId, texts] of Object.entries(MD_BLOCKS)) {
            const el = document.getElementById(elemId);
            if (!el) continue;
            const span = el.querySelector('.md');
            if (!span) continue;
            const html = texts[lang] || texts['en'] || '';
            if (html) span.innerHTML = '<p>' + html + '</p>';
        }
    }

    function translateDOM(lang) {
        translateNodes(lang, document);
        translateMdBlocks(lang);
    }

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
            if (m.target.closest && m.target.closest(SKIP_SELECTORS)) continue;
            for (const node of m.addedNodes) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    if (node.closest && node.closest(SKIP_SELECTORS)) continue;
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
