<script>
  import { onMount, onDestroy } from 'svelte'
  import { api } from './api.js'
  import ResultPagination from './ResultPagination.svelte'
  import { sogniPromptPresets } from './sogniPromptPresets.js'
  import MaskEditor from './MaskEditor.svelte'
  import CannyEditor from './CannyEditor.svelte'
  import ImageModal from './ImageModal.svelte'
  import VideoModal from './VideoModal.svelte'
  import SubtitleModal from './SubtitleModal.svelte'
  import PromptModal from './PromptModal.svelte'
  import LoraStudio from './LoraStudio.svelte'
  import PromptComposer from './PromptComposer.svelte'
  import PromptExamplesModal from './PromptExamplesModal.svelte'
  import RecentImagePicker from './RecentImagePicker.svelte'
  import PresetImagePicker from './PresetImagePicker.svelte'
  import { lockModalScroll } from './modalScroll.js'

  let tab = 'image'
  let config = null
  let settings = null
  let savedMessage = ''
  let settingsSection = 'connection'
  let jobs = []
  let engineStates = { image: 'offline', speech: 'offline', recognition: 'offline', video: 'offline', prompt: 'offline', media: 'offline', trainer: 'offline', upscale: 'offline' }
  let busy = false
  let error = ''
  let refs = []
  let imageForm = { prompt: '', width: 1024, height: 1024, seed: -1, mode: 'create' }
  let imageEnhanceEnabled = true
  let imageEnhancedPrompt = ''
  let imageEnhancedSource = ''
  let imageDisabledMessage = ''
  let imageEnhancementIsActive = false
  let imageEnhancementIsCurrent = false
  let imageResolutionMode = 'smart'
  let imageAspectRatio = '1:1'
  let imageMegapixels = 1
  let upscalingImageJob = ''
  let detailEnhancingImageJob = ''
  let imageCloneMessage = ''
  let cloningImageJob = ''
  let filterPromptPreset = ''
  let promptExamplesOpen = false
  let kreaModules = { identity: false, depth: false, style: false, userLora: false, vision: false, styleReference: false, nk2e: false, anypaint: false }
  let kreaIdentityImage = null
  let kreaIdentityReference = null
  let kreaDepthImage = null
  let kreaNK2EImage = null
  let kreaAnyPaintImage = null
  let kreaAnyPaintMask = null
  let kreaIdentityMask = null
  let kreaStrictMask = null
  let kreaIdentityPreview = ''
  let kreaIdentityReferencePreview = ''
  let kreaDepthPreview = ''
  let kreaNK2EPreview = ''
  let kreaAnyPaintPreview = ''
  let kreaAnyPaintMaskPreview = ''
  let kreaIdentityMaskPreview = ''
  let kreaStrictMaskPreview = ''
  let maskEditorMode = ''
  let cannyEditorOpen = false
  let kreaNK2EPreprocessed = false
  let parentImageJobID = ''
  let identityPreserve = 'identity, facial features, hair, body proportions, and all areas not explicitly changed'
  let identityPreset = ''
  let imageModal = null
  let videoModal = null
  let subtitleModal = null
  let promptModal = null
  let runtimeInfoOpen = false
  let releaseRuntimeInfoScroll = null
  let featureModulesOpen = false
  let recentImagePickerTarget = ''
  let presetImagePickerTarget = ''
  let depthPoseID = ''
  let nk2ePoseID = ''
  let releaseFeatureModulesScroll = null
  let activeKreaModuleLabels = []
  let kreaModuleMessage = ''

  $: {
    if (runtimeInfoOpen && !releaseRuntimeInfoScroll) releaseRuntimeInfoScroll = lockModalScroll()
    else if (!runtimeInfoOpen && releaseRuntimeInfoScroll) {
      releaseRuntimeInfoScroll()
      releaseRuntimeInfoScroll = null
    }
  }

  $: {
    if (featureModulesOpen && !releaseFeatureModulesScroll) releaseFeatureModulesScroll = lockModalScroll()
    else if (!featureModulesOpen && releaseFeatureModulesScroll) {
      releaseFeatureModulesScroll()
      releaseFeatureModulesScroll = null
    }
  }

  onDestroy(() => {
    releaseRuntimeInfoScroll?.()
    releaseFeatureModulesScroll?.()
  })
  let kreaVisionImages = []
  let kreaStyleReferenceImages = []
  let kreaStyleSelections = [{ name: 'retroanime', strength: 1 }]
  let userLoraCatalog = []
  let userLoraSelections = []
  let kreaOptions = {
    identity_strength: 1, ref_boost: 4, grounding_px: 768, steps: 8,
    sampling_preset: 'default',
    depth_strength: 0.8,
    vision_mode: 'descriptor', vision_megapixels: 1, style_reference_strength: 1,
    nk2e_mode: 'edit', nk2e_strength: 0.7, vae_mode: 'default', identity_fit_mode: 'fit',
    strict_mask_grow: 0, strict_mask_feather: 0,
    outpaint_left: 0, outpaint_top: 0, outpaint_right: 0, outpaint_bottom: 0,
    anypaint_strength: 1, anypaint_boundary_redraw_px: 32,
    filter_mode: 'balanced', filter_strength: 1,
    prompt_enhancer: false, prompt_enhancer_strength: 1, prompt_text_scale: 1.75
  }
  let speechForm = { text: '', instructions: '', language: 'Korean', speaker: 'Sohee', seed: -1 }
  let recognitionForm = {
    source: 'file', url: '', language: 'Auto', context: '',
    output_formats: ['srt', 'txt'], translation_mode: 'none', target_language: 'Korean',
    media_part: '', media_source: ''
  }
  let recognitionFile = null
  let recognitionOptions = null
  let loadingRecognitionOptions = false
  let videoForm = { prompt: '', width: 768, height: 512, fps: 24, seed: -1, image_strength: 1 }
  let videoDurationSeconds = 5
  let settingsVideoDurationSeconds = 5
  let videoImage = null
  let videoEnhanceEnabled = true
  let videoEnhancedPrompt = ''
  let videoEnhancedSource = ''
  let videoEnhancedImageKey = ''
  let videoEnhancementIsActive = false
  let videoEnhancementIsCurrent = false
  let enhancingPrompt = false
  let deletingJob = ''
  let cancellingJob = ''
  let retryingJob = ''
  let storage = null
  let cleaningStorage = false
  let subtitleView = 'gallery'
  let imageView = 'gallery'
  let mobileImagePane = 'create'
  let mobileVideoPane = 'create'
  let mobileSpeechPane = 'create'
  let mobileRecognitionPane = 'create'
  let videoView = 'gallery'
  let refreshSequence = 0
  let progressClock = Date.now()
  const pageSizeOptions = [8, 10, 20, 50, 100]
  const imagePageSizeOptions = [8, 10, 12, 16, 20, 24, 28, 50, 100]
  let listPageSizes = { image: 8, video: 8, speech: 10, recognition: 10, history: 20 }
  let listPages = { image: 1, video: 1, speech: 1, recognition: 1, history: 1 }
  let listSortOrders = { image: 'desc', video: 'desc', speech: 'desc', recognition: 'desc', history: 'desc' }
  let mobileEngineOpen = false

  const engineMeta = {
    video: ['video', 'LTX'],
    speech: ['speech', 'TTS'],
    recognition: ['media', 'Media'],
    lora: ['trainer', 'LoRA Trainer']
  }
  const engineStatusCatalog = [
    ['image_create', 'Krea 2 이미지'], ['video', 'LTX 영상'], ['speech', 'Qwen3 TTS'],
    ['recognition', 'Qwen3 ASR'], ['prompt', 'Gemma 프롬프트'], ['upscale', 'SeedVR2 고화질'],
    ['media', '미디어·FFmpeg'], ['trainer', 'Krea 2 LoRA 학습']
  ]
  let monitoredEngineStatuses = []
  let engineAggregate = 'down'
  let engineAggregateLabel = 'API 확인 중'
  let systemUsage = { cpu_percent: null, gpu_percent: null, mem_percent: null, mem_used_gb: null, mem_total_gb: null }
  $: monitoredEngineStatuses = engineStatusCatalog.map(([key, label]) => ({ key, label, online: engineStates[key] === 'online' }))
  $: {
    const onlineCount = monitoredEngineStatuses.filter((item) => item.online).length
    engineAggregate = onlineCount === monitoredEngineStatuses.length ? 'healthy' : onlineCount === 0 ? 'down' : 'degraded'
    engineAggregateLabel = engineAggregate === 'healthy' ? '전체 정상' : engineAggregate === 'degraded' ? '일부 장애' : '전체 장애'
  }
  const imageModeMeta = {
    create: { label: 'Krea 2 Turbo', short: '생성·고급', engine: 'image_create', help: '새 이미지 생성과 Identity·Depth·LoRA·부분 수정 등의 기능을 조합합니다.' },
    edit: { label: 'FLUX.2 Klein 4B', short: '원본 수정', engine: 'image_edit', help: '하나 이상의 참조 이미지를 바탕으로 내용과 스타일을 변경합니다.' },
    detail_enhance: { label: '디테일 재해석', short: 'Krea Detail', engine: 'image_create', help: 'Ostris Edit LoRA로 원본을 다시 그려 세부 묘사를 강화합니다.' },
    upscale: { label: '고화질', short: 'SeedVR2', engine: 'upscale', help: '완성된 이미지를 SeedVR2로 복원하고 확대합니다.' }
  }
  const imageModeChoices = ['create']
  const kreaModuleLabels = {
    identity: '원본 수정', depth: '자세·구도', nk2e: '실험 편집·윤곽', anypaint: '부분 수정·확장',
    style: '스타일 LoRA', userLora: '사용자 LoRA', styleReference: '스타일 이미지 참조', vision: '내용·구도 참조'
  }
  const identityPresetUI = {
    '': { primary: '편집할 원본', primaryHint: '변경할 인물이나 장면', secondary: '보조 참조', secondaryHint: '얼굴·인물·의상·사물 제공', showSecondary: true, guide: '메인 프롬프트에 바꿀 내용을 직접 입력하세요.' },
    restage: { primary: '인물 원본', primaryHint: '다른 장면에 배치할 인물', showSecondary: false, guide: '메인 프롬프트에 새로운 자세와 장면을 입력하세요.' },
    sheet: { primary: '인물 원본', primaryHint: '시트로 만들 인물', secondary: '추가 외형 참조', secondaryHint: '다른 각도나 복장 자료 · 선택 사항', showSecondary: true, guide: '같은 인물의 2×2 시트를 자동으로 구성합니다.' },
    faceSwap: { primary: '편집할 원본', primaryHint: '몸·장면을 유지할 이미지', secondary: '가져올 얼굴', secondaryHint: '교체할 얼굴이 선명한 이미지', secondaryRequired: true, showSecondary: true, guide: '첫 이미지의 얼굴만 두 번째 이미지의 얼굴로 교체합니다.' },
    headSwap: { primary: '편집할 원본', primaryHint: '몸·장면을 유지할 이미지', secondary: '가져올 머리·인물', secondaryHint: '얼굴과 헤어를 함께 가져올 이미지', secondaryRequired: true, showSecondary: true, guide: '첫 이미지의 머리 전체를 두 번째 이미지 기준으로 교체합니다.' },
    personSwap: { primary: '배경·장면 원본', primaryHint: '배경과 구도를 유지할 이미지', secondary: '가져올 인물', secondaryHint: '장면에 넣을 인물 이미지', secondaryRequired: true, showSecondary: true, guide: '첫 이미지의 장면에 두 번째 이미지의 인물을 배치합니다.' },
    tryon: { primary: '편집할 인물 원본', primaryHint: '옷을 바꿀 인물 이미지', secondary: '참고할 의상', secondaryHint: '입힐 옷이나 착장 이미지', secondaryRequired: true, showSecondary: true, guide: '두 번째 이미지의 의상을 참고해 첫 인물의 옷을 변경합니다.' },
    replace: { primary: '편집할 원본', primaryHint: '일부를 교체할 이미지', secondary: '교체 요소 참조', secondaryHint: '새로 넣을 사물·소재 · 선택 사항', showSecondary: true, guide: '메인 프롬프트와 변경 허용 영역으로 교체할 부분을 지정하세요.' }
  }
  $: identityUI = identityPresetUI[identityPreset] || identityPresetUI['']
  const imageAspectRatios = [
    ['1:1', 1, '정사각'], ['3:4', 3 / 4, '세로'], ['4:3', 4 / 3, '가로'],
    ['2:3', 2 / 3, '세로 사진'], ['3:2', 3 / 2, '가로 사진'], ['9:16', 9 / 16, '세로 화면'], ['16:9', 16 / 9, '가로 화면']
  ]
  const outputLabels = { srt: 'SRT', vtt: 'VTT', timestamped_txt: '타임코드 TXT', txt: '일반 TXT' }
  const kindLabels = { image: '이미지', video: '영상', speech: '음성', recognition: '자막' }
  const statusLabels = { queued: '대기 중', running: '처리 중', completed: '완료', failed: '실패', cancelled: '중지됨' }
  const languageCodes = { Korean: 'ko', Japanese: 'ja', English: 'en', Chinese: 'zh' }
  const translationLanguages = [
    'Korean', 'Japanese', 'English', 'Chinese', 'Traditional Chinese',
    'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Russian',
    'Arabic', 'Hindi', 'Vietnamese', 'Thai', 'Indonesian', 'Turkish',
    'Dutch', 'Polish', 'Ukrainian'
  ]
  const filterPromptSource = 'https://www.sogni.ai/loras/krea2-filter-bypass-2#examples'
  const kreaPromptGuideSource = 'https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md'
  const vibePromptGuideSource = 'https://vibeart.app/blog/z-image-turbo-prompt-guide'
  const officialPromptPresets = [
    { id: 'official-rocket', label: '로켓 발사 · 극접사', source: kreaPromptGuideSource, prompt: `immense rocket launch exhaust as seen from extremely close up` },
    { id: 'official-designer-toy', label: '3D 디자이너 토이', source: kreaPromptGuideSource, prompt: `3D rendered matte black designer toy figure, stylized round anthropomorphic shape, backward black baseball cap, oversized gold-rimmed aviator sunglasses, white traditional line-art tattoos of tiger and bird on torso, black studded belt with gold buckle, smooth vinyl texture, studio lighting, solid vibrant blue background, high contrast minimal composition` },
    { id: 'official-collage', label: '빈티지 아날로그 콜라주', source: kreaPromptGuideSource, prompt: `vintage analog collage, central irregularly shaped snowy mountain range with a section featuring distinct wavy edges, structured within a 12x16 grid of square tiles, composition fragments the subject by alternating tiles with solid azure blue background squares, thin white grid lines, grainy paper texture, retro aesthetic of mid-century print, vibrant cyan and warm neutral tones, experimental layout, tactile quality, high-contrast graphic composition` },
    { id: 'official-anime-portrait', label: '애니 인물 · 기울어진 근접 구도', source: kreaPromptGuideSource, prompt: `close-up anime portrait of a young woman, large amber-brown eyes with intricate sparkling reflections, index finger delicately touching a subtle smile, messy dark blue hair with loose strands crossing her face, white and navy school uniform, bright high-key lighting, luminous shadows with cool blue undertones, detailed digital painting, dynamic tilted framing, shallow depth of field on hand` },
    { id: 'official-ocean', label: '미니멀 바다 일러스트', source: kreaPromptGuideSource, prompt: `A minimalist flat-color illustration of a person wading through expansive shallow ocean waves beneath a pale peach sky. The dark-skinned figure, wearing an orange swim cap, light blue top, and bright green shorts, steps carefully through knee-deep water. The ocean is rendered in muted mint green with delicate, thin black linework detailing the continuous ripples and gentle whitecaps. Soft pinkish-peach reflections echo the sky on the water's surface. A dark, jagged rock rests in the lower left foreground near a pale grey shoreline. The horizon features a solid purplish-blue landmass and a stylized, layered yellow and blue cloud. The high-angle wide perspective emphasizes the vast negative space of the water, utilizing a clean ligne claire drawing aesthetic with a subtle paper texture.` },
    { id: 'official-tree-dog', label: '거대한 나무와 작은 인물', source: kreaPromptGuideSource, prompt: `A tiny figure and a small white dog sit side-by-side in the deep green shadow of a massive tree on a sloping grassy hill. The enormous tree canopy dominates the upper composition, textured with thousands of stippled, light blue and yellow dabs representing leaves. A sharp diagonal line divides the vibrant, sunlit yellow-green grass in the foreground from the dark shade sheltering the pair. The stylized, painterly landscape features flattened perspective, visible brushstrokes, and intense color contrast.` },
    { id: 'official-flowers', label: '인물 사진 · 붉은 배경과 꽃', source: kreaPromptGuideSource, prompt: `A close-up portrait of a young East Asian woman with straight black hair, loose strands sweeping across her fair skin, and an intense gaze. She wears a light grey collared shirt with a black tie. A vibrant bouquet of pink and orange lilies with lush green leaves sits in the blurred right foreground. The background is a solid, striking crimson red. Soft, directional studio lighting highlights her facial features, creating a high-contrast composition with a shallow depth of field.` },
    { id: 'official-mouse', label: '야생동물 매크로 사진', source: kreaPromptGuideSource, prompt: `A tiny, russet-brown harvest mouse clings to a slender diagonal branch amid vibrant green lobed leaves and small round buds. The mouse has soft textured fur, glossy black eyes, a pink nose, fine whiskers, and delicate pink paws firmly gripping the wood. In this macro photograph, an extremely shallow depth of field sharply focuses on the animal's face. The deep green background dissolves into a smooth, creamy bokeh, illuminated by soft, diffused natural lighting that highlights the intricate details of the fur and foliage.` },
    { id: 'official-sailor', label: '활기찬 세일러 애니', source: kreaPromptGuideSource, prompt: `A dynamic digital painting of a joyful girl in a sailor uniform stretching her arms high against a solid vibrant blue background. She has short dark windblown hair, amber eyes, and a bright smile. She wears a white shirt, striped blue collar, flowing red neckerchief, and a billowing blue pleated skirt. Expressive thick brushstrokes and bold shading emphasize energetic motion.` },
    { id: 'official-coastal-road', label: '해안 도로 · 회화풍 자동차', source: kreaPromptGuideSource, prompt: `stylized digital painting of a dark convertible on a winding coastal cliff road, high-angle perspective, blocky painterly brushstrokes, golden hour sunlight hitting rocky orange terrain and green vegetation, flock of white abstract birds flying in foreground, blinding bright sun reflection on vast ocean, vibrant warm color palette, sharp graphic shadows` },
    { id: 'official-guardian', label: '거대 수호자 · 로우 앵글', source: kreaPromptGuideSource, prompt: `An extreme low-angle close-up captures a colossal, weathered stone and bronze guardian towering in a dark, cavernous ruin. The foreground is dominated by a massive circular shield, deeply engraved with intricate spiral motifs, geometric borders, and a central star emblem. To the right, a massive gauntlet grips a textured staff. Cinematic shafts of light pierce the dusty gloom, highlighting the rough, aged textures of the ancient armor while the background fades into deep shadows through a shallow depth of field.` },
    { id: 'official-jungle', label: '초현실 정글 일러스트', source: kreaPromptGuideSource, prompt: `A stylized jungle illustration densely packed with oversized flora and surreal characters, rendered with smooth geometric shapes and granular stippled shading. Two pale figures with flowing, star-speckled black hair navigate the lush environment in blue garments. On the left, a figure grasps a vine as a white, long-beaked bird perches on their outstretched hand. On the right, the second figure reclines beside a sleek, pinkish-orange fox. The dense surroundings feature sweeping green stalks and colossal blooms in brilliant golden yellow, coral pink, and deep red. A second white bird emerges from the lower foliage. The vibrant composition forms a seamless tapestry, utilizing rich colors and volumetric grain to create a dreamlike, textured depth.` },
    { id: 'official-retro-future', label: '크롬 행성 · 레트로 퓨처', source: kreaPromptGuideSource, prompt: `A surreal retro-futuristic space scene features liquid chrome forming an abstract face merging with a glowing planetary horizon. The foreground is dominated by swirling, highly reflective metallic fluid that distorts into a stylized, melting facial profile with deep shadows and bright silver highlights. This undulating chrome form rests against the curved, atmospheric edge of a massive planet bathed in a soft electric blue and purple glow. Above the primary planet, a smaller eclipsed celestial sphere sits in the upper center, crowned by a sharp, cross-shaped starburst flare. Two additional radiant flares burst from the left and right edges of the horizon. Set against a deep black starfield, the artwork employs a vintage 1980s airbrush aesthetic with smooth gradients, ethereal lighting, and high-contrast metallic rendering.` },
    { id: 'official-gold-face', label: '금빛 리본 · 매크로 인물', source: kreaPromptGuideSource, prompt: `An extreme close-up portrait featuring pale, freckled skin and a single blue eye wrapped in reflective metallic gold ribbons. Thin gold strips crisscross diagonally over the cheek and forehead, casting sharp, hard shadows onto the face. Strands of copper hair frame the top edge while the left ear softly blurs out of focus. Harsh, direct lighting highlights intricate skin pores and bright golden reflections, isolating the brightly lit features against a pitch-black background in a bold, high-contrast macro editorial style.` },
    { id: 'official-jester', label: '광대 전사 · 다크 판타지', source: kreaPromptGuideSource, prompt: `Stylized digital painting of a menacing jester figure rendered with bold, expressive brushstrokes and a vibrant, almost psychedelic color palette against a pitch-black background. Dynamic low-angle perspective forces a dramatic, imposing composition as the character leans forward, one leg raised high. The jester wears a classic multi-pointed hat with bells, a ruffled collar, puffed sleeves, harlequin-patterned shorts in muted gold and dark brown, and striped tights in alternating shades of purple, blue, and chartreuse. A heavily textured, flowing cape billows outward to the left, decorated with abstract, fluid patterns of saturated purples, greens, and iridescent hues resembling oil slicks or marbled paper. The figure's face is completely obscured, appearing as a smooth, faceless, pale mauve mask with a single, glowing bright white point of light in the center. In its right hand, clad in a grey-blue gauntlet, the jester grips a massive, ornate sword with a wide, glowing, ethereal white blade, its crossguard intricately sculpted. Lighting is dramatic and theatrical, casting strong shadows and highlighting the painterly texture, giving the artwork a dark fantasy, surreal aesthetic reminiscent of concept art.` },
    { id: 'official-fashion-red', label: '패션 화보 · 붉은 배경', source: kreaPromptGuideSource, prompt: `high-fashion editorial portrait of a young East Asian woman, short choppy platinum blonde bob with heavy bangs, looking over her bare shoulder to the right, lips playfully pursed, wearing a structured black top with an architectural protruding bust detail and thin straps, delicate gold hoop earrings, arm bent with hand resting on hip, warm skin tones, solid striking crimson red background, soft directional studio lighting, cinematic color palette, medium close-up shot` },
    { id: 'official-ink-faces', label: '초현실 흑백 잉크화', source: kreaPromptGuideSource, prompt: `A surreal black-and-white ink illustration of three interlocking, heavily wrinkled elderly faces merging into a landscape. The top face covers one eye, crowned by dense leaves, a live bird, and a skeletal bird. It flows into a profile face and a third face featuring a solid black eye and a hand on its cheek. The bottom neck plunges into a cross-section of earth, morphing into swirling subterranean roots, buried bones, and abstract organic forms. Above ground, weathered wooden cabins and tall grass flank the facial monolith. Meticulous stippling and cross-hatching define the high-contrast, intricate vertical composition.` },
    { id: 'official-cel-crowd', label: '1990년대 셀 애니 군중', source: kreaPromptGuideSource, prompt: `1990s vintage anime style cel animation, densely packed crowd of teenagers in summer uniforms, central boy with short black hair raising a clenched right fist, squinting one eye with a determined expression, wearing a white short-sleeve shirt and solid green necktie, surrounding students looking in various directions, girls in white sailor blouses with green striped collars and neckerchiefs, light blue skirts and trousers, tightly framed medium shot, flat shading, soft muted retro.` },
    { id: 'official-wind', label: '바람 부는 애니 인물', source: kreaPromptGuideSource, prompt: `young woman looking over her right shoulder, anime-style illustration, messy black hair blowing dynamically in the wind, striking green eyes, subtle neutral expression, oversized white button-down collared shirt with soft blue shadows, vibrant deep blue sky background, bright fluffy white cumulus clouds, silhouetted utility poles with power lines, low angle portrait, cinematic sunlight, crisp cel-shaded aesthetic` },
    { id: 'official-film-face', label: '필름 그레인 · 얼굴 극접사', source: kreaPromptGuideSource, prompt: `extreme close-up of a woman's face partially obscured by tousled dark brown hair, soft parted lips, smooth skin on lower cheek and jawline, stray hair strands falling loosely across the nose, deep moody shadows enveloping the left frame, cinematic warm lighting, delicate highlights on the mouth, muted earthy color palette, sepia-toned warmth, intimate portrait photography, macro lens, shallow depth of field, distinct film grain texture, vintage atmospheric aesthetic` }
  ]
  const communityPromptPresets = [
    {
      id: 'expression', label: '표정·감정 준수', source: filterPromptSource,
      prompt: `A grainy disposable-camera full-body portrait of a slim woman with a soft oval face and extremely long black hair, standing straight with her feet apart and hands at her sides, facing the viewer. She wears translucent pink sunglasses on her head, a hot-pink buttoned blazer, and a fully covered high-neck white lace top. Her head is tilted and her expression combines intense skepticism, disgust, confusion, and defensiveness: wide fixed eyes, a tense inward brow, and an asymmetric grimace showing clenched teeth. Hard direct on-camera flash blows out facial highlights while a pale wall and sheer curtain fall into a warm blurred background, heavy analog grain, slight motion blur, raw low-fi snapshot.`
    },
    {
      id: 'horror', label: '공포·피 묘사', source: filterPromptSource,
      prompt: `A gritty high-contrast close-up of a vampire mouth, slightly open, with sharp white fangs. Deep crimson blood drips from the fang tips, pools on the lower lip, and runs down the chin. A hand with dark-painted nails touches the lips and smears the glossy blood. Pale desaturated skin, dark lipstick, colored xerox and punk-zine aesthetic, heavy noise, photocopy distortion, raw rebellious horror atmosphere.`
    },
    {
      id: 'diversity', label: '다인물·체형·행동', source: filterPromptSource,
      prompt: `An amateur cell-phone candid photo in a dim upscale disco club. Four clearly distinct adult women sit across one wide straight red couch. From left to right: a slim blonde in a red sequin outfit laughing toward her friends with crossed legs and hands on her lap; a voluptuous Black woman in a fitted green mini dress looking surprised at the camera; a chubby red-haired woman in fitted black clothes smiling with her legs apart, one hand on her thigh and the other holding a whisky glass; and a slim Asian woman in a black skirt and thin-strap blouse, smiling at the camera with crossed legs while holding a cigarette near her face. Preserve all four identities, body types, clothing colors, poses, hand-held objects, and their relative positions. Wooden wall behind the couch, dim homemade photography, provocative nightlife mood.`
    },
    {
      id: 'action', label: '액션·드라마', source: filterPromptSource,
      prompt: `A single 2D anime cel-animation frame in a gritty survival-horror style. In a derelict hospital corridor under flickering fluorescent lights, cold green-grey tones and harsh rim light cut through drifting smoke. A pale bruised woman leans against cracked tile, sweating and breathing hard while gripping a bloodied hatchet. A huge shifting shadow emerges behind her. Capture the instant she lunges and swings the hatchet in a fast sakuga arc, with controlled motion blur, smear-frame energy, flying droplets, handheld-camera urgency, and a tense mid-motion ending.`
    },
    ...sogniPromptPresets
  ]
  const vibePromptPresets = [
    { id: 'vibe-hanfu', label: '한푸 인물 · 복합 지시', category: 'portrait', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/s4t8qua55bf3piytxwktri49-735a86297089cbd8e0a68a7f6262442f.png', prompt: `Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp, bright yellow glow, floating above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda in Xi'an, blurred colorful distant lights. Photorealistic, cinematic, ultra-detailed.` },
    { id: 'vibe-skincare', label: '이중언어 화장품 포스터', category: 'graphic', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/h3d38jbc0m21y5rl25w3mx5t-ce22254b3fa2c97f5173315c04ed754b.png', prompt: `Luxury skincare poster. Frosted glass serum bottle on a cream stone pedestal, soft gold rim light, premium beauty campaign composition, highly realistic product photography. The poster contains exactly four readable text elements only: Chinese "晨光精华", English "Morning Serum", Chinese "轻盈修护", English "Light Repair". Elegant high-end typography, balanced spacing, no extra words, no logo, no watermark.` },
    { id: 'vibe-coffee', label: '이중언어 커피 패키지', category: 'photo', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/s4s75fl0u9cvenhpe99ojruv-c5d5d71e040b79e7034036615951ed41.png', prompt: `Photorealistic premium coffee bag packaging on a neutral warm-gray studio background, matte paper bag, subtle valve, realistic shadows. The front label contains only four readable text elements: Chinese "云南咖啡", English "Yunnan Coffee", Chinese "日晒处理", English "Natural Process". Accurate printed typography on the bag surface, no extra text, no logo, high-end packaging photography.` },
    { id: 'vibe-storefront', label: '이중언어 매장 간판', category: 'photo', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/or1ibt5sjhvzuhzgvcufnqmv-f055ec731b4a6e393f6456ae6a8b3978.png', prompt: `Photorealistic modern tea bar storefront at dusk, clean glass facade, warm interior lighting, elegant urban street scene. The storefront signage contains only short readable bilingual text: Chinese "山茶" and English "Mountain Tea". Menu board visible through the window contains only two short readable items: Chinese "乌龙" and English "Oolong". No other text, no logo clutter, premium branding photography.` },
    { id: 'vibe-mid-autumn', label: '문화적으로 일관된 명절 정물', category: 'photo', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/n98edrhzzbm8guoc8odj0pln-37d2d13fce2ef6b07891c9a85d1551a3.png', prompt: `Culturally coherent Mid-Autumn Festival still life in an elegant Chinese home interior: mooncakes on a porcelain plate, a small white tea set, osmanthus blossoms, rabbit paper-cut decoration, warm lantern glow, and a full moon visible through a round window. The arrangement should feel authentic, harmonious, and logically composed, with no random clutter, no text, no watermark. Photorealistic editorial photography.` },
    { id: 'vibe-seven-objects', label: '정확히 7개 · 지정 위치', category: 'photo', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/w160lbagvsshc8jzomd8xpvz-b51bd7db25de008cc5b99d609d481009.png', prompt: `Top-down studio tabletop on a charcoal surface. Exactly seven objects and nothing else: a blue notebook in the top left, silver fountain pen in the top center, black camera in the top right, green ceramic tea cup in the middle left, white earbuds case in the middle center, red passport in the middle right, and a yellow keychain centered below them. Clean shadows, precise spacing, photorealistic, no text, no logo.` }
  ]
  const filterPromptPresets = [...officialPromptPresets, ...communityPromptPresets, ...vibePromptPresets]
  const kreaStyleCatalog = [
    { name: 'darkbrush', label: 'Dark Brush', detail: '먹선 · 수묵' },
    { name: 'dotmatrix', label: 'Dot Matrix', detail: '점묘 · 망점' },
    { name: 'kidsdrawing', label: 'Kids Drawing', detail: '어린이 그림' },
    { name: 'neondrip', label: 'Neon Drip', detail: '네온 · 추상 질감' },
    { name: 'rainywindow', label: 'Rainy Window', detail: '빗물 낀 창문' },
    { name: 'retroanime', label: 'Retro Anime', detail: '보랏빛 레트로 애니' },
    { name: 'softwatercolor', label: 'Soft Watercolor', detail: '부드러운 수채화' },
    { name: 'sunsetblur', label: 'Sunset Blur', detail: '노을 · 모션 블러' },
    { name: 'vintagetarot', label: 'Vintage Tarot', detail: '빈티지 타로' }
  ]
  const recognitionLanguages = [
    ['Auto', 'Auto · 단일 언어'],
    ['AutoMultilingual', 'Auto · 다중 언어'],
    ['Korean', 'Korean'], ['English', 'English'], ['Chinese', 'Chinese'], ['Japanese', 'Japanese']
  ]

  function recognitionLanguageLabel(language) {
    return recognitionLanguages.find(([value]) => value === language)?.[1] || language
  }

  function captionLanguage(job) {
    const language = job.params?.translation_mode === 'none'
      ? job.params?.detected_language || job.params?.language
      : job.params?.target_language
    return languageCodes[language] || 'und'
  }

  function formatBytes(value) {
    const bytes = Number(value) || 0
    if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(2)} GB`
    if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`
    if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${bytes} B`
  }

  function isAudioMedia(job) {
    const media = job.params?.media || {}
    return media.media_type === 'audio' || String(media.content_type || '').startsWith('audio/')
  }

  function mediaSummary(job) {
    const media = job.params?.media
    if (!media) return ''
    const dimensions = !isAudioMedia(job) && media.width && media.height ? `${media.width}×${media.height} · ` : ''
    return `${dimensions}${formatDuration(media.duration)} · ${formatBytes(media.size)}`
  }

  function durationFromFrames(frames, fps) {
    return Math.round(Math.max(0, (Number(frames) - 1) / Math.max(1, Number(fps))) * 1000) / 1000
  }

  function framesForDuration(seconds, fps) {
    const rawFrames = Math.max(0, Number(seconds) || 0) * Math.max(1, Number(fps) || 1)
    return Math.max(9, Math.round(rawFrames / 8) * 8 + 1)
  }

  function formatDuration(seconds) {
    const total = Math.max(0, Number(seconds) || 0)
    const hours = Math.floor(total / 3600)
    const minutes = Math.floor((total % 3600) / 60)
    const secs = Math.round((total % 60) * 10) / 10
    const secondText = Number.isInteger(secs) ? String(secs).padStart(2, '0') : secs.toFixed(1).padStart(4, '0')
    if (hours) return `${hours}:${String(minutes).padStart(2, '0')}:${secondText}`
    return `${minutes}:${secondText}`
  }

  function videoJobDuration(job) {
    return (Math.max(1, Number(job.params?.num_frames) || 1) - 1) / Math.max(1, Number(job.params?.fps) || 1)
  }

  function imageModuleSummary(job) {
    const params = job.params || {}
    if (params.mode !== 'create') return ''
    const modules = []
    if (params.identity) modules.push('Identity')
    if (params.depth) modules.push('Depth')
    if (params.styles?.length || params.style) modules.push(`LoRA${params.styles?.length > 1 ? ` ×${params.styles.length}` : ''}`)
    if (params.user_loras?.length) modules.push(`사용자 LoRA${params.user_loras.length > 1 ? ` ×${params.user_loras.length}` : ''}`)
    if (params.style_reference) modules.push('Style Ref')
    if (params.vision) modules.push('Vision')
    if (params.nk2e) modules.push(params.nk2e_mode === 'canny' ? 'NK2E Canny' : 'NK2E Edit')
    if (params.anypaint) modules.push(params.anypaint_mask ? 'Inpaint' : 'Outpaint')
    return modules.length ? ` · ${modules.join(' + ')}` : ''
  }

  function imageSamplingSummary(job) {
    const params = job.params || {}
    if (!params.sampler && !params.scheduler && !params.steps) return ''
    return `${params.sampler || '—'} / ${params.scheduler || '—'} · ${params.steps || '—'} steps`
  }

  function compactElapsed(seconds) {
    const value = Math.max(0, Math.round(Number(seconds) || 0))
    if (value < 60) return `${value}초`
    const minutes = Math.floor(value / 60)
    const remainder = value % 60
    if (minutes < 60) return remainder ? `${minutes}분 ${remainder}초` : `${minutes}분`
    const hours = Math.floor(minutes / 60)
    return `${hours}시간 ${minutes % 60}분`
  }

  function imageGenerationKey(job) {
    const params = job.params || {}
    const mode = params.mode || 'create'
    const steps = Number(params.steps) || (mode === 'detail_enhance' ? 10 : 8)
    const sampler = params.sampler || (mode === 'detail_enhance' ? 'er_sde' : 'euler')
    const megapixelBand = Math.max(1, Math.round((Number(params.width) || 1024) * (Number(params.height) || 1024) / 262144))
    const modules = ['identity', 'depth', 'vision', 'style_reference', 'nk2e', 'anypaint'].filter((name) => params[name]).join('+') || 'base'
    return `${mode}|${sampler}|${steps}|${megapixelBand}|${modules}`
  }

  function imageJobDurationSeconds(job) {
    const started = Date.parse(job.params?.started_at || job.created_at || 0)
    const completed = Date.parse(job.updated_at || 0)
    if (!Number.isFinite(started) || !Number.isFinite(completed) || completed <= started) return 0
    return (completed - started) / 1000
  }

  function median(values) {
    const sorted = values.filter((value) => Number.isFinite(value) && value > 0).sort((a, b) => a - b)
    if (!sorted.length) return 0
    const middle = Math.floor(sorted.length / 2)
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2
  }

  function imageGenerationEstimateSeconds(job) {
    const key = imageGenerationKey(job)
    const exact = jobs
      .filter((item) => item.kind === 'image' && item.status === 'completed' && item.id !== job.id && imageGenerationKey(item) === key)
      .slice(0, 12)
      .map(imageJobDurationSeconds)
    const observed = median(exact)
    if (observed) return Math.max(3, observed)
    const params = job.params || {}
    const mode = params.mode || 'create'
    const megapixels = Math.max(.25, (Number(params.width) || 1024) * (Number(params.height) || 1024) / 1_000_000)
    const steps = Number(params.steps) || (mode === 'detail_enhance' ? 10 : 8)
    if (mode === 'upscale') return Math.max(20, 16 * megapixels)
    if (mode === 'detail_enhance') return Math.max(18, 5 + steps * megapixels * 1.5)
    const moduleCount = ['identity', 'depth', 'vision', 'style_reference', 'nk2e', 'anypaint'].filter((name) => params[name]).length
    return Math.max(8, 4 + steps * megapixels * 1.15 * (1 + moduleCount * .18))
  }

  function imageGenerationProgress(job) {
    const created = Date.parse(job.created_at || 0)
    if (job.status === 'queued') {
      const elapsed = Number.isFinite(created) ? (progressClock - created) / 1000 : 0
      return { label: '대기 중', percent: 2, elapsed: `대기 ${compactElapsed(elapsed)}`, eta: '앞선 작업 종료 후 시작' }
    }
    const started = Date.parse(job.params?.started_at || job.updated_at || job.created_at || 0)
    const elapsedSeconds = Number.isFinite(started) ? Math.max(0, (progressClock - started) / 1000) : 0
    const estimateSeconds = imageGenerationEstimateSeconds(job)
    const remainingSeconds = estimateSeconds - elapsedSeconds
    const percent = Math.min(94, Math.max(5, elapsedSeconds / estimateSeconds * 100))
    const finishTime = new Date(progressClock + Math.max(0, remainingSeconds) * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' })
    const timing = `${Math.round(elapsedSeconds)}/${Math.round(estimateSeconds)}초`
    return {
      label: remainingSeconds > 0 ? `${Math.round(percent)}%` : '마무리 중',
      percent,
      elapsed: timing,
      eta: remainingSeconds > 0 ? `${finishTime} 완료 예상` : ''
    }
  }

  function videoGenerationKey(job) {
    const params = job.params || {}
    return `${Number(params.width) || 0}x${Number(params.height) || 0}|${Number(params.num_frames) || 0}|${params.image ? 'i2v' : 't2v'}`
  }

  function videoGenerationWork(job) {
    const params = job.params || {}
    return Math.max(.1, (Number(params.width) || 768) * (Number(params.height) || 512) / 1_000_000) * Math.max(9, Number(params.num_frames) || 97)
  }

  function videoGenerationDurationSeconds(job) {
    const started = Date.parse(job.params?.started_at || job.created_at || 0)
    const completed = Date.parse(job.updated_at || 0)
    if (!Number.isFinite(started) || !Number.isFinite(completed) || completed <= started) return 0
    return (completed - started) / 1000
  }

  function videoGenerationEstimateSeconds(job) {
    const exact = jobs
      .filter((item) => item.kind === 'video' && item.status === 'completed' && item.id !== job.id && videoGenerationKey(item) === videoGenerationKey(job))
      .slice(0, 12)
      .map(videoGenerationDurationSeconds)
    const exactObserved = median(exact)
    if (exactObserved) return Math.max(10, exactObserved)
    const normalized = jobs
      .filter((item) => item.kind === 'video' && item.status === 'completed' && item.id !== job.id && Boolean(item.params?.image) === Boolean(job.params?.image))
      .slice(0, 20)
      .map((item) => videoGenerationDurationSeconds(item) / videoGenerationWork(item))
    const rate = median(normalized)
    if (rate) return Math.max(10, rate * videoGenerationWork(job))
    return Math.max(30, videoGenerationWork(job) * 2.5)
  }

  function videoGenerationProgress(job) {
    const created = Date.parse(job.created_at || 0)
    if (job.status === 'queued') {
      const elapsed = Number.isFinite(created) ? (progressClock - created) / 1000 : 0
      return { label: '대기 중', percent: 2, elapsed: `대기 ${compactElapsed(elapsed)}`, eta: '앞선 작업 종료 후 시작' }
    }
    const started = Date.parse(job.params?.started_at || job.updated_at || job.created_at || 0)
    const elapsedSeconds = Number.isFinite(started) ? Math.max(0, (progressClock - started) / 1000) : 0
    const estimateSeconds = videoGenerationEstimateSeconds(job)
    const remainingSeconds = estimateSeconds - elapsedSeconds
    const percent = Math.min(94, Math.max(5, elapsedSeconds / estimateSeconds * 100))
    const finishTime = new Date(progressClock + Math.max(0, remainingSeconds) * 1000).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23' })
    return {
      label: remainingSeconds > 0 ? `${Math.round(percent)}%` : '마무리 중',
      percent,
      elapsed: `${Math.round(elapsedSeconds)}/${Math.round(estimateSeconds)}초`,
      eta: remainingSeconds > 0 ? `${finishTime} 완료 예상` : ''
    }
  }

  function imagePromptModalText(job) {
    const enhanced = job.params?.enhanced_prompt || job.params?.source_enhanced_prompt
    if (!enhanced) return job.prompt || ''
    return `원문\n${job.prompt || ''}\n\n실제 생성 프롬프트\n${enhanced}`
  }

  function recognitionProgressText(job) {
    const params = job.params || {}
    if (job.status === 'cancelled') return '중지됨'
	if (job.status === 'queued') {
	  const position = recognitionQueuePosition(job)
	  return position ? `대기 ${position}번째 · 앞선 작업 완료 후 자동 시작` : '대기 중'
	}
    if (params.stage === 'media') {
      const labels = {
        starting: '미디어 준비 시작 중', resuming: '저장된 원본에서 작업 재개 중', receiving: '파일 전송 중', resolving: '영상 페이지 분석 중',
        storing: '미디어 저장·재생 형식 정리 중', extracting_audio: '음성 추출·분할 중', complete: '미디어 준비 마무리 중'
      }
      if (params.media_stage === 'downloading') {
        const percent = Number(params.media_percent) || 0
        const amount = params.media_total_bytes ? ` · ${formatBytes(params.media_downloaded_bytes)} / ${formatBytes(params.media_total_bytes)}` : ''
        const eta = params.media_eta_seconds ? ` · 약 ${params.media_eta_seconds}초 남음` : ''
        return `미디어 다운로드 ${percent.toFixed(1)}%${amount}${eta}`
      }
      return labels[params.media_stage] || '미디어 준비 중'
    }
    if (params.stage === 'recognition') return params.segments ? `음성 인식 ${params.progress || 0}/${params.segments} 구간` : '음성 인식 준비 중'
    if (params.stage === 'translation') return `자막 번역 ${params.translation_progress || 0}/${params.translation_total || 0} 배치`
    if (params.stage === 'finalizing') return '자막 파일 생성 중'
    return job.status
  }

  function recognitionProgressPercent(job) {
    const params = job.params || {}
	if (job.status === 'queued') return 2
    if (params.stage === 'media' && params.media_stage === 'downloading') return Math.min(100, Math.max(0, Number(params.media_percent) || 0))
    if (params.stage === 'recognition' && params.segments) return Math.min(100, (Number(params.progress) || 0) * 100 / params.segments)
    if (params.stage === 'translation' && params.translation_total) return Math.min(100, (Number(params.translation_progress) || 0) * 100 / params.translation_total)
    return 0
  }

  function recognitionQueuePosition(job) {
	const queued = jobs
	  .filter((item) => item.kind === 'recognition' && item.status === 'queued')
	  .sort((a, b) => {
		const left = Date.parse(a.params?.queued_at || a.created_at || 0)
		const right = Date.parse(b.params?.queued_at || b.created_at || 0)
		return left - right || String(a.id).localeCompare(String(b.id))
	  })
	const index = queued.findIndex((item) => item.id === job.id)
	return index < 0 ? 0 : index + 1
  }

  const activeJobs = () => jobs.filter((j) => j.status === 'queued' || j.status === 'running')

  function jobsForList(key) {
    return key === 'history' ? jobs : jobs.filter((job) => job.kind === key)
  }

  function pagedJobs(key) {
    const start = (listPages[key] - 1) * listPageSizes[key]
    return jobsForList(key).slice(start, start + listPageSizes[key])
  }

  let imageJobs = []
  let videoJobs = []
  let speechJobs = []
  let recognitionJobs = []
  let pagedImageJobs = []
  let pagedVideoJobs = []
  let pagedSpeechJobs = []
  let pagedRecognitionJobs = []
  let pagedHistoryJobs = []
  function orderedJobs(items, order) {
    return order === 'asc' ? [...items].reverse() : items
  }
  $: imageJobs = orderedJobs(jobs.filter((job) => job.kind === 'image'), listSortOrders.image)
  $: videoJobs = orderedJobs(jobs.filter((job) => job.kind === 'video'), listSortOrders.video)
  $: speechJobs = orderedJobs(jobs.filter((job) => job.kind === 'speech'), listSortOrders.speech)
  $: recognitionJobs = orderedJobs(jobs.filter((job) => job.kind === 'recognition'), listSortOrders.recognition)
  $: pagedImageJobs = imageJobs.slice((listPages.image - 1) * listPageSizes.image, listPages.image * listPageSizes.image)
  $: pagedVideoJobs = videoJobs.slice((listPages.video - 1) * listPageSizes.video, listPages.video * listPageSizes.video)
  $: pagedSpeechJobs = speechJobs.slice((listPages.speech - 1) * listPageSizes.speech, listPages.speech * listPageSizes.speech)
  $: pagedRecognitionJobs = recognitionJobs.slice((listPages.recognition - 1) * listPageSizes.recognition, listPages.recognition * listPageSizes.recognition)
  $: pagedHistoryJobs = orderedJobs(jobs, listSortOrders.history).slice((listPages.history - 1) * listPageSizes.history, listPages.history * listPageSizes.history)

  function clampListPages() {
    const next = { ...listPages }
    for (const key of Object.keys(next)) {
      const lastPage = Math.max(1, Math.ceil(jobsForList(key).length / listPageSizes[key]))
      next[key] = Math.min(Math.max(1, next[key]), lastPage)
    }
    listPages = next
  }

  function setListPage(key, page) {
    const lastPage = Math.max(1, Math.ceil(jobsForList(key).length / listPageSizes[key]))
    listPages = { ...listPages, [key]: Math.min(Math.max(1, page), lastPage) }
  }

  function pageSizeOptionsFor(key) {
    return ['image', 'video', 'recognition'].includes(key) ? imagePageSizeOptions : pageSizeOptions
  }

  function setListPageSize(key, pageSize) {
    const allowedSizes = pageSizeOptionsFor(key)
    const size = allowedSizes.includes(pageSize) ? pageSize : listPageSizes[key]
    listPageSizes = { ...listPageSizes, [key]: size }
    listPages = { ...listPages, [key]: 1 }
    localStorage.setItem(`media-${key}-page-size`, String(size))
  }

  function setListSortOrder(key, order) {
    const nextOrder = order === 'asc' ? 'asc' : 'desc'
    listSortOrders = { ...listSortOrders, [key]: nextOrder }
    listPages = { ...listPages, [key]: 1 }
    localStorage.setItem(`media-${key}-sort-order`, nextOrder)
  }

  function showNewestListPage(key) {
    listPages = { ...listPages, [key]: 1 }
  }

  async function refresh() {
    const sequence = ++refreshSequence
    try {
      const [nextJobs, nextEngines] = await Promise.all([api.jobs(), api.engines()])
      if (sequence !== refreshSequence) return
      jobs = [...nextJobs].sort((a, b) => {
        const createdDifference = Date.parse(b.created_at || 0) - Date.parse(a.created_at || 0)
        return createdDifference || String(b.id).localeCompare(String(a.id))
      })
      clampListPages()
      engineStates = Object.fromEntries(nextEngines.map((item) => [item.kind, item.status]))
    } catch (e) {
      if (sequence === refreshSequence) error = e.message
    }
  }

  async function refreshSystemUsage() {
    try {
      systemUsage = await api.system()
    } catch {
      systemUsage = { cpu_percent: null, gpu_percent: null, mem_percent: null, mem_used_gb: null, mem_total_gb: null }
    }
  }

  onMount(() => {
    subtitleView = localStorage.getItem('media-subtitle-view') === 'list' ? 'list' : 'gallery'
    imageView = localStorage.getItem('media-image-view') === 'list' ? 'list' : 'gallery'
    videoView = localStorage.getItem('media-video-view') === 'list' ? 'list' : 'gallery'
    for (const key of Object.keys(listPageSizes)) {
      const storedSize = Number(localStorage.getItem(`media-${key}-page-size`))
      const allowedSizes = pageSizeOptionsFor(key)
      if (allowedSizes.includes(storedSize)) listPageSizes = { ...listPageSizes, [key]: storedSize }
      const storedOrder = localStorage.getItem(`media-${key}-sort-order`)
      if (storedOrder === 'asc' || storedOrder === 'desc') listSortOrders = { ...listSortOrders, [key]: storedOrder }
    }
    api.config().then((value) => {
      config = value
      settings = structuredClone(value)
      imageForm.width = value.image.default_width
      imageForm.height = value.image.default_height
      applySmartResolution()
      imageForm.mode = imageModeChoices.includes(value.image.default_mode) ? value.image.default_mode : 'create'
      speechForm.language = value.speech.default_language
      speechForm.speaker = value.speech.default_speaker
      recognitionForm.language = value.recognition.default_language
      recognitionForm.output_formats = [...value.recognition.default_output_formats]
      recognitionForm.translation_mode = value.recognition.default_translation_mode
      recognitionForm.target_language = value.recognition.default_translation_language
      videoForm.width = value.video.default_width
      videoForm.height = value.video.default_height
      videoForm.fps = value.video.default_fps
      videoDurationSeconds = durationFromFrames(value.video.default_frames, value.video.default_fps)
      videoEnhanceEnabled = value.prompt_enhancement.default_enabled
      imageEnhanceEnabled = value.prompt_enhancement.default_enabled
      kreaOptions = { ...kreaOptions, prompt_enhancer: Boolean(value.image.default_prompt_enhancer) }
    }).catch((e) => error = e.message)
    refreshUserLoras()
    refresh()
    refreshSystemUsage()
    const timer = setInterval(refresh, 1500)
    const systemTimer = setInterval(refreshSystemUsage, 5000)
    const progressTimer = setInterval(() => { progressClock = Date.now() }, 1000)
    return () => { clearInterval(timer); clearInterval(systemTimer); clearInterval(progressTimer) }
  })

  function setSubtitleView(view) {
    subtitleView = view
    localStorage.setItem('media-subtitle-view', view)
  }

  function setImageView(view) {
    imageView = view
    localStorage.setItem('media-image-view', view)
  }

  function setVideoView(view) {
    videoView = view
    localStorage.setItem('media-video-view', view)
  }

  function addRefs(files) {
    const incoming = [...files].filter((file) => file.type.startsWith('image/')).map((file) => ({ file, name: file.name, preview: URL.createObjectURL(file), server: false }))
    const limit = imageForm.mode === 'control' ? 1 : (config?.image.max_reference_images || 4)
    const combined = [...refs, ...incoming]
    combined.slice(limit).forEach((image) => { if (image.preview?.startsWith('blob:')) URL.revokeObjectURL(image.preview) })
    refs = combined.slice(0, limit)
  }

  function clearRefs() {
    refs.forEach((image) => { if (image.preview?.startsWith('blob:')) URL.revokeObjectURL(image.preview) })
    refs = []
  }

  function removeRef(index) {
    const removed = refs[index]
    if (removed?.preview?.startsWith('blob:')) URL.revokeObjectURL(removed.preview)
    refs = refs.filter((_, itemIndex) => itemIndex !== index)
  }

  function toggleKreaModule(module) {
    kreaModules = { ...kreaModules, [module]: !kreaModules[module] }
    if (kreaModules[module]) {
      if (module === 'identity' && imageForm.width * imageForm.height > 2 * 1024 * 1024) {
        imageMegapixels = 2
        imageResolutionMode = 'smart'
        applySmartResolution()
        imageCloneMessage = 'Identity 편집은 최대 2MP이므로 이미지 크기를 고해상도 2MP로 조정했습니다.'
      }
      return
    }
    if (module === 'identity') {
      setKreaImage('identity', null)
      setKreaImage('identityReference', null)
    }
    if (module === 'depth') setKreaImage('depth', null)
    if (module === 'nk2e') setKreaImage('nk2e', null)
    if (module === 'anypaint') {
      setKreaImage('anypaint', null)
      setKreaImage('anypaintMask', null)
    }
    if (module === 'vision') clearKreaRefs('vision')
    if (module === 'styleReference') clearKreaRefs('styleReference')
  }

  async function refreshUserLoras() {
    try { userLoraCatalog = (await api.userLoras()).filter((item) => item.filename !== 'skc3vo.safetensors') } catch (_) { userLoraCatalog = [] }
    userLoraSelections = userLoraSelections.filter((selection) => userLoraCatalog.some((item) => item.filename === selection.filename))
  }

  function hasUserLora(filename) {
    return userLoraSelections.some((selection) => selection.filename === filename)
  }

  function toggleUserLora(filename) {
    if (hasUserLora(filename)) userLoraSelections = userLoraSelections.filter((selection) => selection.filename !== filename)
    else if (userLoraSelections.length < 5) {
      const lora = userLoraCatalog.find((item) => item.filename === filename)
      const recommended = Number(lora?.recommended_strength)
      userLoraSelections = [...userLoraSelections, {
        filename,
        strength: Number.isFinite(recommended) ? recommended : 1
      }]
    }
  }

  function updateUserLoraStrength(filename, strength) {
    userLoraSelections = userLoraSelections.map((selection) => selection.filename === filename ? { ...selection, strength: Number(strength) } : selection)
  }

  function userLoraLabel(filename) {
    return userLoraCatalog.find((item) => item.filename === filename)?.name || filename
  }

  function hasKreaStyle(name) {
    return kreaStyleSelections.some((style) => style.name === name)
  }

  function toggleKreaStyle(name) {
    kreaStyleSelections = hasKreaStyle(name)
      ? kreaStyleSelections.filter((style) => style.name !== name)
      : [...kreaStyleSelections, { name, strength: 1 }]
  }

  function updateKreaStyleStrength(name, strength) {
    kreaStyleSelections = kreaStyleSelections.map((style) => style.name === name ? { ...style, strength: Number(strength) } : style)
  }

  function kreaStyleLabel(name) {
    return kreaStyleCatalog.find((style) => style.name === name)?.label || name
  }

  function addKreaRefs(kind, files) {
    const incoming = [...files].filter((file) => file.type.startsWith('image/')).map((file) => ({ file, name: file.name, preview: URL.createObjectURL(file), server: false }))
    addKreaRefObjects(kind, incoming)
  }

  function addKreaRefObjects(kind, incoming) {
    const limit = kind === 'vision' ? 4 : 2
    const current = kind === 'vision' ? kreaVisionImages : kreaStyleReferenceImages
    const combined = [...current, ...incoming]
    combined.slice(limit).forEach((image) => { if (image.preview?.startsWith('blob:')) URL.revokeObjectURL(image.preview) })
    if (kind === 'vision') kreaVisionImages = combined.slice(0, limit)
    else kreaStyleReferenceImages = combined.slice(0, limit)
  }

  function clearKreaRefs(kind) {
    const images = kind === 'vision' ? kreaVisionImages : kreaStyleReferenceImages
    images.forEach((image) => { if (image.preview?.startsWith('blob:')) URL.revokeObjectURL(image.preview) })
    if (kind === 'vision') kreaVisionImages = []
    else kreaStyleReferenceImages = []
  }

  function removeKreaRef(kind, index) {
    const images = kind === 'vision' ? kreaVisionImages : kreaStyleReferenceImages
    const removed = images[index]
    if (removed?.preview?.startsWith('blob:')) URL.revokeObjectURL(removed.preview)
    if (kind === 'vision') kreaVisionImages = images.filter((_, i) => i !== index)
    else kreaStyleReferenceImages = images.filter((_, i) => i !== index)
  }

  function setKreaImage(kind, image) {
    const previewKey = `${kind}Preview`
    const previews = {
      identityPreview: kreaIdentityPreview,
      identityReferencePreview: kreaIdentityReferencePreview,
      depthPreview: kreaDepthPreview,
      nk2ePreview: kreaNK2EPreview,
      anypaintPreview: kreaAnyPaintPreview,
      anypaintMaskPreview: kreaAnyPaintMaskPreview
      ,identityMaskPreview: kreaIdentityMaskPreview
      ,strictMaskPreview: kreaStrictMaskPreview
    }
    if (previews[previewKey]?.startsWith('blob:')) URL.revokeObjectURL(previews[previewKey])
    const preview = image ? (image.server ? image.url : URL.createObjectURL(image)) : ''
    if (kind === 'identity') {
      kreaIdentityImage = image
      kreaIdentityPreview = preview
      parentImageJobID = image?.server && image.role === 'output' ? String(image.ref || '').split(':')[0] : ''
    } else if (kind === 'identityReference') {
      kreaIdentityReference = image
      kreaIdentityReferencePreview = preview
    } else if (kind === 'depth') {
      kreaDepthImage = image
      kreaDepthPreview = preview
      depthPoseID = image?.poseID || ''
    } else if (kind === 'nk2e') {
      kreaNK2EImage = image
      kreaNK2EPreview = preview
      nk2ePoseID = image?.poseID || ''
      kreaNK2EPreprocessed = Boolean(image?.preprocessed)
    } else if (kind === 'anypaint') {
      if (image && kreaAnyPaintImage !== image && kreaAnyPaintMask) setKreaImage('anypaintMask', null)
      kreaAnyPaintImage = image
      kreaAnyPaintPreview = preview
    } else if (kind === 'anypaintMask') {
      kreaAnyPaintMask = image
      kreaAnyPaintMaskPreview = preview
    } else if (kind === 'identityMask') {
      kreaIdentityMask = image
      kreaIdentityMaskPreview = preview
    } else {
      kreaStrictMask = image
      kreaStrictMaskPreview = preview
    }
  }

  function appendImageInput(form, uploadField, reuseField, image) {
    if (!image) return
    if (image.server) form.append(reuseField, image.ref)
    else form.append(uploadField, image.file || image)
  }

  function useRecentModuleImage(job) {
    if (!job?.output_url || !recentImagePickerTarget) return
    const target = recentImagePickerTarget
    const image = {
      server: true,
      ref: `${job.id}:output:0`,
      url: job.output_url,
      name: `결과 ${job.id.slice(0, 8)}.png`,
      role: 'output'
    }
    if (target === 'vision' || target === 'styleReference') addKreaRefObjects(target, [image])
    else setKreaImage(target, image)
    recentImagePickerTarget = ''
  }

  async function usePresetModuleImage(item) {
    if (!item?.url || !presetImagePickerTarget) return
    const target = presetImagePickerTarget
    try {
      const response = await fetch(item.url)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const blob = await response.blob()
      const file = new File([blob], item.filename, { type: blob.type || 'image/webp' })
      file.poseID = item.library === 'pose' ? item.id : ''
      if (target === 'vision' || target === 'styleReference') addKreaRefs(target, [file])
      else setKreaImage(target, file)
      presetImagePickerTarget = ''
    } catch (cause) {
      error = `프리셋 이미지를 불러오지 못했습니다: ${cause.message}`
    }
  }

  function showImage(event, src, title, detail = '', jobID = '') {
    event?.preventDefault()
    event?.stopPropagation()
    if (src) imageModal = { src, title, detail, jobID }
  }

  function showImageOnKey(event, src, title, detail = '') {
    if (event.key === 'Enter' || event.key === ' ') showImage(event, src, title, detail)
  }

  function showVideo(job) {
    if (!job?.output_url) return
    const details = [
      `${job.params?.width || '—'}×${job.params?.height || '—'}`,
      formatDuration(videoJobDuration(job)),
      `${job.params?.fps || '—'} fps`,
      job.params?.seed >= 0 ? `seed ${job.params.seed}` : ''
    ].filter(Boolean)
    videoModal = { src: job.output_url, title: '생성 영상', detail: details.join(' · '), prompt: job.prompt }
  }

  function showSubtitle(job) {
    if (!job || (!job.media_url && !job.params?.text && !job.outputs && !job.output_url)) return
    const outputs = job.outputs
      ? Object.entries(job.outputs).map(([kind, url]) => ({ label: outputLabels[kind] || kind, url }))
      : job.output_url ? [{ label: '결과 열기', url: job.output_url }] : []
    const details = [
      job.params?.detected_language || recognitionLanguageLabel(job.params?.language),
      job.params?.segments ? `${job.params.segments}구간` : '',
      job.params?.media ? mediaSummary(job) : ''
    ].filter(Boolean)
    subtitleModal = {
      mediaSrc: job.media_url,
      audio: isAudioMedia(job),
      captionSrc: job.caption_url,
      captionLang: captionLanguage(job),
      captionLabel: job.params?.translation_mode === 'none' ? '원문' : job.params?.target_language || '번역',
      transcript: job.params?.text,
      prompt: job.prompt,
      detail: details.join(' · '),
      outputs
    }
  }

  function usePaintedMask(file) {
    if (maskEditorMode === 'identity') setKreaImage('identityMask', file)
    else if (maskEditorMode === 'strict') setKreaImage('strictMask', file)
    else setKreaImage('anypaintMask', file)
    maskEditorMode = ''
  }

  function useCannyMap(file) {
    setKreaImage('nk2e', file)
    kreaNK2EPreprocessed = true
    cannyEditorOpen = false
  }

  function rawImagePrompt() {
    const change = imageForm.prompt.trim()
    if (!kreaModules.identity || !identityPreserve.trim()) return change
    return `Change: ${change}\nPreserve: ${identityPreserve.trim()}`
  }

  function applyIdentityPreset(value) {
    identityPreset = value
    if (!(identityPresetUI[value] || identityPresetUI['']).showSecondary) setKreaImage('identityReference', null)
    const presets = {
      restage: ['Place the same person in a new scene and pose as described', 'the same identity, facial features, hair, body proportions, and recognizable appearance'],
      sheet: ['Create a clean 2x2 character sheet on a plain background: front view upper-left, three-quarter view upper-right, left profile lower-left, and back view lower-right', 'the exact same identity, face, hairstyle, body proportions, outfit design, and color palette across all four panels'],
      tryon: ['Change the subject to wear the described outfit', 'identity, face, hair, body proportions, pose unless requested, background, and lighting'],
      replace: ['Replace only the selected object or region as described', 'identity, composition, lighting, perspective, and every unselected pixel'],
      faceSwap: ['Replace only the face of the person in Image One with the face from Image Two', 'the hairstyle, head shape, clothing, body, pose, composition, lighting, and background from Image One'],
      headSwap: ['Replace the entire head of the person in Image One with the head from Image Two', 'the clothing, body, pose, composition, lighting, and background from Image One'],
      personSwap: ['Replace the entire person in Image One with the person from Image Two', 'the pose where possible, composition, lighting, props, and complete background from Image One']
    }
    if (!presets[value]) return
    imageForm.prompt = presets[value][0]
    identityPreserve = presets[value][1]
    resetImageEnhancement()
  }

  function isPureOutpaint() {
    return kreaModules.anypaint
      && Boolean(kreaAnyPaintImage)
      && !kreaAnyPaintMask
      && ['outpaint_left', 'outpaint_top', 'outpaint_right', 'outpaint_bottom'].some((key) => Number(kreaOptions[key]) > 0)
  }

  function imageDisabledReason() {
    if (busy) return '요청을 전송하고 있습니다.'
    if (activeJobs().some((j) => j.kind === 'image' || j.kind === 'video')) return '다른 이미지 또는 영상 작업이 끝나면 시작할 수 있습니다.'
    if (!imageForm.prompt.trim() && !isPureOutpaint()) return '프롬프트를 입력하세요.'
    if (imageForm.mode === 'edit' && refs.length === 0) return '편집할 참조 이미지를 추가하세요.'
    if (imageForm.mode === 'control' && refs.length !== 1) return 'Canny 제어 이미지 1장을 추가하세요.'
    return kreaModuleDisabledReason()
  }

  function kreaModuleDisabledReason() {
    if (kreaModules.identity && !kreaIdentityImage) return `원본 수정의 ${identityUI.primary} 이미지를 선택하세요.`
    if (kreaModules.identity && identityUI.secondaryRequired && !kreaIdentityReference) return `원본 수정의 ${identityUI.secondary} 이미지를 선택하세요.`
    if (kreaModules.depth && !kreaDepthImage) return '자세·구도 모듈의 구도 참조 이미지를 선택하세요.'
    if (kreaModules.vision && kreaVisionImages.length === 0) return '내용·구도 참조 이미지를 선택하세요.'
    if (kreaModules.styleReference && kreaStyleReferenceImages.length === 0) return '스타일 참조 이미지를 선택하세요.'
    if (kreaModules.style && kreaStyleSelections.length === 0) return '적용할 스타일 LoRA를 하나 이상 선택하세요.'
    if (kreaModules.userLora && userLoraSelections.length === 0) return '적용할 사용자 LoRA를 하나 이상 선택하세요.'
    if (kreaModules.nk2e && !kreaNK2EImage) return 'NK2E 편집·윤곽 모듈의 참조 이미지를 선택하세요.'
    if (kreaModules.anypaint && !kreaAnyPaintImage) return '부분 수정·확장 모듈의 원본 이미지를 선택하세요.'
    if (kreaModules.anypaint && !kreaAnyPaintMask && !['outpaint_left', 'outpaint_top', 'outpaint_right', 'outpaint_bottom'].some((key) => Number(kreaOptions[key]) > 0)) return '수정 마스크를 선택하거나 확장할 방향을 지정하세요.'
    if (kreaModules.vision && kreaModules.identity) return '내용·구도 참조와 Identity는 아직 함께 사용할 수 없습니다.'
    if (kreaModules.styleReference && Object.entries(kreaModules).some(([name, enabled]) => name !== 'styleReference' && enabled)) return '스타일 이미지 참조는 현재 단독으로 사용하세요.'
    if (kreaModules.nk2e && Object.entries(kreaModules).some(([name, enabled]) => name !== 'nk2e' && enabled)) return 'NK2E 편집·윤곽은 현재 다른 Krea 모듈과 함께 사용할 수 없습니다.'
    if (kreaModules.anypaint && Object.entries(kreaModules).some(([name, enabled]) => name !== 'anypaint' && enabled)) return '부분 수정·확장은 현재 다른 Krea 모듈과 함께 사용할 수 없습니다.'
    return ''
  }

  function disableAllKreaModules() {
    for (const name of Object.keys(kreaModules)) {
      if (kreaModules[name]) toggleKreaModule(name)
    }
  }

  function handleFeatureModulesKeydown(event) {
    if (event.key !== 'Escape' || !featureModulesOpen) return
    if (maskEditorMode || cannyEditorOpen || imageModal || runtimeInfoOpen || recentImagePickerTarget || presetImagePickerTarget) return
    featureModulesOpen = false
  }

  function looksLikeStructuredPrompt(value = imageForm.prompt) {
    const text = value.trim()
    if (!text || (text[0] !== '{' && text[0] !== '[')) return false
    try { JSON.parse(text); return true } catch { return false }
  }

  function imageEnhancementActive(enabled = imageEnhanceEnabled, prompt = imageForm.prompt) {
    return enabled && prompt.trim() !== '' && !looksLikeStructuredPrompt(prompt)
  }

  function imageEnhancementCurrent(enhanced = imageEnhancedPrompt, source = imageEnhancedSource, current = rawImagePrompt()) {
    return enhanced.trim() !== '' && source === current
  }

  // These values are rendered in the submit controls. Keep their dependencies
  // explicit so nested form bindings immediately update the button state.
  $: imageEnhancementIsActive = imageEnhancementActive(imageEnhanceEnabled, imageForm.prompt)
  $: activeKreaModuleLabels = Object.entries(kreaModules).filter(([, enabled]) => enabled).map(([name]) => kreaModuleLabels[name])
  $: kreaModuleMessage = (
    kreaModules, identityPreset, kreaIdentityImage, kreaIdentityReference, kreaDepthImage, kreaVisionImages, kreaStyleReferenceImages,
    kreaStyleSelections, userLoraSelections, kreaNK2EImage, kreaAnyPaintImage, kreaAnyPaintMask, kreaOptions,
    kreaModuleDisabledReason()
  )
  $: imageEnhancementIsCurrent = (
    imageForm, identityPreserve, kreaModules,
    imageEnhancementCurrent(imageEnhancedPrompt, imageEnhancedSource, rawImagePrompt())
  )
  $: imageDisabledMessage = (
    busy, jobs, imageForm, refs, kreaModules, identityPreset, kreaIdentityImage, kreaIdentityReference, kreaDepthImage,
    kreaVisionImages, kreaStyleReferenceImages, kreaStyleSelections, userLoraSelections,
    kreaNK2EImage, kreaAnyPaintImage, kreaAnyPaintMask, kreaOptions,
    imageDisabledReason()
  )

  function resetImageEnhancement() {
    imageEnhancedPrompt = ''
    imageEnhancedSource = ''
  }

  async function enhanceImagePrompt() {
    const original = rawImagePrompt()
    if (!original || looksLikeStructuredPrompt(original)) return
    enhancingPrompt = true; error = ''
    try {
      const form = new FormData()
      form.append('prompt', original)
      form.append('mode', kreaModules.identity ? 'edit' : kreaModules.anypaint ? 'paint' : (kreaModules.depth || kreaModules.nk2e) ? 'control' : 't2i')
      const result = await api.enhancePrompt(form)
      imageEnhancedPrompt = result.enhanced_prompt
      imageEnhancedSource = original
    } catch (e) { error = e.message }
    finally { enhancingPrompt = false }
  }

  function applySmartResolution() {
    if (imageResolutionMode !== 'smart') return
    const ratio = imageAspectRatios.find((item) => item[0] === imageAspectRatio)?.[1] || 1
    // Treat the familiar "1MP" preset as the native 1024x1024 class used by
    // current image models, then preserve the requested aspect ratio.
    const pixels = Number(imageMegapixels) * 1024 * 1024
    const width = Math.sqrt(pixels * ratio)
    const height = width / ratio
    imageForm.width = Math.min(2048, Math.max(256, Math.floor(width / 16) * 16))
    imageForm.height = Math.min(2048, Math.max(256, Math.floor(height / 16) * 16))
  }

  function useCustomImageResolution() {
    imageResolutionMode = 'custom'
  }

  function cloneImagePrompt(job) {
    filterPromptPreset = ''
    imageForm.prompt = job.prompt || ''
    resetImageEnhancement()
  }

  function applyPromptExample(preset, mode) {
    if (!preset) return
    const currentPrompt = imageForm.prompt.trimEnd()
    filterPromptPreset = preset.id
    imageForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${preset.prompt}` : preset.prompt
    promptExamplesOpen = false
    resetImageEnhancement()
  }

  function filterModeDefault(mode) {
    if (mode === 'adherence') return 0.05
    if (mode === 'balanced' || mode === 'strong') return 1
    return 0
  }

  function filterModeMaximum(mode) {
    return mode === 'adherence' ? 0.2 : 2
  }

  function cloneImageSettings(job) {
    const params = job.params || {}
    const legacyKlein = params.mode === 'edit'
    const storedStyles = Array.isArray(params.styles) && params.styles.length
      ? params.styles.map((style) => ({ name: style.name, strength: Number(style.strength) }))
      : (params.style ? [{ name: params.style, strength: params.style_strength !== undefined ? Number(params.style_strength) : 1 }] : [])
    const storedUserLoras = Array.isArray(params.user_loras) ? params.user_loras.filter((selection) => selection.filename !== 'skc3vo.safetensors').map((selection) => ({ filename: selection.filename, strength: Number(selection.strength) })) : []
    imageForm.mode = imageModeChoices.includes(params.mode) ? params.mode : 'create'
    imageForm.width = Number(params.width) || imageForm.width
    imageForm.height = Number(params.height) || imageForm.height
    imageResolutionMode = 'custom'
    imageForm.seed = Number.isFinite(Number(params.seed)) ? Number(params.seed) : -1
    if (imageForm.mode === 'create') {
      kreaModules = {
        identity: legacyKlein || Boolean(params.identity),
        depth: Boolean(params.depth),
        style: storedStyles.length > 0,
        userLora: storedUserLoras.length > 0,
        vision: Boolean(params.vision),
        styleReference: Boolean(params.style_reference),
        nk2e: Boolean(params.nk2e),
        anypaint: Boolean(params.anypaint)
      }
      kreaOptions = {
        ...kreaOptions,
        identity_strength: params.identity_strength !== undefined ? Number(params.identity_strength) : 1,
        ref_boost: params.ref_boost !== undefined ? Number(params.ref_boost) : 4,
        grounding_px: Number(params.grounding_px) || 768,
        steps: Number(params.steps) || (params.identity ? 10 : 8),
        depth_strength: params.depth_strength !== undefined ? Number(params.depth_strength) : 0.8
        ,vision_mode: params.vision_mode || 'descriptor'
        ,vision_megapixels: params.vision_megapixels !== undefined ? Number(params.vision_megapixels) : 1
        ,style_reference_strength: params.style_reference_strength !== undefined ? Number(params.style_reference_strength) : 1
        ,nk2e_mode: params.nk2e_mode || 'edit'
        ,nk2e_strength: params.nk2e_strength !== undefined ? Number(params.nk2e_strength) : 0.7
        ,vae_mode: params.vae_mode || 'default'
        ,identity_fit_mode: params.identity_fit_mode || 'fit'
        ,strict_mask_grow: Number(params.strict_mask_grow) || 0
        ,strict_mask_feather: Number(params.strict_mask_feather) || 0
        ,outpaint_left: Number(params.outpaint_left) || 0
        ,outpaint_top: Number(params.outpaint_top) || 0
        ,outpaint_right: Number(params.outpaint_right) || 0
        ,outpaint_bottom: Number(params.outpaint_bottom) || 0
        ,anypaint_strength: params.anypaint_strength !== undefined ? Number(params.anypaint_strength) : 1
        ,anypaint_boundary_redraw_px: params.anypaint_boundary_redraw_px !== undefined ? Number(params.anypaint_boundary_redraw_px) : 32
        ,filter_mode: params.filter_mode || 'balanced'
        ,filter_strength: params.filter_strength !== undefined ? Number(params.filter_strength) : filterModeDefault(params.filter_mode || 'balanced')
        ,prompt_enhancer: Boolean(params.prompt_enhancer)
        ,prompt_enhancer_strength: params.prompt_enhancer_strength !== undefined ? Number(params.prompt_enhancer_strength) : 1
        ,prompt_text_scale: params.prompt_text_scale !== undefined ? Number(params.prompt_text_scale) : 1.75
        ,sampling_preset: params.sampling_preset || (params.sampler === 'er_sde' ? 'detail' : 'default')
      }
      kreaStyleSelections = storedStyles.length ? storedStyles : [{ name: 'retroanime', strength: 1 }]
      userLoraSelections = storedUserLoras
    }
  }

  async function cloneImageReferences(job) {
    const inputs = await api.imageInputs(job.id)
    const stored = inputs.map((input) => ({ ...input, server: true }))
    clearRefs()
    setKreaImage('identity', null)
    setKreaImage('identityReference', null)
    setKreaImage('depth', null)
    setKreaImage('nk2e', null)
    setKreaImage('anypaint', null)
    setKreaImage('anypaintMask', null)
    setKreaImage('identityMask', null)
    setKreaImage('strictMask', null)
    clearKreaRefs('vision')
    clearKreaRefs('styleReference')
    const legacyKlein = job.params?.mode === 'edit'
    imageForm.mode = imageModeChoices.includes(job.params?.mode) ? job.params.mode : 'create'
    for (const input of stored) {
      if (input.role === 'reference' && legacyKlein && !kreaIdentityImage) setKreaImage('identity', input)
      else if (input.role === 'reference' && legacyKlein && !kreaIdentityReference) setKreaImage('identityReference', input)
      else if (input.role === 'reference') refs = [...refs, input]
      else if (input.role === 'identity') setKreaImage('identity', input)
      else if (input.role === 'identity_reference') setKreaImage('identityReference', input)
      else if (input.role === 'identity_mask') setKreaImage('identityMask', input)
      else if (input.role === 'strict_mask') setKreaImage('strictMask', input)
      else if (input.role === 'depth') setKreaImage('depth', input)
      else if (input.role === 'vision') kreaVisionImages = [...kreaVisionImages, { ...input, preview: input.url }]
      else if (input.role === 'style_reference') kreaStyleReferenceImages = [...kreaStyleReferenceImages, { ...input, preview: input.url }]
      else if (input.role === 'nk2e') setKreaImage('nk2e', input)
      else if (input.role === 'anypaint') setKreaImage('anypaint', input)
      else if (input.role === 'anypaint_mask') setKreaImage('anypaintMask', input)
    }
    if (imageForm.mode === 'create') {
      kreaModules = {
        ...kreaModules,
        identity: legacyKlein || stored.some((input) => input.role === 'identity'),
        depth: stored.some((input) => input.role === 'depth')
        ,vision: stored.some((input) => input.role === 'vision')
        ,styleReference: stored.some((input) => input.role === 'style_reference')
        ,nk2e: stored.some((input) => input.role === 'nk2e')
        ,anypaint: stored.some((input) => input.role === 'anypaint')
      }
    }
    return stored.length
  }

  function continueEditing(job) {
    const source = { server: true, ref: `${job.id}:output:0`, url: job.output_url, name: `결과 ${job.id.slice(0, 8)}.png`, role: 'output' }
    setKreaImage('identity', source)
    setKreaImage('identityReference', null); setKreaImage('identityMask', null); setKreaImage('strictMask', null)
    kreaModules = { identity: true, depth: false, style: false, userLora: false, vision: false, styleReference: false, nk2e: false, anypaint: false }
    parentImageJobID = job.id
    imageForm.width = Number(job.params?.width) || 1024; imageForm.height = Number(job.params?.height) || 1024; imageResolutionMode = 'custom'
    imageForm.prompt = ''; identityPreserve = 'identity, facial features, hair, body proportions, composition, and all areas not explicitly changed'
    resetImageEnhancement(); mobileImagePane = 'create'; window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  async function cloneImageJob(job, part) {
    cloningImageJob = `${job.id}:${part}`
    imageCloneMessage = ''
    error = ''
    try {
      if (part === 'prompt' || part === 'all') cloneImagePrompt(job)
      if (part === 'settings' || part === 'all') cloneImageSettings(job)
      let inputCount = null
      if (part === 'references' || part === 'all') inputCount = await cloneImageReferences(job)
      const labels = { prompt: '프롬프트', references: '참조 이미지', settings: '설정', all: '전체 작업' }
      imageCloneMessage = inputCount === 0 && part === 'references'
        ? '이 작업에는 복제할 참조 이미지가 없습니다.'
        : `${labels[part]}을 현재 작업으로 복제했습니다${inputCount ? ` · 이미지 ${inputCount}장` : ''}.`
      mobileImagePane = 'create'
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e) {
      error = e.message
    } finally {
      cloningImageJob = ''
    }
  }

  async function generateImage() {
    if (imageEnhancementActive() && !imageEnhancementCurrent()) {
      await enhanceImagePrompt()
      return
    }
    busy = true; error = ''
    try {
      const form = new FormData()
      Object.entries(imageForm).forEach(([key, value]) => form.append(key, key === 'prompt' ? (imageEnhancementActive() ? imageEnhancedPrompt : rawImagePrompt()) : value))
      form.append('original_prompt', rawImagePrompt())
      if (parentImageJobID) form.append('parent_job_id', parentImageJobID)
      if (imageForm.mode === 'create') {
        form.append('steps', kreaOptions.steps)
        form.append('filter_mode', kreaOptions.filter_mode)
        form.append('filter_strength', kreaOptions.filter_strength)
        form.append('prompt_enhancer', kreaOptions.prompt_enhancer)
        form.append('prompt_enhancer_strength', kreaOptions.prompt_enhancer_strength)
        form.append('prompt_text_scale', kreaOptions.prompt_text_scale)
        form.append('sampling_preset', kreaOptions.sampling_preset)
        if (kreaModules.identity) {
          appendImageInput(form, 'identity_image', 'reuse_identity_image', kreaIdentityImage)
          appendImageInput(form, 'identity_reference', 'reuse_identity_reference', kreaIdentityReference)
          appendImageInput(form, 'identity_mask', 'reuse_identity_mask', kreaIdentityMask)
          appendImageInput(form, 'strict_mask', 'reuse_strict_mask', kreaStrictMask)
          form.append('identity_strength', kreaOptions.identity_strength)
          form.append('ref_boost', kreaOptions.ref_boost)
          form.append('grounding_px', kreaOptions.grounding_px)
          form.append('strict_mask_grow', kreaOptions.strict_mask_grow)
          form.append('strict_mask_feather', kreaOptions.strict_mask_feather)
          form.append('vae_mode', kreaOptions.vae_mode)
          form.append('identity_fit_mode', kreaOptions.identity_fit_mode)
        }
        if (kreaModules.depth) {
          appendImageInput(form, 'depth_image', 'reuse_depth_image', kreaDepthImage)
          form.append('depth_strength', kreaOptions.depth_strength)
        }
        if (kreaModules.style) {
          form.append('styles', JSON.stringify(kreaStyleSelections))
        }
        if (kreaModules.userLora) {
          form.append('user_loras', JSON.stringify(userLoraSelections))
        }
        if (kreaModules.vision) {
          kreaVisionImages.forEach((image) => appendImageInput(form, 'vision_images', 'reuse_vision_images', image))
          form.append('vision_mode', kreaOptions.vision_mode)
          form.append('vision_megapixels', kreaOptions.vision_megapixels)
        }
        if (kreaModules.styleReference) {
          kreaStyleReferenceImages.forEach((image) => appendImageInput(form, 'style_reference_images', 'reuse_style_reference_images', image))
          form.append('style_reference_strength', kreaOptions.style_reference_strength)
        }
        if (kreaModules.nk2e) {
          appendImageInput(form, 'nk2e_image', 'reuse_nk2e_image', kreaNK2EImage)
          form.append('nk2e_mode', kreaOptions.nk2e_mode)
          form.append('nk2e_strength', kreaOptions.nk2e_strength)
          form.append('nk2e_preprocessed', kreaNK2EPreprocessed)
        }
        if (kreaModules.anypaint) {
          appendImageInput(form, 'anypaint_image', 'reuse_anypaint_image', kreaAnyPaintImage)
          appendImageInput(form, 'anypaint_mask', 'reuse_anypaint_mask', kreaAnyPaintMask)
          form.append('outpaint_left', kreaOptions.outpaint_left)
          form.append('outpaint_top', kreaOptions.outpaint_top)
          form.append('outpaint_right', kreaOptions.outpaint_right)
          form.append('outpaint_bottom', kreaOptions.outpaint_bottom)
          form.append('anypaint_strength', kreaOptions.anypaint_strength)
          form.append('anypaint_boundary_redraw_px', kreaOptions.anypaint_boundary_redraw_px)
        }
      }
      refs.forEach((image) => appendImageInput(form, 'references', 'reuse_references', image))
      await api.image(form)
      mobileImagePane = 'results'
      imageForm.prompt = ''; filterPromptPreset = ''; resetImageEnhancement(); clearRefs()
      parentImageJobID = ''; identityPreset = ''
      setKreaImage('identity', null); setKreaImage('identityReference', null); setKreaImage('depth', null); setKreaImage('nk2e', null)
      setKreaImage('anypaint', null); setKreaImage('anypaintMask', null)
      setKreaImage('identityMask', null); setKreaImage('strictMask', null)
      clearKreaRefs('vision'); clearKreaRefs('styleReference')
      showNewestListPage('image'); await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

  async function upscaleImage(job) {
    if (!job.output_url || upscalingImageJob || activeJobs().some((item) => item.kind === 'image' || item.kind === 'video')) return
    upscalingImageJob = job.id; error = ''
    try {
      await api.upscaleImage(job.id, { scale: 2, seed: -1 })
      showNewestListPage('image')
      await refresh()
    } catch (e) { error = e.message }
    finally { upscalingImageJob = '' }
  }

  async function detailEnhanceImage(job) {
    if (!job.output_url || detailEnhancingImageJob || activeJobs().some((item) => item.kind === 'image' || item.kind === 'video')) return
    detailEnhancingImageJob = job.id; error = ''
    try {
      await api.detailEnhanceImage(job.id, { strength: 1, seed: -1, vae: 'wan' })
      showNewestListPage('image')
      await refresh()
    } catch (e) { error = e.message }
    finally { detailEnhancingImageJob = '' }
  }

  async function generateSpeech() {
    busy = true; error = ''
    try {
      const form = new FormData()
      form.append('text', speechForm.text)
      form.append('instructions', speechForm.instructions)
      form.append('language', speechForm.language); form.append('speaker', speechForm.speaker); form.append('seed', speechForm.seed)
      await api.speech(form); speechForm.text = ''; showNewestListPage('speech'); await refresh()
      mobileSpeechPane = 'results'
    } catch (e) { error = e.message } finally { busy = false }
  }

  async function recognizeSpeech() {
    if ((recognitionForm.source === 'file' && !recognitionFile) || (recognitionForm.source === 'url' && !recognitionForm.url.trim())) return
    busy = true; error = ''
    try {
      const form = new FormData()
      if (recognitionForm.source === 'file') form.append('media', recognitionFile)
      else form.append('url', recognitionForm.url.trim())
      if (recognitionForm.source === 'url' && recognitionForm.media_part) form.append('media_part', recognitionForm.media_part)
      if (recognitionForm.source === 'url' && recognitionForm.media_source) form.append('media_source', recognitionForm.media_source)
      form.append('language', recognitionForm.language)
      form.append('context', recognitionForm.context)
      form.append('output_formats', recognitionForm.output_formats.join(','))
      form.append('translation_mode', recognitionForm.translation_mode)
      form.append('target_language', recognitionForm.target_language)
      await api.recognition(form)
      showNewestListPage('recognition')
      recognitionFile = null
      recognitionForm.url = ''
      resetRecognitionOptions()
      await refresh()
      mobileRecognitionPane = 'results'
    } catch (e) { error = e.message } finally { busy = false }
  }

  function resetRecognitionOptions() {
    recognitionOptions = null
    recognitionForm.media_part = ''
    recognitionForm.media_source = ''
  }

  function selectedRecognitionPart() {
    return recognitionOptions?.parts?.find((part) => part.id === recognitionForm.media_part) || recognitionOptions?.parts?.[0]
  }

  function selectRecognitionPart(partID) {
    recognitionForm.media_part = partID
    recognitionForm.media_source = ''
  }

  async function loadRecognitionOptions() {
    const url = recognitionForm.url.trim()
    if (!url) return
    loadingRecognitionOptions = true
    error = ''
    try {
      const options = await api.mediaOptions(url)
      if (url !== recognitionForm.url.trim()) return
      recognitionOptions = options
      recognitionForm.media_part = options.parts?.[0]?.id || ''
      recognitionForm.media_source = ''
    } catch (e) {
      error = e.message
      resetRecognitionOptions()
    } finally {
      loadingRecognitionOptions = false
    }
  }

  async function generateVideo() {
    if (videoEnhancementActive() && !videoEnhancementCurrent()) {
      await enhanceVideoPrompt()
      return
    }
    busy = true; error = ''
    try {
      const form = new FormData()
      Object.entries(videoForm).forEach(([key, value]) => form.append(key, key === 'prompt' && videoEnhancementActive() ? videoEnhancedPrompt : value))
      form.append('num_frames', framesForDuration(videoDurationSeconds, videoForm.fps))
      form.append('original_prompt', videoForm.prompt)
      if (videoImage) form.append('image', videoImage)
      await api.video(form)
      showNewestListPage('video')
      videoForm.prompt = ''; videoImage = null; resetVideoEnhancement(); await refresh()
      mobileVideoPane = 'results'
    } catch (e) { error = e.message } finally { busy = false }
  }

  function videoImageKey() {
    return videoImage ? `${videoImage.name}:${videoImage.size}:${videoImage.lastModified}` : ''
  }

  function videoEnhancementCurrent(enhanced = videoEnhancedPrompt, source = videoEnhancedSource, prompt = videoForm.prompt, imageKey = videoEnhancedImageKey, currentImageKey = videoImageKey()) {
    return enhanced.trim() !== '' && source === prompt.trim() && imageKey === currentImageKey
  }

  function videoEnhancementActive(enabled = videoEnhanceEnabled, image = videoImage, currentConfig = config) {
    return enabled && !(image && !currentConfig?.prompt_enhancement.vision_enabled)
  }

  $: videoEnhancementIsActive = videoEnhancementActive(videoEnhanceEnabled, videoImage, config)
  $: videoEnhancementIsCurrent = (
    videoImage,
    videoEnhancementCurrent(videoEnhancedPrompt, videoEnhancedSource, videoForm.prompt, videoEnhancedImageKey, videoImageKey())
  )

  function resetVideoEnhancement() {
    videoEnhancedPrompt = ''
    videoEnhancedSource = ''
    videoEnhancedImageKey = ''
  }

  function selectVideoImage(file) {
    videoImage = file || null
    resetVideoEnhancement()
  }

  async function enhanceVideoPrompt() {
    const original = videoForm.prompt.trim()
    if (!original) return
    enhancingPrompt = true; error = ''
    try {
      const form = new FormData()
      form.append('prompt', original)
      form.append('mode', videoImage ? 'i2v' : 't2v')
      if (videoImage) form.append('image', videoImage)
      const result = await api.enhancePrompt(form)
      videoEnhancedPrompt = result.enhanced_prompt
      videoEnhancedSource = original
      videoEnhancedImageKey = videoImageKey()
    } catch (e) { error = e.message }
    finally { enhancingPrompt = false }
  }

  async function deleteJob(job) {
    if (!confirm(`이 ${job.status === 'failed' ? '실패한 작업' : '작업'}과 저장 파일을 삭제할까요?`)) return
    deletingJob = job.id; error = ''
    try { await api.deleteJob(job.id); await refresh() }
    catch (e) { error = e.message }
    finally { deletingJob = '' }
  }

  async function cancelJob(job) {
    cancellingJob = job.id; error = ''
    try { await api.cancelJob(job.id); await refresh() }
    catch (e) { error = e.message }
    finally { cancellingJob = '' }
  }

  async function retryJob(job) {
    retryingJob = job.id; error = ''
    try { await api.retryJob(job.id); await refresh() }
    catch (e) { error = e.message }
    finally { retryingJob = '' }
  }

  async function clearFinishedJobs() {
    const count = jobs.filter((job) => job.status !== 'queued' && job.status !== 'running').length
    if (!count || !confirm(`완료·실패 작업 ${count}개와 저장 파일을 모두 삭제할까요?`)) return
    deletingJob = 'all'; error = ''
    try { await api.deleteFinishedJobs(); await refresh() }
    catch (e) { error = e.message }
    finally { deletingJob = '' }
  }

  function openSettings() {
    settings = structuredClone(config)
    settingsVideoDurationSeconds = durationFromFrames(config.video.default_frames, config.video.default_fps)
    savedMessage = ''
    error = ''
    tab = 'settings'
    storage = null
    api.storage().then((value) => storage = value).catch((e) => error = e.message)
  }

  async function cleanupTemporaryStorage() {
    const amount = formatBytes(storage?.reclaimable_bytes || 0)
    if (!confirm(`실행 중인 작업을 제외한 임시 파일 ${amount}을(를) 삭제할까요?`)) return
    cleaningStorage = true; error = ''; savedMessage = ''
    try {
      const result = await api.cleanupTemporaryStorage()
      storage = await api.storage()
      savedMessage = `임시 폴더 ${result.removed_directories}개, ${formatBytes(result.removed_bytes)}을(를) 정리했습니다.`
    } catch (e) { error = e.message }
    finally { cleaningStorage = false }
  }

  async function saveSettings() {
    busy = true; error = ''; savedMessage = ''
    try {
      settings.video.default_frames = framesForDuration(settingsVideoDurationSeconds, settings.video.default_fps)
      const result = await api.saveConfig(settings)
      config = result.config
      settings = structuredClone(result.config)
      imageForm.width = config.image.default_width
      imageForm.height = config.image.default_height
      imageForm.mode = imageModeChoices.includes(config.image.default_mode) ? config.image.default_mode : 'create'
      speechForm.language = config.speech.default_language
      speechForm.speaker = config.speech.default_speaker
      recognitionForm.language = config.recognition.default_language
      recognitionForm.output_formats = [...config.recognition.default_output_formats]
      recognitionForm.translation_mode = config.recognition.default_translation_mode
      recognitionForm.target_language = config.recognition.default_translation_language
      videoForm.width = config.video.default_width
      videoForm.height = config.video.default_height
      videoForm.fps = config.video.default_fps
      videoDurationSeconds = durationFromFrames(config.video.default_frames, config.video.default_fps)
      settingsVideoDurationSeconds = durationFromFrames(config.video.default_frames, config.video.default_fps)
      videoEnhanceEnabled = config.prompt_enhancement.default_enabled
      imageEnhanceEnabled = config.prompt_enhancement.default_enabled
      kreaOptions = { ...kreaOptions, prompt_enhancer: Boolean(config.image.default_prompt_enhancer) }
      savedMessage = result.restart_required
        ? '저장했습니다. Listen 주소 또는 데이터 폴더 변경은 Media 재시작 후 적용됩니다.'
        : '저장했습니다. API 연결과 생성 기본값이 즉시 적용됐습니다.'
      await refresh()
    } catch (e) { error = e.message } finally { busy = false }
  }

</script>

<svelte:window onkeydown={handleFeatureModulesKeydown} />

<datalist id="translation-languages">
  {#each translationLanguages as language}<option value={language}></option>{/each}
</datalist>

<svelte:head><meta name="theme-color" content="#101318"></svelte:head>

<header>
  <div><span class="mark">✦</span><h1>Spark Media</h1></div>
  <div class="engine-strip">
    <span class="system-usage" title="5초 간격으로 갱신되는 DGX Spark 사용률"><b>CPU</b> {systemUsage.cpu_percent ?? '–'}% <b>GPU</b> {systemUsage.gpu_percent ?? '–'}% <b>MEM</b> {systemUsage.mem_used_gb == null ? '–' : Number(systemUsage.mem_used_gb).toFixed(1)}/{systemUsage.mem_total_gb == null ? '–' : Number(systemUsage.mem_total_gb).toFixed(1)}GB({systemUsage.mem_percent ?? '–'}%)</span>
    {#if tab === 'image'}
      <span class:running={engineStates[imageModeMeta[imageForm.mode].engine] === 'online'}><i></i>{imageModeMeta[imageForm.mode].short} API<span class="engine-state-text"> · {engineStates[imageModeMeta[imageForm.mode].engine] || 'offline'}</span></span>
      <span class:running={engineStates.prompt === 'online'}><i></i>Enhancer API<span class="engine-state-text"> · {engineStates.prompt || 'offline'}</span></span>
      <span class:running={engineStates.upscale === 'online'}><i></i>Upscale API<span class="engine-state-text"> · {engineStates.upscale || 'offline'}</span></span>
    {:else if engineMeta[tab]}
      <span class:running={engineStates[engineMeta[tab][0]] === 'online'}><i></i>{engineMeta[tab][1]} API<span class="engine-state-text"> · {engineStates[engineMeta[tab][0]] || 'offline'}</span></span>
    {/if}
    {#if tab === 'video'}
      <span class:running={engineStates.prompt === 'online'}><i></i>Enhancer API<span class="engine-state-text"> · {engineStates.prompt || 'offline'}</span></span>
    {/if}
    {#if tab === 'recognition'}
      <span class:running={engineStates.recognition === 'online'}><i></i>ASR API<span class="engine-state-text"> · {engineStates.recognition || 'offline'}</span></span>
      {#if recognitionForm.translation_mode !== 'none'}<span class:running={engineStates.prompt === 'online'}><i></i>Translator API<span class="engine-state-text"> · {engineStates.prompt || 'offline'}</span></span>{/if}
    {/if}
  </div>
  <div class="mobile-engine-area">
    <span class="mobile-system-usage" title={`MEM ${systemUsage.mem_used_gb == null ? '–' : Number(systemUsage.mem_used_gb).toFixed(1)}/${systemUsage.mem_total_gb == null ? '–' : Number(systemUsage.mem_total_gb).toFixed(1)}GB(${systemUsage.mem_percent ?? '–'}%)`}>C {systemUsage.cpu_percent ?? '–'}% · G {systemUsage.gpu_percent ?? '–'}% · M {systemUsage.mem_used_gb == null ? '–' : Number(systemUsage.mem_used_gb).toFixed(1)}/{systemUsage.mem_total_gb == null ? '–' : Number(systemUsage.mem_total_gb).toFixed(1)}GB({systemUsage.mem_percent ?? '–'}%)</span>
    <button type="button" class="mobile-engine-summary {engineAggregate}" aria-expanded={mobileEngineOpen} aria-label={`API 상태: ${engineAggregateLabel}`} onclick={() => mobileEngineOpen = !mobileEngineOpen}><i></i><span>API</span></button>
    {#if mobileEngineOpen}
      <button type="button" class="mobile-engine-dismiss" aria-label="API 상태 닫기" onclick={() => mobileEngineOpen = false}></button>
      <section class="mobile-engine-popover" aria-label="각 API 상태">
        <header><strong>API 상태</strong><span class={engineAggregate}>{engineAggregateLabel}</span></header>
        <div>{#each monitoredEngineStatuses as item}<p class:online={item.online}><i></i><span>{item.label}</span><small>{item.online ? '정상' : '오프라인'}</small></p>{/each}</div>
      </section>
    {/if}
  </div>
</header>

<main>
  <nav>
    <button class:active={tab === 'image'} onclick={() => tab = 'image'}>이미지</button>
    <button class:active={tab === 'video'} onclick={() => tab = 'video'}>영상</button>
    <button class:active={tab === 'speech'} onclick={() => tab = 'speech'}>음성</button>
    <button class:active={tab === 'recognition'} onclick={() => tab = 'recognition'}>자막</button>
    <button class:active={tab === 'lora'} onclick={() => tab = 'lora'}>LoRA</button>
    <button class:active={tab === 'history'} onclick={() => tab = 'history'}>기록 <b>{jobs.length}</b></button>
    <button class:active={tab === 'settings'} onclick={openSettings}>설정</button>
  </nav>

  {#if error}<div class="error"><span>{error}</span><button onclick={() => error = ''}>×</button></div>{/if}

  {#if tab === 'image'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 이미지 화면">
      <button type="button" role="tab" aria-selected={mobileImagePane === 'create'} class:active={mobileImagePane === 'create'} onclick={() => mobileImagePane = 'create'}><span>만들기</span><small>설정·기능 모듈</small></button>
      <button type="button" role="tab" aria-selected={mobileImagePane === 'results'} class:active={mobileImagePane === 'results'} onclick={() => mobileImagePane = 'results'}><span>최근 이미지</span><small>{imageJobs.length}개{#if activeJobs().some((job) => job.kind === 'image')} · 생성 중{/if}</small></button>
    </div>
    <section class="workspace image-workspace" class:mobile-results={mobileImagePane === 'results'}>
      <form class="image-create-pane" onsubmit={(e) => { e.preventDefault(); generateImage() }}>
        <div class="section-title"><div><span>01</span><h2>이미지 생성</h2></div></div>
        {#if imageCloneMessage}<div class="clone-notice"><span>{imageCloneMessage}</span><button type="button" aria-label="복제 안내 닫기" onclick={() => imageCloneMessage = ''}>×</button></div>{/if}
        {#if imageForm.mode === 'create'}
          <div class="prompt-tools-row">
            <button type="button" class="prompt-tool-open" onclick={() => promptExamplesOpen = true}><span>프롬프트 예제</span>{#if filterPromptPreset}<b>선택됨</b>{/if}</button>
            <PromptComposer activeStyles={kreaModules.style ? kreaStyleSelections.map((style) => style.name) : []} onApply={(prompt, mode) => {
              const currentPrompt = imageForm.prompt.trimEnd()
              imageForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${prompt}` : prompt
              filterPromptPreset = ''
              resetImageEnhancement()
            }} />
          </div>
        {/if}
        <label>{kreaModules.identity ? '변경할 내용' : '프롬프트'}<textarea bind:value={imageForm.prompt} rows="7" placeholder="{kreaModules.identity ? '원본에서 바꿀 내용만 구체적으로 입력하세요.' : isPureOutpaint() ? '선택 사항 · 비워두면 원본을 자연스럽게 이어서 확장합니다.' : '만들고 싶은 장면을 입력하세요.'}"></textarea></label>
        {#if kreaModules.identity}
          <label>유지할 내용<textarea bind:value={identityPreserve} rows="3" placeholder="얼굴, 머리, 구도, 배경처럼 바꾸지 않을 요소"></textarea></label>
        {/if}
        <div class="enhanced-prompt image-enhancer-panel" class:inactive={!imageEnhancementIsActive}>
          <div class="image-enhancer-panel-header">
            <div class="enhancer-panel-title"><strong title="Gemma 4 E2B가 Krea 2용 영어 프롬프트로 확장합니다.">프롬프트 향상</strong><a href={kreaPromptGuideSource} target="_blank" rel="noreferrer">출처 ↗</a></div>
            <div class="enhancer-panel-actions">
              <button type="button" class="quiet enhancer-run" disabled={!imageEnhancementIsActive || enhancingPrompt || !imageForm.prompt.trim()} onclick={enhanceImagePrompt}>{enhancingPrompt ? '처리 중…' : imageEnhancementIsCurrent ? '다시 처리' : '미리 향상'}</button>
            <div class="segmented compact">
              <button type="button" class:active={imageEnhanceEnabled} onclick={() => imageEnhanceEnabled = true}>켜짐</button>
              <button type="button" class:active={!imageEnhanceEnabled} onclick={() => imageEnhanceEnabled = false}>꺼짐</button>
            </div>
            </div>
          </div>
          <textarea bind:value={imageEnhancedPrompt} rows="5" aria-label="Krea 향상 프롬프트" placeholder={imageEnhanceEnabled ? '미리 향상을 실행하면 여기에 결과가 표시됩니다.' : '프롬프트 향상을 켜면 사용할 수 있습니다.'}></textarea>
          <small>{looksLikeStructuredPrompt() ? 'JSON 형식은 원문을 유지합니다.' : imageEnhancementIsActive ? '실제 생성에 사용할 문장입니다. 확인하고 직접 수정할 수 있습니다.' : '꺼짐 · 기존 결과는 보존되며 실제 생성에는 원문을 사용합니다.'}</small>
        </div>
        {#if imageForm.mode === 'create'}
          <section class="krea-runtime-controls" aria-label="Krea 모델 내부 조정">
            <div class="runtime-control-heading"><div><strong>모델 내부 조정</strong><small>필터 벡터와 텍스트 조건 강도를 간단히 조절합니다.</small></div><button type="button" class="runtime-info-button" aria-label="모델 내부 조정 설명" title="설명 보기" onclick={() => runtimeInfoOpen = true}>i</button></div>
            <div class="runtime-control-row">
              <label><span>필터 완화</span><select value={kreaOptions.filter_mode} onchange={(event) => { const mode = event.currentTarget.value; kreaOptions = { ...kreaOptions, filter_mode: mode, filter_strength: filterModeDefault(mode) } }}><option value="off">꺼짐 · 원본</option><option value="adherence">준수 강화 · skc3vo</option><option value="balanced">균형 · 2-vector</option><option value="strong">강함 · 3-vector</option></select></label>
              <label><span>완화 강도</span><input type="range" min="0" max={filterModeMaximum(kreaOptions.filter_mode)} step="0.01" disabled={kreaOptions.filter_mode === 'off'} bind:value={kreaOptions.filter_strength}><b>{Number(kreaOptions.filter_strength).toFixed(2)}</b></label>
            </div>
            <div class="runtime-control-row adherence">
              <div><strong>프롬프트 준수 강화</strong><small>Krea2T Enhancer · 객체 수와 배치 같은 복잡한 지시를 더 강하게 반영</small></div>
              <div class="segmented compact"><button type="button" class:active={kreaOptions.prompt_enhancer} onclick={() => kreaOptions = { ...kreaOptions, prompt_enhancer: true }}>켜짐</button><button type="button" class:active={!kreaOptions.prompt_enhancer} onclick={() => kreaOptions = { ...kreaOptions, prompt_enhancer: false }}>꺼짐</button></div>
            </div>
            {#if kreaOptions.prompt_enhancer}<div class="runtime-control-row"><label><span>강화 강도</span><input type="range" min="0" max="2" step="0.05" bind:value={kreaOptions.prompt_enhancer_strength}><b>{Number(kreaOptions.prompt_enhancer_strength).toFixed(2)}</b></label><label><span>텍스트 비중</span><input type="range" min="0.25" max="4" step="0.05" bind:value={kreaOptions.prompt_text_scale}><b>{Number(kreaOptions.prompt_text_scale).toFixed(2)}</b></label></div>{/if}
          </section>
        {/if}
        {#if imageForm.mode === 'create'}
          <div class="feature-module-launch" class:has-warning={Boolean(kreaModuleMessage)}>
            <button type="button" aria-haspopup="dialog" onclick={() => featureModulesOpen = true}>
              <span><strong>기능 모듈</strong><small>{activeKreaModuleLabels.length ? activeKreaModuleLabels.join(' · ') : '필요한 기능만 선택해 사용'}</small></span>
              <b>{activeKreaModuleLabels.length ? `${activeKreaModuleLabels.length}개 사용` : '설정'}</b>
            </button>
            {#if kreaModuleMessage}<small>{kreaModuleMessage}</small>{/if}
          </div>
          {#if featureModulesOpen}
            <div class="feature-module-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) featureModulesOpen = false }}>
              <section class="feature-module-modal" role="dialog" aria-modal="true" aria-label="Krea 2 기능 모듈">
                <header>
                  <div><strong>Krea 2 기능 모듈</strong><small>필요한 기능만 켜면 내부 연결은 자동으로 구성됩니다. 변경 내용은 즉시 유지됩니다.</small></div>
                  <button type="button" aria-label="닫기" onclick={() => featureModulesOpen = false}>×</button>
                </header>
                <div class="feature-module-content">
                  {#if kreaModuleMessage}<div class="feature-module-warning">{kreaModuleMessage}</div>{/if}
                  <section class="module-panel" aria-label="Krea 생성 모듈">
            <article class="module-card" class:enabled={kreaModules.identity}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.identity} onclick={() => toggleKreaModule('identity')}>
                <span class="module-icon">REF</span><span><strong>원본 수정</strong><small>원본의 인물이나 장면을 유지하면서 원하는 부분 변경</small></span><i></i>
              </button>
              {#if kreaModules.identity}
                <div class="module-body">
                  {#if parentImageJobID}<div class="clone-notice"><span>결과 작업 {parentImageJobID.slice(0, 8)}에서 계속 편집 중</span><button type="button" onclick={() => parentImageJobID = ''}>×</button></div>{/if}
                  <label>무엇을 할까요?<select value={identityPreset} onchange={(event) => applyIdentityPreset(event.currentTarget.value)}><option value="">직접 지시</option><option value="restage">같은 인물로 장면 변경</option><option value="sheet">2×2 캐릭터 시트</option><option value="faceSwap">얼굴 교체</option><option value="headSwap">머리 전체 교체</option><option value="personSwap">인물 교체</option><option value="tryon">의상 교체</option><option value="replace">선택 영역 교체</option></select></label>
                  <div class="module-source-field"><label class="module-file">{identityUI.primary}<small>{identityUI.primaryHint}</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('identity', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaIdentityPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaIdentityPreview} alt={`${identityUI.primary} 미리보기`} title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaIdentityPreview, identityUI.primary)} onkeydown={(event) => showImageOnKey(event, kreaIdentityPreview, identityUI.primary)}>{:else}<i>REF</i>{/if}<b title={kreaIdentityImage?.name || identityUI.primaryHint}>{kreaIdentityImage?.name || identityUI.primaryHint}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'identity'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'identity'}>프리셋</button></div></div>
                  {#if identityUI.showSecondary}<div class="module-source-field"><label class="module-file" class:optional={!identityUI.secondaryRequired}>{identityUI.secondary}<small>{identityUI.secondaryHint}{identityUI.secondaryRequired ? ' · 필수' : ' · 선택 사항'}</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('identityReference', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaIdentityReferencePreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaIdentityReferencePreview} alt={`${identityUI.secondary} 미리보기`} title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaIdentityReferencePreview, identityUI.secondary)} onkeydown={(event) => showImageOnKey(event, kreaIdentityReferencePreview, identityUI.secondary)}>{:else}<i>+1</i>{/if}<b title={kreaIdentityReference?.name || identityUI.secondaryHint}>{kreaIdentityReference?.name || identityUI.secondaryHint}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'identityReference'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'identityReference'}>프리셋</button></div></div>{/if}
                  <p class="identity-prompt-guide">{identityUI.guide}</p>
                  <details class="module-advanced">
                    <summary><span>고급 설정</span><small>닮음·참조 해석·마스크</small></summary>
                    <div class="module-advanced-body">
                      <div class="module-controls">
                        <label><span>닮음 강도 <b>{kreaOptions.ref_boost}</b></span><input type="range" min="0" max="10" step="0.5" bind:value={kreaOptions.ref_boost}></label>
                        <label>참조 해석<select bind:value={kreaOptions.grounding_px}><option value={512}>변경 우선</option><option value={768}>균형</option><option value={1024}>얼굴 우선</option></select></label>
                      </div>
                      <div class="module-controls"><label>참조 맞춤<select bind:value={kreaOptions.identity_fit_mode}><option value="fit">전체 보존 · Fit</option><option value="crop">얼굴 확대 · Crop</option></select></label><label>VAE<select bind:value={kreaOptions.vae_mode}><option value="default">Qwen VAE</option><option value="wan">Wan 2.1 VAE · 권장</option><option value="real">Real VAE · 실험</option></select></label></div>
                      <div class="module-controls">
                        <label class="module-file optional">닮음 집중 마스크 <small>흰 영역의 Identity 주의만 높임</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('identityMask', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaIdentityMaskPreview}<img src={kreaIdentityMaskPreview} alt="닮음 집중 마스크">{:else}<i>FOCUS</i>{/if}<b>{kreaIdentityMask?.name || '선택 사항'}</b></span></label>
                        <button type="button" class="mask-editor-open" disabled={!kreaIdentityPreview} onclick={() => maskEditorMode = 'identity'}>얼굴·특징 집중 영역 칠하기</button>
                      </div>
                      <div class="module-controls">
                        <label class="module-file optional">변경 허용 마스크 <small>흰 영역 밖 픽셀을 원본 그대로 보존</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('strictMask', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaStrictMaskPreview}<img src={kreaStrictMaskPreview} alt="변경 허용 마스크">{:else}<i>LOCK</i>{/if}<b>{kreaStrictMask?.name || '선택 사항'}</b></span></label>
                        <button type="button" class="mask-editor-open" disabled={!kreaIdentityPreview} onclick={() => maskEditorMode = 'strict'}>변경 허용 영역 칠하기</button>
                      </div>
                      {#if kreaStrictMask}<div class="module-controls"><label>마스크 확장<input type="number" min="0" max="128" bind:value={kreaOptions.strict_mask_grow}></label><label>경계 부드럽게<input type="number" min="0" max="128" step="0.5" bind:value={kreaOptions.strict_mask_feather}></label></div>{/if}
                    </div>
                  </details>
                </div>
              {/if}
            </article>
            <article class="module-card" class:enabled={kreaModules.depth}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.depth} onclick={() => toggleKreaModule('depth')}>
                <span class="module-icon">3D</span><span><strong>자세·구도</strong><small>Depth Control · 다른 이미지의 공간과 동작 반영</small></span><i></i>
              </button>
              {#if kreaModules.depth}
                <div class="module-body">
                  <div class="module-source-field depth-source-field"><label class="module-file">구도 참조 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('depth', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaDepthPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaDepthPreview} alt="구도 참조 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaDepthPreview, 'Depth 구도 참조')} onkeydown={(event) => showImageOnKey(event, kreaDepthPreview, 'Depth 구도 참조')}>{:else}<i>3D</i>{/if}<b title={kreaDepthImage?.name || '원하는 자세와 구도의 이미지 선택'}>{kreaDepthImage?.name || '원하는 자세와 구도의 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'depth'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'depth'}>프리셋</button></div></div>
                  <label class="module-slider"><span>구도 고정 강도 <b>{Number(kreaOptions.depth_strength).toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" bind:value={kreaOptions.depth_strength}></label>
                </div>
              {/if}
            </article>
            <article class="module-card" class:enabled={kreaModules.nk2e}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.nk2e} onclick={() => toggleKreaModule('nk2e')}>
                <span class="module-icon">N2</span><span><strong>실험 편집·윤곽</strong><small>NK2E v0.3 · 국소 변경 또는 Canny 자세 반영</small></span><i></i>
              </button>
              {#if kreaModules.nk2e}
                <div class="module-body">
                  <div class="module-source-field"><label class="module-file">참조 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('nk2e', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaNK2EPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaNK2EPreview} alt="NK2E 참조 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaNK2EPreview, 'NK2E 참조')} onkeydown={(event) => showImageOnKey(event, kreaNK2EPreview, 'NK2E 참조')}>{:else}<i>N2</i>{/if}<b title={kreaNK2EImage?.name || '편집하거나 윤곽을 가져올 이미지 선택'}>{kreaNK2EImage?.name || '편집하거나 윤곽을 가져올 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'nk2e'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'nk2e'}>프리셋</button></div></div>
                  <div class="module-controls">
                    <label>작업 방식<select bind:value={kreaOptions.nk2e_mode}><option value="edit">국소 편집</option><option value="canny">윤곽·자세 반영</option></select></label>
                    <label class="module-slider"><span>반영 강도 <b>{Number(kreaOptions.nk2e_strength).toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" bind:value={kreaOptions.nk2e_strength}></label>
                  </div>
                  {#if kreaOptions.nk2e_mode === 'canny'}<button type="button" class="mask-editor-open" disabled={!kreaNK2EPreview} onclick={() => cannyEditorOpen = true}>{kreaNK2EPreprocessed ? '완성된 윤곽맵 다시 편집' : 'Canny 미리보기·편집'}</button>{/if}
                  <small class="module-caution">실험 기능입니다. 짧고 구체적인 변경 지시가 안정적이며, 현재 다른 Krea 모듈과는 함께 실행하지 않습니다.</small>
                </div>
              {/if}
            </article>
            <article class="module-card" class:enabled={kreaModules.anypaint}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.anypaint} onclick={() => toggleKreaModule('anypaint')}>
                <span class="module-icon">PAINT</span><span><strong>부분 수정·확장</strong><small>AnyPaint · 선택 영역 수정 또는 캔버스 바깥 생성</small></span><i></i>
              </button>
              {#if kreaModules.anypaint}
                <div class="module-body">
                  <div class="module-controls">
                    <div class="module-source-field"><label class="module-file">원본 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('anypaint', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaAnyPaintPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaAnyPaintPreview} alt="부분 수정 원본 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaAnyPaintPreview, '부분 수정·확장 원본')} onkeydown={(event) => showImageOnKey(event, kreaAnyPaintPreview, '부분 수정·확장 원본')}>{:else}<i>IMG</i>{/if}<b title={kreaAnyPaintImage?.name || '수정하거나 확장할 이미지 선택'}>{kreaAnyPaintImage?.name || '수정하거나 확장할 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'anypaint'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'anypaint'}>프리셋</button></div></div>
                    <label class="module-file optional">수정 마스크 <small>선택 사항 · 흰 영역을 새로 생성</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('anypaintMask', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaAnyPaintMaskPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaAnyPaintMaskPreview} alt="부분 수정 마스크 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaAnyPaintMaskPreview, '수정 마스크')} onkeydown={(event) => showImageOnKey(event, kreaAnyPaintMaskPreview, '수정 마스크')}>{:else}<i>MASK</i>{/if}<b title={kreaAnyPaintMask?.name || '확장만 할 때는 비워두기'}>{kreaAnyPaintMask?.name || '확장만 할 때는 비워두기'}</b></span></label>
                  </div>
                  <button type="button" class="mask-editor-open" disabled={!kreaAnyPaintPreview} onclick={() => maskEditorMode = 'anypaint'}>원본 위에서 수정 영역 칠하기</button>
                  <div class="outpaint-controls">
                    <strong>이미지 확장</strong><small>원본 크기에 선택한 픽셀만큼 더합니다.</small>
                    <div>
                      <label>왼쪽<select bind:value={kreaOptions.outpaint_left}><option value={0}>없음</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                      <label>위쪽<select bind:value={kreaOptions.outpaint_top}><option value={0}>없음</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                      <label>오른쪽<select bind:value={kreaOptions.outpaint_right}><option value={0}>없음</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                      <label>아래쪽<select bind:value={kreaOptions.outpaint_bottom}><option value={0}>없음</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                    </div>
                  </div>
                  <div class="module-controls">
                    <label class="module-slider"><span>생성 강도 <b>{Number(kreaOptions.anypaint_strength).toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" bind:value={kreaOptions.anypaint_strength}></label>
                    <label>경계 다시 그리기<select bind:value={kreaOptions.anypaint_boundary_redraw_px}><option value={0}>0px · 원본 우선</option><option value={16}>16px · 약하게</option><option value={32}>32px · 균형</option><option value={64}>64px · 자연스럽게</option></select></label>
                  </div>
                  <small class="module-caution">프롬프트에는 완성될 전체 장면을 적으세요. 원본 해상도 기준으로 작업하며 현재 다른 Krea 모듈과는 함께 실행하지 않습니다.</small>
                </div>
              {/if}
            </article>
            <article class="module-card" class:enabled={kreaModules.style}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.style} onclick={() => toggleKreaModule('style')}>
                <span class="module-icon">FX</span><span><strong>스타일 LoRA</strong><small>기본 모델 위에 시각 스타일 추가</small></span><i></i>
              </button>
              {#if kreaModules.style}
                <div class="module-body">
                  <div class="lora-picker" aria-label="스타일 LoRA 선택">
                    {#each kreaStyleCatalog as style}
                      <button type="button" class:selected={hasKreaStyle(style.name)} aria-pressed={hasKreaStyle(style.name)} onclick={() => toggleKreaStyle(style.name)}><i>{hasKreaStyle(style.name) ? '✓' : '+'}</i><span><strong title={style.label}>{style.label}</strong><small title={style.detail}>{style.detail}</small></span></button>
                    {/each}
                  </div>
                  {#if kreaStyleSelections.length}
                    <div class="lora-stack">
                      <header><strong>적용 순서</strong><span>{kreaStyleSelections.length}개 중첩</span></header>
                      {#each kreaStyleSelections as style, index}
                        <div class="lora-stack-item">
                          <span><b>{index + 1}</b><strong title={kreaStyleLabel(style.name)}>{kreaStyleLabel(style.name)}</strong></span>
                          <label><input type="range" min="0" max="2" step="0.1" value={style.strength} oninput={(event) => updateKreaStyleStrength(style.name, event.currentTarget.value)}><b>{Number(style.strength).toFixed(1)}</b></label>
                          <button type="button" aria-label={`${kreaStyleLabel(style.name)} 제거`} onclick={() => toggleKreaStyle(style.name)}>×</button>
                        </div>
                      {/each}
                    </div>
                  {:else}
                    <small class="module-caution">위 목록에서 적용할 LoRA를 선택하세요.</small>
                  {/if}
                  <small class="module-caution">선택한 순서대로 중첩됩니다. 3개 이상은 스타일 충돌로 형태나 색이 과해질 수 있습니다.</small>
                </div>
              {/if}
            </article>
            <article class="module-card" class:enabled={kreaModules.userLora}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.userLora} onclick={() => toggleKreaModule('userLora')}>
                <span class="module-icon">MY</span><span><strong>사용자 LoRA</strong><small>LoRA 제작소에서 학습한 인물·캐릭터·스타일</small></span><i></i>
              </button>
              {#if kreaModules.userLora}
                <div class="module-body">
                  <div class="module-toolbar"><small>최대 5개까지 중첩할 수 있습니다.</small><button type="button" class="quiet" onclick={refreshUserLoras}>새로고침</button></div>
                  {#if userLoraCatalog.length}
                    <div class="lora-picker" aria-label="사용자 LoRA 선택">
                      {#each userLoraCatalog as lora}
                        <button type="button" class:selected={hasUserLora(lora.filename)} aria-pressed={hasUserLora(lora.filename)} onclick={() => toggleUserLora(lora.filename)}><i>{hasUserLora(lora.filename) ? '✓' : '+'}</i><span><strong title={lora.name || lora.filename}>{lora.name || lora.filename}</strong><small title={lora.trigger_word || '트리거 없음'}>{lora.trigger_word || '트리거 없음'}</small></span></button>
                      {/each}
                    </div>
                    {#if userLoraSelections.length}
                      <div class="lora-stack">
                        <header><strong>적용 순서</strong><span>{userLoraSelections.length}개 중첩</span></header>
                        {#each userLoraSelections as selection, index}
                          <div class="lora-stack-item">
                            <span><b>{index + 1}</b><strong title={userLoraLabel(selection.filename)}>{userLoraLabel(selection.filename)}</strong></span>
                            <label><input type="range" min="0" max="2" step="0.01" value={selection.strength} oninput={(event) => updateUserLoraStrength(selection.filename, event.currentTarget.value)}><b>{Number(selection.strength).toFixed(2)}</b></label>
                            <button type="button" aria-label={`${selection.filename} 제거`} onclick={() => toggleUserLora(selection.filename)}>×</button>
                          </div>
                        {/each}
                      </div>
                    {/if}
                  {:else}
                    <small class="module-caution">등록된 LoRA가 없습니다. 상단 LoRA 탭에서 먼저 학습하세요.</small>
                  {/if}
                </div>
              {/if}
            </article>
            <article class="module-card" class:enabled={kreaModules.styleReference}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.styleReference} onclick={() => toggleKreaModule('styleReference')}>
                <span class="module-icon">REF</span><span><strong>스타일 이미지 참조</strong><small>Ostris Style Reference · 화풍·색감·질감 반영</small></span><i></i>
              </button>
              {#if kreaModules.styleReference}
                <div class="module-body">
                  <div class="module-source-field"><label class="module-file">스타일 이미지 · 최대 2장<input type="file" accept="image/*" multiple onchange={(e) => addKreaRefs('styleReference', e.currentTarget.files)}><span class="module-file-display"><i>REF</i><b>{kreaStyleReferenceImages.length ? `${kreaStyleReferenceImages.length}장 선택됨` : '화풍을 가져올 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'styleReference'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'styleReference'}>프리셋</button></div></div>
                  {#if kreaStyleReferenceImages.length}<div class="reference-previews">{#each kreaStyleReferenceImages as image, i}<div><button type="button" class="reference-preview-open" onclick={(event) => showImage(event, image.preview || image.url, `스타일 참조 ${i + 1}`)}><img src={image.preview || image.url} alt="스타일 참조 {i + 1}"></button><button type="button" class="reference-preview-remove" aria-label="스타일 참조 제거" onclick={() => removeKreaRef('styleReference', i)}>×</button></div>{/each}</div>{/if}
                  <label class="module-slider"><span>참조 강도 <b>{Number(kreaOptions.style_reference_strength).toFixed(1)}</b></span><input type="range" min="0" max="2" step="0.1" bind:value={kreaOptions.style_reference_strength}></label>
                  <small class="module-caution">전용 INT8 모델을 사용하므로 다른 Krea 모듈과는 아직 함께 실행하지 않습니다.</small>
                </div>
              {/if}
            </article>
            <article class="module-card" class:enabled={kreaModules.vision}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.vision} onclick={() => toggleKreaModule('vision')}>
                <span class="module-icon">VL</span><span><strong>내용·구도 참조</strong><small>Qwen3-VL · 사물·배치·시각적 내용을 의미적으로 반영</small></span><i></i>
              </button>
              {#if kreaModules.vision}
                <div class="module-body">
                  <div class="module-source-field"><label class="module-file">참조 이미지 · 최대 4장<input type="file" accept="image/*" multiple onchange={(e) => addKreaRefs('vision', e.currentTarget.files)}><span class="module-file-display"><i>VL</i><b>{kreaVisionImages.length ? `${kreaVisionImages.length}장 선택됨` : '내용을 참고할 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'vision'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'vision'}>프리셋</button></div></div>
                  {#if kreaVisionImages.length}<div class="reference-previews">{#each kreaVisionImages as image, i}<div><button type="button" class="reference-preview-open" onclick={(event) => showImage(event, image.preview || image.url, `내용·구도 참조 ${i + 1}`)}><img src={image.preview || image.url} alt="내용 참조 {i + 1}"></button><button type="button" class="reference-preview-remove" aria-label="내용 참조 제거" onclick={() => removeKreaRef('vision', i)}>×</button></div>{/each}</div>{/if}
                  <div class="module-controls">
                    <label>참조 방식<select bind:value={kreaOptions.vision_mode}><option value="descriptor">자연스럽게 반영</option><option value="instruct">변경 지시와 결합</option></select></label>
                    <label>이미지 해석<select bind:value={kreaOptions.vision_megapixels}><option value={0.5}>빠르게</option><option value={1}>균형</option><option value={2}>세밀하게</option></select></label>
                  </div>
                  <small class="module-caution">정확한 얼굴 고정이나 인페인팅이 아닌 의미 기반 참조입니다.</small>
                </div>
              {/if}
            </article>
                  </section>
                </div>
                <footer><button type="button" class="feature-modules-clear" disabled={!activeKreaModuleLabels.length} onclick={disableAllKreaModules}>모두 끄기</button><button type="button" class="feature-modules-done" onclick={() => featureModulesOpen = false}>완료</button></footer>
              </section>
            </div>
          {/if}
        {:else}
          <div class="drop" role="button" tabindex="0" ondragover={(e) => e.preventDefault()} ondrop={(e) => { e.preventDefault(); addRefs(e.dataTransfer.files) }}>
            <input type="file" accept="image/*" multiple={imageForm.mode === 'edit'} onchange={(e) => addRefs(e.currentTarget.files)}>
            <strong>{refs.length ? `${imageForm.mode === 'control' ? '제어' : '참조'} 이미지 ${refs.length}개` : `${imageForm.mode === 'control' ? '제어' : '참조'} 이미지 놓기`}</strong>
            <small>{imageForm.mode === 'control' ? '필수 · 윤곽을 추출할 이미지 1장' : `필수 · 최대 ${config?.image.max_reference_images || 4}개`} · 클릭하거나 드래그</small>
            {#if refs.length}<div class="drop-reference-previews">{#each refs as image, i}<div><button type="button" class="reference-preview-open" onclick={(event) => showImage(event, image.preview || image.url, `${imageModeMeta[imageForm.mode].label} 원본 ${i + 1}`)}><img src={image.preview || image.url} alt="참조 원본 {i + 1}"></button><button type="button" class="reference-preview-remove" aria-label="참조 원본 제거" onclick={(event) => { event.preventDefault(); event.stopPropagation(); removeRef(i) }}>×</button></div>{/each}</div>{/if}
          </div>
        {/if}
        <div class="resolution-control">
          <div class="resolution-heading"><div><strong>이미지 크기</strong><small>{imageForm.width}×{imageForm.height} · {(imageForm.width * imageForm.height / 1_000_000).toFixed(2)}MP</small></div><div class="segmented compact"><button type="button" class:active={imageResolutionMode === 'smart'} onclick={() => { imageResolutionMode = 'smart'; applySmartResolution() }}>간편</button><button type="button" class:active={imageResolutionMode === 'custom'} onclick={useCustomImageResolution}>직접</button></div></div>
          {#if imageResolutionMode === 'smart'}
            <div class="fields two smart-resolution-fields">
              <label>화면 비율<select bind:value={imageAspectRatio} onchange={applySmartResolution}>{#each imageAspectRatios as aspect}<option value={aspect[0]}>{aspect[0]} · {aspect[2]}</option>{/each}</select></label>
              <label>크기<select bind:value={imageMegapixels} onchange={applySmartResolution}><option value={0.75}>빠르게 · 0.75MP</option><option value={1}>기본 · 1MP</option><option value={2}>고해상도 · 2MP</option><option value={4} disabled={kreaModules.identity}>최대 품질 · 4MP</option></select></label>
            </div>
          {:else}
            <div class="fields two">
              <label>너비<input type="number" min="256" max="2048" step="16" bind:value={imageForm.width}></label>
              <label>높이<input type="number" min="256" max="2048" step="16" bind:value={imageForm.height}></label>
            </div>
          {/if}
        </div>
        {#if imageForm.mode === 'create'}
          <section class="image-generation-controls" aria-label="이미지 생성 설정">
            <div class="generation-control-heading"><strong>생성 설정</strong><small>{kreaOptions.sampling_preset === 'detail' ? 'ER-SDE / Simple' : 'Euler / Simple'} · {kreaOptions.steps} steps</small></div>
            <div class="generation-control-grid">
              <label class="sampling-field"><span>샘플링 프리셋</span><select bind:value={kreaOptions.sampling_preset}><option value="default">기본 · Euler / Simple</option><option value="detail">디테일 · ER-SDE / Simple</option></select></label>
              <label><span>스텝</span><select bind:value={kreaOptions.steps}><option value={8}>8 · 기본</option><option value={10}>10 · 균형</option><option value={12}>12 · 디테일</option></select></label>
              <label><span>시드 <small>-1 무작위</small></span><input type="number" min="-1" bind:value={imageForm.seed}></label>
            </div>
          </section>
        {:else}
          <div class="fields"><label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={imageForm.seed}></label></div>
        {/if}
        <button class="primary" disabled={Boolean(imageDisabledMessage) || enhancingPrompt}>{enhancingPrompt ? '프롬프트 향상 중…' : busy ? '요청 중…' : imageEnhancementIsActive && !imageEnhancementIsCurrent ? '프롬프트 향상 후 확인' : `${imageModeMeta[imageForm.mode].label} 시작`}</button>
        {#if imageDisabledMessage}<small class="submit-hint">{imageDisabledMessage}</small>{/if}
      </form>
      <aside class="image-results-pane">
        <div class="results-heading">
          <h3>최근 이미지</h3>
          <div class="view-switch" aria-label="최근 이미지 보기 방식">
            <button type="button" class:active={imageView === 'gallery'} onclick={() => setImageView('gallery')}>갤러리</button>
            <button type="button" class:active={imageView === 'list'} onclick={() => setImageView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="최근 이미지" total={imageJobs.length} page={listPages.image} pageSize={listPageSizes.image} pageSizes={imagePageSizeOptions} sortOrder={listSortOrders.image} onPageChange={(page) => setListPage('image', page)} onPageSizeChange={(size) => setListPageSize('image', size)} onSortOrderChange={(order) => setListSortOrder('image', order)} />
        <div class="gallery image-results" class:list-view={imageView === 'list'}>
        {#each pagedImageJobs as job (job.id)}
          {@const generationProgress = job.status === 'queued' || job.status === 'running' ? imageGenerationProgress(job) : null}
          <article class:pending={job.status !== 'completed'}>
            {#if imageView === 'list'}
              {#if job.output_url}<button type="button" class="image-list-thumb image-zoom" aria-label="생성 이미지 크게 보기" onclick={(event) => showImage(event, job.output_url, '생성 결과', job.prompt, job.id)}><img src={job.output_url} alt={job.prompt}></button>{:else}<div class="image-list-thumb placeholder">{#if generationProgress}<div class="image-generation-status compact"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small></div>{:else}<span>{job.status}</span>{/if}</div>{/if}
              <div class="image-list-content">
                <span>{imageModeMeta[job.params?.mode]?.label || '이미지'}{imageModuleSummary(job)} · {job.params?.width || '—'}×{job.params?.height || '—'}{#if imageSamplingSummary(job)} · {imageSamplingSummary(job)}{/if}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</span>
                <button type="button" class="image-prompt" title="클릭하여 전체 프롬프트 보기" onclick={() => promptModal = { title: '전체 프롬프트', detail: `${imageModeMeta[job.params?.mode]?.label || '이미지'} · ${job.params?.width || '—'}×${job.params?.height || '—'}${imageSamplingSummary(job) ? ` · ${imageSamplingSummary(job)}` : ''}`, text: imagePromptModalText(job) }}>{job.prompt}</button>
                {#if job.error}<em>{job.error}</em>{/if}
                <div class="image-clone-actions" aria-label="이 작업에서 복제">
                  <span>복제:</span>
                  <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'prompt')}>프롬프트</button>
                  <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'references')}>참조</button>
                  <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'settings')}>설정</button>
                  <button type="button" class="clone-all" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'all')}>{cloningImageJob === `${job.id}:all` ? '복제 중…' : '전체'}</button>
                </div>
                {#if job.status === 'completed'}<div class="image-post-actions"><span>후처리:</span><button type="button" title="이 결과를 Identity 원본으로 불러와 계속 편집" onclick={() => continueEditing(job)}>편집</button><button type="button" title="Ostris Edit LoRA로 다시 그립니다. 얼굴·색·글자·구도가 달라질 수 있습니다." disabled={Boolean(detailEnhancingImageJob) || Boolean(upscalingImageJob) || engineStates.image_create !== 'online' || activeJobs().some((item) => item.kind === 'image' || item.kind === 'video')} onclick={() => detailEnhanceImage(job)}>{detailEnhancingImageJob === job.id ? '처리 중…' : '디테일'}</button><button type="button" title="SeedVR2로 복원하고 2배 확대" disabled={Boolean(detailEnhancingImageJob) || Boolean(upscalingImageJob) || engineStates.upscale !== 'online' || activeJobs().some((item) => item.kind === 'image' || item.kind === 'video')} onclick={() => upscaleImage(job)}>{upscalingImageJob === job.id ? '처리 중…' : '고화질'}</button></div>{/if}
              </div>
            {:else}
              {#if job.output_url}<button type="button" class="gallery-image image-zoom" aria-label="생성 이미지 크게 보기" onclick={(event) => showImage(event, job.output_url, '생성 결과', job.prompt, job.id)}><img src={job.output_url} alt={job.prompt}></button>{:else}<div class="placeholder">{#if generationProgress}<div class="image-generation-status"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small>{#if generationProgress.eta}<small>{generationProgress.eta}</small>{/if}</div>{:else}<span>{job.status}</span>{/if}</div>{/if}<span class="image-mode-badge" title={`${imageModeMeta[job.params?.mode]?.label || '이미지'}${imageModuleSummary(job)}`}>{imageModeMeta[job.params?.mode]?.label || '이미지'}{imageModuleSummary(job)}</span>
              <button type="button" class="image-prompt" title="클릭하여 전체 프롬프트 보기" onclick={() => promptModal = { title: '전체 프롬프트', detail: `${imageModeMeta[job.params?.mode]?.label || '이미지'} · ${job.params?.width || '—'}×${job.params?.height || '—'}${imageSamplingSummary(job) ? ` · ${imageSamplingSummary(job)}` : ''}`, text: imagePromptModalText(job) }}>{job.prompt}</button>
              {#if job.error}<em>{job.error}</em>{/if}
              <div class="image-clone-actions" aria-label="이 작업에서 복제">
                <span>복제:</span>
                <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'prompt')}>프롬프트</button>
                <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'references')}>참조</button>
                <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'settings')}>설정</button>
                <button type="button" class="clone-all" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'all')}>{cloningImageJob === `${job.id}:all` ? '복제 중…' : '전체'}</button>
              </div>
              {#if job.status === 'completed'}<div class="image-post-actions"><span>후처리:</span><button type="button" title="이 결과를 Identity 원본으로 불러와 계속 편집" onclick={() => continueEditing(job)}>편집</button><button type="button" title="Ostris Edit LoRA로 다시 그립니다. 얼굴·색·글자·구도가 달라질 수 있습니다." disabled={Boolean(detailEnhancingImageJob) || Boolean(upscalingImageJob) || engineStates.image_create !== 'online' || activeJobs().some((item) => item.kind === 'image' || item.kind === 'video')} onclick={() => detailEnhanceImage(job)}>{detailEnhancingImageJob === job.id ? '처리 중…' : '디테일'}</button><button type="button" title="SeedVR2로 복원하고 2배 확대" disabled={Boolean(detailEnhancingImageJob) || Boolean(upscalingImageJob) || engineStates.upscale !== 'online' || activeJobs().some((item) => item.kind === 'image' || item.kind === 'video')} onclick={() => upscaleImage(job)}>{upscalingImageJob === job.id ? '처리 중…' : '고화질'}</button></div>{/if}
            {/if}
            {#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}
          </article>
        {:else}<div class="empty">첫 이미지가 여기에 나타납니다.</div>{/each}
      </div>
      <ResultPagination compact label="최근 이미지" total={imageJobs.length} page={listPages.image} pageSize={listPageSizes.image} onPageChange={(page) => setListPage('image', page)} />
      </aside>
    </section>
  {:else if tab === 'video'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 영상 화면">
      <button type="button" role="tab" aria-selected={mobileVideoPane === 'create'} class:active={mobileVideoPane === 'create'} onclick={() => mobileVideoPane = 'create'}><span>만들기</span><small>영상 생성 설정</small></button>
      <button type="button" role="tab" aria-selected={mobileVideoPane === 'results'} class:active={mobileVideoPane === 'results'} onclick={() => mobileVideoPane = 'results'}><span>최근 영상</span><small>{videoJobs.length}개{#if activeJobs().some((job) => job.kind === 'video')} · 생성 중{/if}</small></button>
    </div>
    <section class="workspace mobile-media-workspace" class:mobile-results={mobileVideoPane === 'results'}>
      <form class="mobile-create-pane" onsubmit={(e) => { e.preventDefault(); generateVideo() }}>
        <div class="section-title"><div><span>02</span><h2>LTX 2.5 영상 생성</h2></div></div>
        <label>원본 프롬프트<textarea bind:value={videoForm.prompt} rows="5" placeholder="장면과 움직임을 자연스럽게 입력하세요." required></textarea></label>
        <label class="file-field">시작 이미지 <small>선택 사항 · image-to-video</small><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => selectVideoImage(e.currentTarget.files?.[0])}><span>{videoImage?.name || '시작 이미지 선택'}</span></label>
        <div class="enhancer-control">
          <div>
            <strong>프롬프트 향상</strong>
            <small>{videoImage && !config?.prompt_enhancement.vision_enabled ? '현재 E2B 번들은 이미지를 볼 수 없어 I2V에서는 원문을 그대로 사용합니다.' : 'LTX 캡션 형식의 영어 프롬프트로 확장합니다.'}</small>
          </div>
          <div class="segmented compact">
            <button type="button" class:active={videoEnhanceEnabled} onclick={() => videoEnhanceEnabled = true}>켜짐</button>
            <button type="button" class:active={!videoEnhanceEnabled} onclick={() => videoEnhanceEnabled = false}>꺼짐</button>
          </div>
        </div>
        {#if videoEnhancementIsActive}
          <div class="enhanced-prompt">
            <div><span>향상된 프롬프트</span><button type="button" class="quiet" disabled={enhancingPrompt || !videoForm.prompt.trim()} onclick={enhanceVideoPrompt}>{enhancingPrompt ? '향상 중…' : videoEnhancementIsCurrent ? '다시 향상' : '미리 향상'}</button></div>
            {#if videoEnhancedPrompt}
              <textarea bind:value={videoEnhancedPrompt} rows="8" aria-label="향상된 프롬프트"></textarea>
              <small>{videoImage ? '시작 이미지를 분석해 확장했습니다.' : '텍스트 기반 T2V 확장입니다.'} 생성 전에 직접 수정할 수 있습니다.</small>
            {:else}
              <p>영상 만들기를 누르면 먼저 프롬프트를 향상하여 보여줍니다. 내용을 확인하거나 수정한 뒤 다시 누르면 생성합니다.</p>
            {/if}
          </div>
        {/if}
        <div class="fields three">
          <label>너비<input type="number" min="256" max="1920" step="64" bind:value={videoForm.width}></label>
          <label>높이<input type="number" min="256" max="1920" step="64" bind:value={videoForm.height}></label>
          <label class="duration-field"><span>길이 (초) <small>{framesForDuration(videoDurationSeconds, videoForm.fps)} 프레임 · 8k+1</small></span><input aria-label="영상 길이 초" type="number" min="0.1" step="0.1" bind:value={videoDurationSeconds}></label>
          <label>FPS<input type="number" min="1" max="60" step="1" bind:value={videoForm.fps}></label>
          <label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={videoForm.seed}></label>
          <label>이미지 강도<input type="number" min="0" max="1" step="0.05" bind:value={videoForm.image_strength}></label>
        </div>
        <button class="primary" disabled={busy || enhancingPrompt || activeJobs().some((j) => j.kind === 'image' || j.kind === 'video')}>{enhancingPrompt ? '프롬프트 향상 중…' : busy ? '요청 중…' : videoEnhancementIsActive && !videoEnhancementIsCurrent ? '프롬프트 향상 후 확인' : '영상 만들기'}</button>
      </form>
      <aside class="video-results-pane mobile-results-pane">
        <div class="results-heading">
          <h3>최근 영상</h3>
          <div class="view-switch" aria-label="최근 영상 보기 방식">
            <button type="button" class:active={videoView === 'gallery'} onclick={() => setVideoView('gallery')}>갤러리</button>
            <button type="button" class:active={videoView === 'list'} onclick={() => setVideoView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="최근 영상" total={videoJobs.length} page={listPages.video} pageSize={listPageSizes.video} pageSizes={imagePageSizeOptions} sortOrder={listSortOrders.video} onPageChange={(page) => setListPage('video', page)} onPageSizeChange={(size) => setListPageSize('video', size)} onSortOrderChange={(order) => setListSortOrder('video', order)} />
        <div class="video-list" class:list-view={videoView === 'list'}>
        {#each pagedVideoJobs as job (job.id)}
          {@const generationProgress = job.status === 'queued' || job.status === 'running' ? videoGenerationProgress(job) : null}
          <article class:pending={job.status !== 'completed'}>
            {#if videoView === 'list'}
              {#if job.output_url}<button type="button" class="video-list-thumb" aria-label="영상 크게 보기" onclick={() => showVideo(job)}><!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.output_url}></video></button>{:else}<div class="video-list-thumb empty-thumb">{#if generationProgress}<div class="image-generation-status compact"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small></div>{:else}<span>{job.status}</span>{/if}</div>{/if}
              <div class="video-list-content"><small>{job.params?.width}×{job.params?.height} · {formatDuration(videoJobDuration(job))} · {job.params?.fps} fps{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small><p title={job.prompt}>{job.prompt}</p>{#if job.error}<em>{job.error}</em>{/if}</div>
            {:else}
              {#if job.output_url}<button type="button" class="video-gallery-thumb" aria-label="영상 크게 보기" title="클릭하여 크게 보기" onclick={() => showVideo(job)}><!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.output_url}></video></button>{:else}<div class="video-placeholder">{#if generationProgress}<div class="image-generation-status"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small>{#if generationProgress.eta}<small>{generationProgress.eta}</small>{/if}</div>{:else}<span>{job.status}</span>{/if}</div>{/if}<p title={job.prompt}>{job.prompt}</p><small>{job.params?.width}×{job.params?.height} · {formatDuration(videoJobDuration(job))} · {job.params?.fps} fps{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small>{#if job.error}<em>{job.error}</em>{/if}
            {/if}
            {#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}
          </article>
        {:else}<div class="empty">첫 영상이 여기에 나타납니다.</div>{/each}
      </div>
      <ResultPagination compact label="최근 영상" total={videoJobs.length} page={listPages.video} pageSize={listPageSizes.video} onPageChange={(page) => setListPage('video', page)} />
      </aside>
    </section>
  {:else if tab === 'speech'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 음성 화면">
      <button type="button" role="tab" aria-selected={mobileSpeechPane === 'create'} class:active={mobileSpeechPane === 'create'} onclick={() => mobileSpeechPane = 'create'}><span>만들기</span><small>음성 생성 설정</small></button>
      <button type="button" role="tab" aria-selected={mobileSpeechPane === 'results'} class:active={mobileSpeechPane === 'results'} onclick={() => mobileSpeechPane = 'results'}><span>최근 음성</span><small>{speechJobs.length}개{#if activeJobs().some((job) => job.kind === 'speech')} · 생성 중{/if}</small></button>
    </div>
    <section class="workspace mobile-media-workspace" class:mobile-results={mobileSpeechPane === 'results'}>
      <form class="mobile-create-pane" onsubmit={(e) => { e.preventDefault(); generateSpeech() }}>
        <div class="section-title"><div><span>03</span><h2>CustomVoice 음성 생성</h2></div></div>
        <label>읽을 문장<textarea bind:value={speechForm.text} rows="7" placeholder="음성으로 변환할 문장을 입력하세요." required></textarea></label>
        <label>연기 지시 <small>선택 사항 · 1.7B instruction control</small><textarea bind:value={speechForm.instructions} rows="3" placeholder="예: 기쁘고 활기찬 목소리로, 중요한 단어는 힘주어 말해 주세요."></textarea></label>
        <div class="fields three">
          <label>언어<select bind:value={speechForm.language}><option>Korean</option><option>English</option><option>Chinese</option><option>Japanese</option><option>Auto</option></select></label>
          <label>화자<select bind:value={speechForm.speaker}><option>Sohee</option><option>Vivian</option><option>Serena</option><option>Ryan</option><option>Aiden</option><option>Ono_Anna</option></select></label>
          <label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={speechForm.seed}></label>
        </div>
        <button class="primary" disabled={busy || activeJobs().some((j) => j.kind === 'speech')}>{busy ? '요청 중…' : '음성 만들기'}</button>
      </form>
      <aside class="mobile-results-pane"><div class="results-heading"><h3>최근 음성</h3></div>
        <ResultPagination label="최근 음성" total={speechJobs.length} page={listPages.speech} pageSize={listPageSizes.speech} pageSizes={pageSizeOptions} sortOrder={listSortOrders.speech} onPageChange={(page) => setListPage('speech', page)} onPageSizeChange={(size) => setListPageSize('speech', size)} onSortOrderChange={(order) => setListSortOrder('speech', order)} />
        <div class="audio-list">
        {#each pagedSpeechJobs as job (job.id)}<article><div><span>{job.params?.speaker}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</span><p>{job.prompt}</p></div>{#if job.params?.instructions}<small class="instruction">지시 · {job.params.instructions}</small>{/if}{#if job.output_url}<audio controls src={job.output_url}></audio>{:else}<small>{job.status}</small>{/if}{#if job.error}<em>{job.error}</em>{/if}{#if job.status === 'completed' || job.status === 'failed'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</article>{:else}<div class="empty">첫 음성이 여기에 나타납니다.</div>{/each}
      </div>
      <ResultPagination compact label="최근 음성" total={speechJobs.length} page={listPages.speech} pageSize={listPageSizes.speech} onPageChange={(page) => setListPage('speech', page)} />
      </aside>
    </section>
  {:else if tab === 'recognition'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 자막 화면">
      <button type="button" role="tab" aria-selected={mobileRecognitionPane === 'create'} class:active={mobileRecognitionPane === 'create'} onclick={() => mobileRecognitionPane = 'create'}><span>만들기</span><small>자막 생성 설정</small></button>
      <button type="button" role="tab" aria-selected={mobileRecognitionPane === 'results'} class:active={mobileRecognitionPane === 'results'} onclick={() => mobileRecognitionPane = 'results'}><span>최근 자막</span><small>{recognitionJobs.length}개{#if activeJobs().some((job) => job.kind === 'recognition')} · 처리 중{/if}</small></button>
    </div>
    <section class="workspace mobile-media-workspace" class:mobile-results={mobileRecognitionPane === 'results'}>
      <form class="mobile-create-pane" onsubmit={(e) => { e.preventDefault(); recognizeSpeech() }}>
        <div class="section-title"><div><span>04</span><h2>자막과 스크립트</h2></div></div>
        <div class="segmented source-selector">
          <button type="button" class:active={recognitionForm.source === 'file'} onclick={() => recognitionForm.source = 'file'}>파일 업로드</button>
          <button type="button" class:active={recognitionForm.source === 'url'} onclick={() => recognitionForm.source = 'url'}>영상 링크</button>
        </div>
        {#if recognitionForm.source === 'file'}
          <label class="file-field">영상·음성 파일<input type="file" accept="audio/*,video/*,.mkv,.mp4,.webm,.mov,.m4v,.avi,.wav,.flac,.ogg,.mp3,.m4a,.aac" onchange={(e) => recognitionFile = e.currentTarget.files?.[0] || null}><span>{recognitionFile?.name || '영상 또는 음성 파일 선택'}</span></label>
          <small class="form-note">긴 파일은 메모리에 올리지 않고 작업 폴더로 스트리밍 업로드합니다.</small>
        {:else}
          <label>영상 페이지 주소<input type="url" bind:value={recognitionForm.url} oninput={resetRecognitionOptions} placeholder="https://www.youtube.com/watch?v=…" required></label>
          <small class="form-note">영상을 호스트 저장소에 보관하고 음성을 분리합니다. 직접 추출 실패 시 Chromium·Firefox 해석기를 사용합니다.</small>
          <button type="button" class="quiet media-options-load" disabled={loadingRecognitionOptions || !recognitionForm.url.trim()} onclick={loadRecognitionOptions}>{loadingRecognitionOptions ? '영상 내부 선택지 조회 중…' : '영상 내부 선택지 조회'}</button>
          {#if recognitionOptions}
            <div class="media-options">
              {#if recognitionOptions.parts?.length}
                {#if recognitionOptions.parts.length > 1}
                  <div class="media-option-row"><strong>파트</strong><div class="media-option-buttons">
                    {#each recognitionOptions.parts as part (part.id)}<button type="button" class:active={recognitionForm.media_part === part.id} onclick={() => selectRecognitionPart(part.id)}>{part.label}</button>{/each}
                  </div></div>
                {/if}
                <div class="media-option-row"><strong>영상 출처</strong><div class="media-option-buttons">
                  <button type="button" class:active={!recognitionForm.media_source} onclick={() => recognitionForm.media_source = ''}>자동 · StreamTape 우선</button>
                  {#each selectedRecognitionPart()?.sources || [] as source (source.id)}<button type="button" class:active={recognitionForm.media_source === source.id} onclick={() => recognitionForm.media_source = source.id}>{source.label}</button>{/each}
                </div></div>
              {:else}
                <small>이 주소에는 선택 가능한 파트나 별도 영상 출처가 없습니다. 기본 방식으로 처리합니다.</small>
              {/if}
            </div>
          {/if}
        {/if}
        <div class="fields">
          <label>언어<select bind:value={recognitionForm.language}>{#each recognitionLanguages as option}<option value={option[0]}>{option[1]}</option>{/each}</select></label>
          <label>구간 길이<input value={`${config?.recognition.segment_seconds || 180}초`} disabled></label>
        </div>
        <label>컨텍스트·전문용어<textarea bind:value={recognitionForm.context} rows="4" placeholder="선택 사항 · 인명, 제품명, 전문용어 등을 입력하세요."></textarea></label>
        <fieldset class="format-options">
          <legend>결과 형식 <small>복수 선택 가능</small></legend>
          <label><input type="checkbox" value="srt" bind:group={recognitionForm.output_formats}>SRT 자막</label>
          <label><input type="checkbox" value="vtt" bind:group={recognitionForm.output_formats}>VTT 자막</label>
          <label><input type="checkbox" value="timestamped_txt" bind:group={recognitionForm.output_formats}>타임코드 TXT</label>
          <label><input type="checkbox" value="txt" bind:group={recognitionForm.output_formats}>일반 TXT</label>
        </fieldset>
        <div class="fields">
          <label>번역<select bind:value={recognitionForm.translation_mode}><option value="none">번역 안 함</option><option value="translated">번역문만</option><option value="bilingual">원문과 번역문</option></select></label>
          <label>번역 언어<input list="translation-languages" bind:value={recognitionForm.target_language} disabled={recognitionForm.translation_mode === 'none'} placeholder="Korean"></label>
        </div>
        <button class="primary" disabled={busy || recognitionForm.output_formats.length === 0 || (recognitionForm.source === 'file' ? !recognitionFile : !recognitionForm.url.trim())}>{busy ? '등록 중…' : activeJobs().some((j) => j.kind === 'recognition') ? '자막 큐에 추가' : '자막 만들기'}</button>
      </form>
      <aside class="subtitle-results-pane mobile-results-pane">
        <div class="results-heading">
          <h3>최근 자막</h3>
          <div class="view-switch" aria-label="최근 자막 보기 방식">
            <button type="button" class:active={subtitleView === 'gallery'} onclick={() => setSubtitleView('gallery')}>갤러리</button>
            <button type="button" class:active={subtitleView === 'list'} onclick={() => setSubtitleView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="최근 자막" total={recognitionJobs.length} page={listPages.recognition} pageSize={listPageSizes.recognition} pageSizes={imagePageSizeOptions} sortOrder={listSortOrders.recognition} onPageChange={(page) => setListPage('recognition', page)} onPageSizeChange={(size) => setListPageSize('recognition', size)} onSortOrderChange={(order) => setListSortOrder('recognition', order)} />
        <div class="audio-list subtitle-results" class:list-view={subtitleView === 'list'}>
        {#each pagedRecognitionJobs as job (job.id)}
          <article class:pending={job.status === 'queued' || job.status === 'running'}>
            {#if subtitleView === 'list'}
              {#if job.media_url || job.params?.text}<button type="button" class="subtitle-list-thumb" class:empty-thumb={!job.media_url || isAudioMedia(job)} aria-label="자막 결과 크게 보기" onclick={() => showSubtitle(job)}>{#if job.media_url && !isAudioMedia(job)}<!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.media_url}></video>{:else}<span>{job.media_url && isAudioMedia(job) ? 'AUDIO' : job.params?.text ? 'TEXT' : job.status}</span>{/if}</button>{:else}<div class="subtitle-list-thumb empty-thumb"><span>{job.status}</span></div>{/if}
            {:else}
              {#if job.media_url || job.params?.text}<button type="button" class="subtitle-gallery-thumb" class:empty-thumb={!job.media_url || isAudioMedia(job)} aria-label="자막 결과 크게 보기" onclick={() => showSubtitle(job)}>{#if job.media_url && !isAudioMedia(job)}<!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.media_url}></video>{:else}<span>{job.media_url && isAudioMedia(job) ? 'AUDIO' : job.params?.text ? 'TEXT' : statusLabels[job.status] || job.status}</span>{/if}</button>{:else}<div class="subtitle-gallery-thumb empty-thumb"><span>{statusLabels[job.status] || job.status}</span></div>{/if}
            {/if}
            <div class="subtitle-result-title"><span>{job.params?.detected_language || recognitionLanguageLabel(job.params?.language)}{#if job.params?.segments} · {job.params.segments}구간{/if}{#if job.params?.media_part} · 파트 {job.params.media_part}{/if}{#if job.params?.media_source} · {job.params.media_source}{/if}</span><p title={job.prompt}>{job.prompt}</p>{#if job.params?.media}<small>{mediaSummary(job)}</small>{/if}</div>
            {#if !job.params?.text}<small class="recognition-progress-text">{recognitionProgressText(job)}</small>{/if}
            {#if job.status === 'queued' || job.status === 'running'}<div class="recognition-progress" aria-label={recognitionProgressText(job)}><i style={`width: ${recognitionProgressPercent(job)}%`}></i></div>{/if}
            {#if job.outputs}<div class="output-links">{#each Object.entries(job.outputs) as output}<a href={output[1]} target="_blank">{outputLabels[output[0]] || output[0]} ↗</a>{/each}</div>{:else if job.output_url}<a href={job.output_url} target="_blank">결과 열기 ↗</a>{/if}
            {#if job.error}<em>{job.error}</em>{/if}
            {#if job.status === 'queued' || job.status === 'running'}<button class="job-stop" disabled={cancellingJob === job.id} onclick={() => cancelJob(job)}>{cancellingJob === job.id ? '중지 중…' : job.status === 'queued' ? '대기 취소' : '중지'}</button>{:else}<div class="job-actions">{#if job.status === 'failed' || job.status === 'cancelled'}<button class="job-retry" disabled={retryingJob === job.id} onclick={() => retryJob(job)}>{retryingJob === job.id ? '재개 중…' : '재개'}</button>{/if}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button></div>{/if}
          </article>
        {:else}<div class="empty">첫 자막 작업이 여기에 나타납니다.</div>{/each}
      </div>
      <ResultPagination compact label="최근 자막" total={recognitionJobs.length} page={listPages.recognition} pageSize={listPageSizes.recognition} onPageChange={(page) => setListPage('recognition', page)} />
      </aside>
    </section>
  {:else if tab === 'lora'}
    <LoraStudio />
  {:else if tab === 'settings' && settings}
    <form class="settings" onsubmit={(e) => { e.preventDefault(); saveSettings() }}>
      <div class="section-title"><div><span>SET</span><h2>연결 및 기본 설정</h2></div></div>
      {#if savedMessage}<div class="success">{savedMessage}</div>{/if}

      <div class="settings-tabs" role="tablist" aria-label="설정 구역">
        <button type="button" role="tab" aria-selected={settingsSection === 'connection'} class:active={settingsSection === 'connection'} onclick={() => settingsSection = 'connection'}><span>연결</span><small>앱·API 주소</small></button>
        <button type="button" role="tab" aria-selected={settingsSection === 'defaults'} class:active={settingsSection === 'defaults'} onclick={() => settingsSection = 'defaults'}><span>생성 기본값</span><small>이미지·영상·음성·자막</small></button>
        <button type="button" role="tab" aria-selected={settingsSection === 'metadata'} class:active={settingsSection === 'metadata'} onclick={() => settingsSection = 'metadata'}><span>이미지 정보</span><small>EXIF 제작자 정보</small></button>
        <button type="button" role="tab" aria-selected={settingsSection === 'storage'} class:active={settingsSection === 'storage'} onclick={() => settingsSection = 'storage'}><span>저장소</span><small>용량·자동 정리</small></button>
      </div>

      {#if settingsSection === 'connection'}
      <div class="settings-section connection-settings">
      <section class="settings-card">
        <h3>Media 앱</h3>
        <p>Listen 주소와 데이터 폴더는 저장되지만 실행 중인 서버에는 재시작 후 적용됩니다.</p>
        <div class="fields">
          <label>Listen 주소<input bind:value={settings.listen} required></label>
          <label>데이터 폴더<input bind:value={settings.data_dir} required></label>
        </div>
      </section>

      <section class="settings-card">
        <h3>API 연결</h3>
        <div class="endpoint-list">
          {#each [['video', 'LTX 영상'], ['speech', 'Qwen3 TTS'], ['recognition', 'Qwen3 ASR'], ['prompt', 'Gemma 프롬프트·번역'], ['upscale', 'SeedVR2 고화질'], ['media', '미디어 접근·FFmpeg'], ['trainer', 'Krea 2 LoRA 학습']] as item}
            <label><span>{item[1]} <small class:online={engineStates[item[0]] === 'online'}>{engineStates[item[0]] || 'offline'}</small></span><input type="url" bind:value={settings.engines[item[0]].endpoint} required></label>
          {/each}
        </div>
      </section>
      </div>
      {/if}

      {#if settingsSection === 'storage'}
      <section class="settings-card storage-card">
        <div class="storage-heading">
          <div><h3>저장소 관리</h3><p>실행 중인 작업은 정리 대상에서 제외됩니다.</p></div>
          <button type="button" class="quiet danger" disabled={cleaningStorage || !storage?.reclaimable_directories} onclick={cleanupTemporaryStorage}>{cleaningStorage ? '정리 중…' : '찌꺼기 정리'}</button>
        </div>
        <div class="storage-stats">
          <span><small>임시 파일</small><strong>{storage ? formatBytes(storage.temporary_bytes) : '확인 중…'}</strong></span>
          <span><small>정리 가능</small><strong>{storage ? `${storage.reclaimable_directories}개 · ${formatBytes(storage.reclaimable_bytes)}` : '확인 중…'}</strong></span>
          <span><small>사용 중</small><strong>{storage ? `${storage.active_directories}개` : '확인 중…'}</strong></span>
        </div>
        <div class="fields storage-policy">
          <label>시작 시 자동 정리<select bind:value={settings.storage.cleanup_on_startup}><option value={true}>사용</option><option value={false}>꺼짐</option></select></label>
          <label>자동 정리 보존 시간<input type="number" min="1" max="8760" bind:value={settings.storage.temp_retention_hours}><small>이 시간보다 오래된 중단 작업만 앱 시작 시 정리합니다.</small></label>
        </div>
      </section>
      {/if}

      <div class="settings-grid">
        {#if settingsSection === 'defaults'}
        <section class="settings-card">
          <h3>이미지</h3>
          {#each imageModeChoices as mode}
            <div class="backend-setting"><strong>{imageModeMeta[mode].label}</strong><label>Endpoint<input type="url" bind:value={settings.image.backends[mode].endpoint} required></label><label>모델<input bind:value={settings.image.backends[mode].model} required></label></div>
          {/each}
          <div class="fields three">
            <label>기본 너비<input type="number" min="256" step="16" bind:value={settings.image.default_width}></label>
            <label>기본 높이<input type="number" min="256" step="16" bind:value={settings.image.default_height}></label>
            <label>참조 이미지 수<input type="number" min="1" max="16" bind:value={settings.image.max_reference_images}></label>
          </div>
        </section>

        <section class="settings-card">
          <h3>영상</h3>
          <label>모델<input bind:value={settings.video.model} required></label>
          <div class="fields">
            <label>기본 너비<input type="number" min="256" step="64" bind:value={settings.video.default_width}></label>
            <label>기본 높이<input type="number" min="256" step="64" bind:value={settings.video.default_height}></label>
            <label class="duration-field"><span>기본 길이 (초) <small>{framesForDuration(settingsVideoDurationSeconds, settings.video.default_fps)} 프레임 · 8k+1</small></span><input aria-label="기본 영상 길이 초" type="number" min="0.1" step="0.1" bind:value={settingsVideoDurationSeconds}></label>
            <label>기본 FPS<input type="number" min="1" max="60" bind:value={settings.video.default_fps}></label>
          </div>
        </section>

        <section class="settings-card">
          <h3>프롬프트</h3>
          <label>향상 모델<input bind:value={settings.prompt_enhancement.model} required></label>
          <div class="fields">
            <label>프롬프트 향상 기본값<select bind:value={settings.prompt_enhancement.default_enabled}><option value={true}>켜짐</option><option value={false}>꺼짐</option></select></label>
            <label>프롬프트 준수 강화 기본값<select bind:value={settings.image.default_prompt_enhancer}><option value={true}>켜짐</option><option value={false}>꺼짐</option></select></label>
            <label>최대 토큰<input type="number" min="64" max="2048" bind:value={settings.prompt_enhancement.max_tokens}></label>
            <label>이미지 인식<select bind:value={settings.prompt_enhancement.vision_enabled}><option value={false}>꺼짐</option><option value={true}>켜짐</option></select></label>
          </div>
          <small>프롬프트 향상은 이미지와 영상에 함께 적용됩니다. 준수 강화는 Krea 2 이미지 생성의 Krea2T 기본값입니다.</small>
          <small>I2V 시작 이미지는 LTX에 직접 전달하므로 이미지 인식 기반 프롬프트 향상은 기본적으로 꺼짐을 유지하세요.</small>
        </section>
        {/if}

        {#if settingsSection === 'metadata'}
        <section class="settings-card">
          <h3>이미지 EXIF 제작자 정보</h3>
          <p>비워둔 항목은 새 이미지의 EXIF에서 생략됩니다.</p>
          <div class="fields">
            <label>제작자 이름<input maxlength="256" bind:value={settings.image_metadata.creator} placeholder="이름 또는 스튜디오명"></label>
            <label>저작권 문구<input maxlength="512" bind:value={settings.image_metadata.copyright} placeholder="© 2026 이름. All rights reserved."></label>
            <label>웹사이트·연락처<input maxlength="2048" bind:value={settings.image_metadata.website} placeholder="https://… 또는 이메일"></label>
          </div>
          <label>메모<textarea rows="3" maxlength="2000" bind:value={settings.image_metadata.note} placeholder="작품이나 제작자에 관한 짧은 안내"></textarea></label>
        </section>
        {/if}

        {#if settingsSection === 'defaults'}
        <section class="settings-card">
          <h3>음성 생성</h3>
          <label>CustomVoice 모델<input bind:value={settings.speech.custom_voice_model} required></label>
          <div class="fields">
            <label>기본 언어<input bind:value={settings.speech.default_language} required></label>
            <label>기본 화자<input bind:value={settings.speech.default_speaker} required></label>
          </div>
        </section>

        <section class="settings-card">
          <h3>자막</h3>
          <label>ASR 모델<input bind:value={settings.recognition.model} required></label>
          <div class="fields">
            <label>기본 언어<select bind:value={settings.recognition.default_language}>{#each recognitionLanguages as option}<option value={option[0]}>{option[1]}</option>{/each}</select></label>
            <label>최대 업로드 MB<input type="number" min="1" bind:value={settings.recognition.max_upload_mb}></label>
            <label>구간 길이(초)<input type="number" min="5" max="180" bind:value={settings.recognition.segment_seconds}></label>
            <label>기본 번역<select bind:value={settings.recognition.default_translation_mode}><option value="none">번역 안 함</option><option value="translated">번역문만</option><option value="bilingual">원문과 번역문</option></select></label>
          </div>
          <label>기본 번역 언어<input list="translation-languages" bind:value={settings.recognition.default_translation_language} required></label>
          <fieldset class="format-options settings-formats">
            <legend>기본 결과 형식</legend>
            <label><input type="checkbox" value="srt" bind:group={settings.recognition.default_output_formats}>SRT</label>
            <label><input type="checkbox" value="vtt" bind:group={settings.recognition.default_output_formats}>VTT</label>
            <label><input type="checkbox" value="timestamped_txt" bind:group={settings.recognition.default_output_formats}>타임코드 TXT</label>
            <label><input type="checkbox" value="txt" bind:group={settings.recognition.default_output_formats}>일반 TXT</label>
          </fieldset>
        </section>
        {/if}
      </div>
      <div class="settings-save-bar"><small>변경 내용은 모든 설정 구역에 함께 저장됩니다.</small><button class="primary settings-save" disabled={busy}>{busy ? '저장 중…' : '설정 저장'}</button></div>
    </form>
  {:else}
    <section class="history"><div class="section-title"><div><span>05</span><h2>생성 기록</h2></div>{#if jobs.some((job) => job.status !== 'queued' && job.status !== 'running')}<button class="quiet danger" disabled={deletingJob === 'all'} onclick={clearFinishedJobs}>모두 비우기</button>{/if}</div>
      <ResultPagination label="생성 기록" total={jobs.length} page={listPages.history} pageSize={listPageSizes.history} pageSizes={pageSizeOptions} sortOrder={listSortOrders.history} onPageChange={(page) => setListPage('history', page)} onPageSizeChange={(size) => setListPageSize('history', size)} onSortOrderChange={(order) => setListSortOrder('history', order)} />
      {#each pagedHistoryJobs as job (job.id)}<article><span class="kind">{kindLabels[job.kind] || job.kind}</span><div><button type="button" class="history-prompt" title="전체 내용 보기" onclick={() => promptModal = { title: `${kindLabels[job.kind] || job.kind} 작업`, detail: `${new Date(job.created_at).toLocaleString()} · ${statusLabels[job.status] || job.status}`, text: job.prompt }}>{job.prompt}</button><small>{new Date(job.created_at).toLocaleString()} · {statusLabels[job.status] || job.status}</small>{#if job.error}<em>{job.error}</em>{/if}</div><div class="job-actions">{#if job.output_url}<a href={job.output_url} target="_blank">열기 ↗</a>{/if}{#if job.kind === 'recognition' && (job.status === 'failed' || job.status === 'cancelled')}<button class="job-retry" disabled={retryingJob === job.id} onclick={() => retryJob(job)}>{retryingJob === job.id ? '재개 중…' : '재개'}</button>{/if}{#if job.status !== 'queued' && job.status !== 'running'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</div></article>{:else}<div class="empty">아직 생성 기록이 없습니다.</div>{/each}
      <ResultPagination compact label="생성 기록" total={jobs.length} page={listPages.history} pageSize={listPageSizes.history} onPageChange={(page) => setListPage('history', page)} />
    </section>
  {/if}
</main>

{#if runtimeInfoOpen}
  <div class="runtime-info-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) runtimeInfoOpen = false }}>
    <section class="runtime-info-modal" role="dialog" aria-modal="true" aria-label="모델 내부 조정 설명">
      <header><div><strong>모델 내부 조정</strong><small>필터 벡터는 한 번에 하나를 고르고 텍스트 조건은 필요할 때 함께 사용합니다.</small></div><button type="button" aria-label="닫기" onclick={() => runtimeInfoOpen = false}>×</button></header>
      <div class="runtime-info-content">
        <article><strong>준수 강화 · skc3vo</strong><p>text-fusion projector 전체를 조절하는 rank-1 벡터입니다. 세부 지시를 더 직접적으로 따르게 하며 기본 강도는 0.05입니다.</p><a href="https://www.reddit.com/r/StableDiffusion/comments/1ueacq2/comment/otix1aa/" target="_blank" rel="noreferrer">출처 ↗</a></article>
        <article><strong>균형 · 2-vector</strong><p>Fedor 구현으로 projector의 두 필터 축만 완화합니다. 먼저 1.0에서 시작하고 부족할 때 2.0까지 올립니다.</p><a href="https://github.com/CliffNodes/fedor_bypass" target="_blank" rel="noreferrer">출처 ↗</a></article>
        <article><strong>강함 · 3-vector</strong><p>2-vector에 세 번째 필터 축을 더한 강한 대안입니다. 2-vector가 부족할 때 전환하며 두 방식을 중첩하지 않습니다.</p><a href="https://huggingface.co/uzumix/krea2filterbypass3.safetensors" target="_blank" rel="noreferrer">출처 ↗</a></article>
        <article><strong>프롬프트 준수 강화 · Krea2T</strong><p>Krea 2의 text-fusion 경로와 결합된 텍스트 토큰 비중을 조절해 객체 수, 배치, 관계 같은 지시를 더 강하게 전달합니다.</p><a href="https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer" target="_blank" rel="noreferrer">출처 ↗</a></article>
      </div>
    </section>
  </div>
{/if}

<MaskEditor open={Boolean(maskEditorMode)} source={maskEditorMode === 'anypaint' ? kreaAnyPaintPreview : kreaIdentityPreview} existingMask={maskEditorMode === 'anypaint' ? kreaAnyPaintMaskPreview : maskEditorMode === 'identity' ? kreaIdentityMaskPreview : kreaStrictMaskPreview} title={maskEditorMode === 'identity' ? '닮음 집중 영역' : maskEditorMode === 'strict' ? '변경 허용 영역' : '수정 영역 칠하기'} description={maskEditorMode === 'identity' ? '빨간 영역의 Identity 주의를 더 높입니다.' : maskEditorMode === 'strict' ? '빨간 영역만 생성 결과를 쓰고 바깥 픽셀은 원본 그대로 둡니다.' : '빨간 영역을 Krea가 새로 생성합니다.'} outputName={`${maskEditorMode || 'krea'}-mask.png`} onApply={usePaintedMask} onClose={() => maskEditorMode = ''} />
<CannyEditor open={cannyEditorOpen} source={kreaNK2EPreview} preprocessed={kreaNK2EPreprocessed} onApply={useCannyMap} onClose={() => cannyEditorOpen = false} />
<ImageModal image={imageModal} onClose={() => imageModal = null} />
<VideoModal video={videoModal} onClose={() => videoModal = null} />
<SubtitleModal result={subtitleModal} onClose={() => subtitleModal = null} />
<PromptModal prompt={promptModal} onClose={() => promptModal = null} />
<PromptExamplesModal open={promptExamplesOpen} examples={filterPromptPresets} selectedID={filterPromptPreset} officialSource={kreaPromptGuideSource} communitySource={filterPromptSource} vibeSource={vibePromptGuideSource} onApply={applyPromptExample} onClose={() => promptExamplesOpen = false} />
<RecentImagePicker open={Boolean(recentImagePickerTarget)} title={recentImagePickerTarget === 'identityReference' ? `${identityUI.secondary} 선택` : recentImagePickerTarget === 'depth' ? '자세·구도 이미지 선택' : recentImagePickerTarget === 'nk2e' ? '편집·윤곽 이미지 선택' : recentImagePickerTarget === 'anypaint' ? '부분 수정·확장 원본 선택' : recentImagePickerTarget === 'styleReference' ? '스타일 참조 이미지 추가' : recentImagePickerTarget === 'vision' ? '내용·구도 참조 이미지 추가' : `${identityUI.primary} 선택`} jobs={imageJobs} selectedRef={recentImagePickerTarget === 'identityReference' ? (kreaIdentityReference?.ref || '') : recentImagePickerTarget === 'depth' ? (kreaDepthImage?.ref || '') : recentImagePickerTarget === 'nk2e' ? (kreaNK2EImage?.ref || '') : recentImagePickerTarget === 'anypaint' ? (kreaAnyPaintImage?.ref || '') : recentImagePickerTarget === 'identity' ? (kreaIdentityImage?.ref || '') : ''} onSelect={useRecentModuleImage} onClose={() => recentImagePickerTarget = ''} />
<PresetImagePicker open={Boolean(presetImagePickerTarget)} title={presetImagePickerTarget === 'identityReference' ? `${identityUI.secondary} 프리셋 선택` : presetImagePickerTarget === 'depth' ? '자세·구도 프리셋 선택' : presetImagePickerTarget === 'nk2e' ? '편집·윤곽 프리셋 선택' : presetImagePickerTarget === 'anypaint' ? '부분 수정·확장 원본 프리셋' : presetImagePickerTarget === 'styleReference' ? '스타일 참조 프리셋 추가' : presetImagePickerTarget === 'vision' ? '내용·구도 참조 프리셋 추가' : `${identityUI.primary} 프리셋 선택`} examples={filterPromptPresets} initialTab={presetImagePickerTarget === 'depth' || presetImagePickerTarget === 'nk2e' ? 'pose' : 'example'} onSelect={usePresetModuleImage} onClose={() => presetImagePickerTarget = ''} />
