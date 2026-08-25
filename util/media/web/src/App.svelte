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
  import AudioModal from './AudioModal.svelte'
  import PromptModal from './PromptModal.svelte'
  import LoraStudio from './LoraStudio.svelte'
  import PromptComposer from './PromptComposer.svelte'
  import PromptExamplesModal from './PromptExamplesModal.svelte'
  import RecentImagePicker from './RecentImagePicker.svelte'
  import PresetImagePicker from './PresetImagePicker.svelte'
  import RemoteImageModal from './RemoteImageModal.svelte'
  import AssistantChat from './AssistantChat.svelte'
  import SparkBolt from './SparkBolt.svelte'
  import GarmentExtractorModal from './GarmentExtractorModal.svelte'
  import { lockModalScroll } from './modalScroll.js'

  let tab = 'image'
  let config = null
  let settings = null
  let savedMessage = ''
  let settingsSection = 'connection'
  let jobs = []
  let engineStates = { image: 'offline', speech: 'offline', recognition: 'offline', video: 'offline', prompt: 'offline', media: 'offline', trainer: 'offline', upscale: 'offline', garment: 'offline' }
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
  let promptExamplesTarget = 'image'
  let garmentExtractorOpen = false
  let garmentExtractorInitialJob = null
  let imageSequenceOpen = false
  let imageSequencePrompts = ['', '']
  let imageSequenceRegions = ['all', 'all']
  let imageSequenceMasks = [null, null]
  let imageSequenceMaskPreviews = ['', '']
  let imageSequenceBase = null
  let imageSequenceMaskEditorIndex = -1
  let imageSequenceRegionPicker = -1
  let imageSequenceStrength = 0.8
  let releaseImageSequenceScroll = null
  let kreaModules = { identity: false, depth: false, style: false, userLora: false, vision: false, styleReference: false, nk2e: false, anypaint: false }
  let kreaIdentityImage = null
  let kreaIdentityReference = null
  let kreaIdentityReferences = []
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
  const identityPreserveCatalog = [
    { id: 'identity', label: '인물 정체성', prompt: 'exact identity' },
    { id: 'face', label: '얼굴 특징', prompt: 'facial features' },
    { id: 'hair', label: '헤어', prompt: 'hairstyle' },
    { id: 'body', label: '체형', prompt: 'body proportions' },
    { id: 'clothing', label: '의상', prompt: 'original clothing' },
    { id: 'pose', label: '자세', prompt: 'original pose and body orientation' },
    { id: 'background', label: '배경', prompt: 'background' },
    { id: 'lighting', label: '조명', prompt: 'lighting' },
    { id: 'composition', label: '구도', prompt: 'composition and camera viewpoint' },
    { id: 'untouched', label: '나머지 영역', prompt: 'all areas not explicitly changed' }
  ]
  const defaultIdentityPreserveItems = ['identity', 'face', 'hair', 'body', 'clothing', 'pose', 'background', 'lighting', 'composition', 'untouched']
  let identityPreserveItems = [...defaultIdentityPreserveItems]
  let identityPreserveCustom = ''
  let identityPreset = ''
  let imageModal = null
  let videoModal = null
  let subtitleModal = null
  let audioModal = null
  let promptModal = null
  let runtimeInfoOpen = false
  let releaseRuntimeInfoScroll = null
  let featureModulesOpen = false
  let recentImagePickerTarget = ''
  let presetImagePickerTarget = ''
  let remoteImageTarget = ''
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

  $: {
    if (imageSequenceOpen && !releaseImageSequenceScroll) releaseImageSequenceScroll = lockModalScroll()
    else if (!imageSequenceOpen && releaseImageSequenceScroll) {
      releaseImageSequenceScroll()
      releaseImageSequenceScroll = null
    }
  }

  onDestroy(() => {
    releaseRuntimeInfoScroll?.()
    releaseFeatureModulesScroll?.()
    releaseImageSequenceScroll?.()
    releaseVideoImage(videoImage)
    releaseVideoImage(videoEndImage)
    videoKeyframes.forEach((keyframe) => releaseVideoImage(keyframe.image))
  })
  let kreaVisionImages = []
  let kreaStyleReferenceImages = []
  let kreaStyleSelections = [{ name: 'retroanime', strength: 1 }]
  let userLoraCatalog = []
  let userLoraSelections = []
  let kreaOptions = {
    checkpoint: 'official',
    identity_strength: 1, ref_boost: 4, source_ref_boost: 1, grounding_px: 768, steps: 8,
    identity_model: 'convrot', identity_encoder: 'heretic',
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
    source: 'url', url: '', language: 'Auto', context: '',
    output_formats: ['srt', 'txt'], translation_mode: 'none', target_language: 'Korean',
    media_part: '', media_source: ''
  }
  let recognitionFile = null
  let recognitionFileInput
  let recognitionOptions = null
  let loadingRecognitionOptions = false
  let videoForm = { prompt: '', width: 768, height: 512, fps: 24, seed: -1, image_strength: 1 }
  let videoDurationSeconds = 5
  let settingsVideoDurationSeconds = 5
  let videoImage = null
  let videoEndImage = null
  let videoEndStrength = 1
  let videoKeyframes = []
  let nextVideoKeyframeID = 1
  let videoImagePickerTarget = ''
  let videoRemoteImageTarget = ''
  let videoEnhanceEnabled = true
  let videoEnhancedPrompt = ''
  let videoEnhancedSource = ''
  let videoEnhancedImageKey = ''
  let videoEnhancementIsActive = false
  let videoEnhancementIsCurrent = false
  let creatingVideoPrompt = false
  let videoPromptCreationMessage = ''
  let videoPromptPreset = ''
  let enhancingPrompt = false
  let deletingJob = ''
  let cancellingJob = ''
  let retryingJob = ''
  let storage = null
  let cleaningStorage = false
  let videoModelStatus = null
  let hfToken = ''
  let preparingVideoModels = false
  let imageCheckpointStatus = null
  let civitaiToken = ''
  let savingDownloadCredentials = false
  let preparingImageCheckpoints = false
  let checkpointSelection = {
    'ray-v1': true, 'ray-v2': true, 'ray-v3': true, 'ray-v4': true,
    'moody-v7': true, 'moody-cutie-v4': true, 'moody-amateur-v1': true,
    'chriscole-edit-v1.1': true
  }
  const checkpointDisplayChoices = [
    ['chriscole-edit-v1.1', 'Krea 2 Turbo Edit v1.1 · FP8'],
    ['moody-v7', 'Moody Krea 2 Mix V7 · NVFP4'], ['moody-cutie-v4', 'Moody Cutie Mix V4 · NVFP4'], ['moody-amateur-v1', 'Moody Amateur Mix V1 · NVFP4'],
    ['ray-v1', 'Ray Artshoot V1 · FP8'], ['ray-v2', 'Ray Artshoot V2 · FP8'], ['ray-v2-nvfp4', 'Ray Artshoot V2 · NVFP4'],
    ['ray-v3', 'Ray Artshoot V3 · INT8'], ['ray-v4', 'Ray Artshoot V4 · INT8'], ['ray-v4-nvfp4', 'Ray Artshoot V4 · NVFP4']
  ]
  let nvfp4Selection = { 'ray-v2': true, 'ray-v4': true }
  let convertingImageCheckpoints = false
  let removeBF16Sources = false
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
    lora: ['image_create', 'LoRA 관리']
  }
  const engineStatusCatalog = [
    ['image_create', 'Krea 2 이미지'], ['video', 'LTX 영상'], ['speech', 'Qwen3 TTS'],
    ['recognition', 'Qwen3 ASR'], ['prompt', '프롬프트·번역'], ['upscale', 'SeedVR2 고화질'], ['garment', '의상 추출'],
    ['media', '미디어·FFmpeg']
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
    upscale: { label: '고화질', short: 'SeedVR2', engine: 'upscale', help: '완성된 이미지를 SeedVR2로 복원하고 확대합니다.' },
    garment_extract: { label: '의상 추출', short: 'Garment', engine: 'garment', help: '의상만 투명 PNG와 마스크로 분리합니다.' }
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
  const wildcardPromptSource = 'https://huggingface.co/datasets/Crocody/mymuse/tree/main/Wildcards'
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
    { id: 'vibe-hanbok', label: '한복 인물 · 복합 지시', category: 'portrait', source: vibePromptGuideSource, image: 'vibe-hanbok.png', prompt: `A young Korean woman in pastel pink traditional Joseon hanbok, with intricate embroidery. Perfect makeup. Delicate indigo braided hair, elegantly adorned with red flowers and beads in exquisite detail. A woman with a hairpin tucked behind her head, holding a round folding fan with a tree and a bird. Neon lightning lamp, a bright yellow light floats above with the left hand open. A softly lit outdoor night background, with the silhouette in the draft visible over the vast, drifting Gyeongbokgung Palace, and a variety of distant lights faintly blurred. Realistic, cinematic in feel, as shown in the photos, with ultra-high-resolution detail.` },
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
    if (params.identity && !params.sequence_total) modules.push('Identity')
    if (params.depth) modules.push('Depth')
    if (params.styles?.length || params.style) modules.push(`LoRA${params.styles?.length > 1 ? ` ×${params.styles.length}` : ''}`)
    if (params.user_loras?.length) modules.push(`사용자 LoRA${params.user_loras.length > 1 ? ` ×${params.user_loras.length}` : ''}`)
    if (params.style_reference) modules.push('Style Ref')
    if (params.vision) modules.push('Vision')
    if (params.nk2e) modules.push(params.nk2e_mode === 'canny' ? 'NK2E Canny' : 'NK2E Edit')
    if (params.anypaint) modules.push(params.anypaint_mask ? 'Inpaint' : 'Outpaint')
    if (params.sequence_total) modules.push(`연속 ${params.sequence_index}/${params.sequence_total}`)
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
    if (mode === 'garment_extract') return 12
    if (mode === 'upscale') return Math.max(20, 16 * megapixels)
    if (mode === 'detail_enhance') return Math.max(18, 5 + steps * megapixels * 1.5)
    const moduleCount = ['identity', 'depth', 'vision', 'style_reference', 'nk2e', 'anypaint'].filter((name) => params[name]).length
    return Math.max(8, 4 + steps * megapixels * 1.15 * (1 + moduleCount * .18))
  }

  function imageGenerationProgress(job) {
    const created = Date.parse(job.created_at || 0)
    if (job.status === 'queued') {
      const elapsed = Number.isFinite(created) ? (progressClock - created) / 1000 : 0
      const position = generationQueuePosition(job)
      return { label: position ? `대기 ${position}번째` : '대기 중', percent: 2, elapsed: `대기 ${compactElapsed(elapsed)}`, eta: '앞선 작업 종료 후 시작' }
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
      const position = generationQueuePosition(job)
      return { label: position ? `대기 ${position}번째` : '대기 중', percent: 2, elapsed: `대기 ${compactElapsed(elapsed)}`, eta: '앞선 작업 종료 후 시작' }
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
    const enhanced = job.params?.generated_edit_prompt || job.params?.enhanced_prompt || job.params?.source_enhanced_prompt
    if (!enhanced) return job.prompt || ''
    return `원문\n${job.prompt || ''}\n\n실제 생성 프롬프트\n${enhanced}`
  }

  function videoPromptModalText(job) {
    const original = job.prompt || ''
    const enhanced = job.params?.enhanced_prompt || job.params?.source_enhanced_prompt
    if (!enhanced || enhanced.trim() === original.trim()) return original
    return `원문\n${original}\n\n실제 생성 프롬프트\n${enhanced}`
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

  function generationQueuePosition(job) {
	const queued = jobs
	  .filter((item) => ['image', 'video', 'speech'].includes(item.kind) && item.status === 'queued')
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

  async function refreshVideoModelStatus() {
    try {
      videoModelStatus = await api.videoModels()
    } catch {
      videoModelStatus = null
    }
  }

  async function refreshImageCheckpointStatus() {
    try {
      imageCheckpointStatus = await api.imageCheckpoints()
    } catch {
      imageCheckpointStatus = null
    }
  }

  async function prepareImageCheckpoints() {
    const variants = Object.entries(checkpointSelection).filter(([, selected]) => selected).map(([id]) => id)
    if (!variants.length) {
      error = '준비할 Krea 체크포인트를 하나 이상 선택하세요.'
      return
    }
    preparingImageCheckpoints = true
    error = ''
    savedMessage = ''
    try {
      imageCheckpointStatus = await api.prepareImageCheckpoints(civitaiToken.trim(), hfToken.trim(), variants)
      civitaiToken = ''
      hfToken = ''
      savedMessage = imageCheckpointStatus.started ? 'Krea 체크포인트 준비를 시작했습니다.' : '이미 모델 준비가 진행 중입니다.'
      await refreshImageCheckpointStatus()
    } catch (e) {
      error = e.message
    } finally {
      preparingImageCheckpoints = false
    }
  }

  async function convertImageCheckpointsNVFP4() {
    const variants = Object.entries(nvfp4Selection).filter(([, selected]) => selected).map(([id]) => id)
    if (!variants.length) {
      error = '변환할 Krea 체크포인트를 하나 이상 선택하세요.'
      return
    }
    convertingImageCheckpoints = true
    error = ''
    savedMessage = ''
    try {
      imageCheckpointStatus = await api.convertImageCheckpointsNVFP4(civitaiToken.trim(), variants, removeBF16Sources)
      civitaiToken = ''
      savedMessage = imageCheckpointStatus.started ? 'BF16 다운로드와 NVFP4 변환을 시작했습니다.' : '이미 변환 작업이 진행 중입니다.'
      await refreshImageCheckpointStatus()
    } catch (e) {
      error = e.message
    } finally {
      convertingImageCheckpoints = false
    }
  }

  async function prepareVideoModels() {
    preparingVideoModels = true
    error = ''
    savedMessage = ''
    try {
      videoModelStatus = await api.prepareVideoModels(hfToken.trim())
      hfToken = ''
      savedMessage = videoModelStatus.ready
        ? 'LTX 영상 모델이 이미 준비되어 있습니다.'
        : '모델 준비를 시작했습니다. 이 화면에서 진행 상태를 확인할 수 있습니다.'
      await refreshVideoModelStatus()
    } catch (e) {
      error = e.message
    } finally {
      preparingVideoModels = false
    }
  }

  async function saveDownloadCredentials() {
    if (!civitaiToken.trim() && !hfToken.trim()) return
    savingDownloadCredentials = true
    error = ''
    savedMessage = ''
    try {
      await api.saveLoraTokens(civitaiToken.trim(), hfToken.trim())
      civitaiToken = ''
      hfToken = ''
      savedMessage = '다운로드 인증 정보를 저장했습니다.'
      await Promise.all([refreshImageCheckpointStatus(), refreshVideoModelStatus()])
    } catch (e) {
      error = e.message
    } finally {
      savingDownloadCredentials = false
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
      kreaOptions = {
        ...kreaOptions,
        checkpoint: value.image.default_checkpoint || 'official',
        sampling_preset: samplingPresetForCheckpoint(value.image.default_checkpoint || 'official', kreaOptions.sampling_preset),
        ...((value.image.default_checkpoint || 'official') === 'official' ? {} : { filter_mode: 'off', filter_strength: 0 })
      }
    }).catch((e) => error = e.message)
    refreshUserLoras()
    refresh()
    refreshSystemUsage()
    refreshVideoModelStatus()
    refreshImageCheckpointStatus()
    const timer = setInterval(refresh, 1500)
    const systemTimer = setInterval(refreshSystemUsage, 5000)
    const modelTimer = setInterval(() => {
      if (tab === 'settings') {
        refreshVideoModelStatus()
        refreshImageCheckpointStatus()
      }
    }, 3000)
    const progressTimer = setInterval(() => { progressClock = Date.now() }, 1000)
    return () => { clearInterval(timer); clearInterval(systemTimer); clearInterval(modelTimer); clearInterval(progressTimer) }
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
      if (module === 'depth') setIdentityPreserveItems(identityPreserveItems.filter((id) => id !== 'pose' && id !== 'composition'))
      if (module === 'identity' && imageForm.width * imageForm.height > 2 * 1024 * 1024) {
        imageMegapixels = 2
        imageResolutionMode = 'smart'
        applySmartResolution()
        imageCloneMessage = 'Identity 편집은 최대 2MP이므로 이미지 크기를 고해상도 2MP로 조정했습니다.'
      }
      if (module === 'identity' && Number(kreaOptions.steps) < 10) {
        kreaOptions = { ...kreaOptions, steps: 10 }
      }
      if (module === 'identity') {
        // The published Identity Edit graph has no filter-vector LoRA. Keep the
        // reliable baseline when the module is first enabled; users may still
        // turn filtering back on afterwards for an intentional comparison.
        kreaOptions = { ...kreaOptions, filter_mode: 'off', filter_strength: 0 }
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

  function addIdentityReferenceObjects(incoming) {
    const normalized = [...incoming].filter(Boolean).map((image) => {
      if (image.server || image.preview) return image
      return { file: image, name: image.name, preview: URL.createObjectURL(image), server: false }
    })
    const combined = [...kreaIdentityReferences, ...normalized]
    combined.slice(3).forEach((image) => { if (image.preview?.startsWith('blob:')) URL.revokeObjectURL(image.preview) })
    kreaIdentityReferences = combined.slice(0, 3)
    kreaIdentityReference = kreaIdentityReferences[0] || null
    kreaIdentityReferencePreview = kreaIdentityReference?.preview || kreaIdentityReference?.url || ''
  }

  function addIdentityReferences(files) {
    addIdentityReferenceObjects([...files].filter((file) => file.type.startsWith('image/')))
  }

  function clearIdentityReferences() {
    kreaIdentityReferences.forEach((image) => { if (image.preview?.startsWith('blob:')) URL.revokeObjectURL(image.preview) })
    kreaIdentityReferences = []
    kreaIdentityReference = null
    kreaIdentityReferencePreview = ''
  }

  function removeIdentityReference(index) {
    const removed = kreaIdentityReferences[index]
    if (removed?.preview?.startsWith('blob:')) URL.revokeObjectURL(removed.preview)
    kreaIdentityReferences = kreaIdentityReferences.filter((_, i) => i !== index)
    kreaIdentityReference = kreaIdentityReferences[0] || null
    kreaIdentityReferencePreview = kreaIdentityReference?.preview || kreaIdentityReference?.url || ''
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
      clearIdentityReferences()
      if (image) addIdentityReferenceObjects([image.server ? { ...image, preview } : { file: image.file || image, name: image.name, preview, server: false }])
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
    if (target === 'sequenceBase') {
      imageSequenceBase = { ...image, jobID: job.id, prompt: job.prompt || '' }
      imageSequencePrompts = imageSequencePrompts.map((prompt, index) => index === 0 ? (job.prompt || prompt) : prompt)
    } else if (target === 'vision' || target === 'styleReference') addKreaRefObjects(target, [image])
    else if (target === 'identityReference') addIdentityReferenceObjects([image])
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
      file.posePrompt = item.library === 'pose' ? (item.prompt || item.name || '') : ''
      if (target === 'vision' || target === 'styleReference') addKreaRefs(target, [file])
      else if (target === 'identityReference') addIdentityReferences([file])
      else setKreaImage(target, file)
      presetImagePickerTarget = ''
    } catch (cause) {
      error = `프리셋 이미지를 불러오지 못했습니다: ${cause.message}`
    }
  }

  const remoteImageTitles = {
    identity: '원본 이미지를 URL에서 가져오기',
    identityReference: '보조 참조를 URL에서 가져오기',
    depth: '자세·구도 이미지를 URL에서 가져오기',
    nk2e: '편집·윤곽 이미지를 URL에서 가져오기',
    anypaint: '부분 수정·확장 원본을 URL에서 가져오기',
    styleReference: '스타일 참조를 URL에서 추가',
    vision: '내용·구도 참조를 URL에서 추가'
  }

  function useRemoteModuleImage(file) {
    const target = remoteImageTarget
    if (!file || !target) return
    if (target === 'vision' || target === 'styleReference') addKreaRefs(target, [file])
    else if (target === 'identityReference') addIdentityReferences([file])
    else setKreaImage(target, file)
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
    subtitleModal = null
    const details = [
      `${job.params?.width || '—'}×${job.params?.height || '—'}`,
      formatDuration(videoJobDuration(job)),
      `${job.params?.fps || '—'} fps`,
      job.params?.seed >= 0 ? `seed ${job.params.seed}` : ''
    ].filter(Boolean)
    videoModal = {
      src: job.output_url,
      title: '생성 영상',
      detail: details.join(' · '),
      prompt: job.prompt,
      thumbnails: {
        url: `/api/jobs/${job.id}/video-preview.jpg`,
        number: 50,
        column: 10,
        width: 160,
        height: 90,
        scale: 1
      }
    }
  }

  function showSubtitle(job) {
    if (!job || (!job.media_url && !job.params?.text && !job.outputs && !job.output_url)) return
    videoModal = null
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

  function showAudio(job) {
    if (!job?.output_url) return
    audioModal = {
      src: job.output_url,
      detail: [job.params?.speaker, job.params?.language, job.params?.seed >= 0 ? `seed ${job.params.seed}` : ''].filter(Boolean).join(' · '),
      prompt: job.prompt,
      instructions: job.params?.instructions || ''
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

  function implicitModulePrompt() {
    const identityActions = {
      restage: 'Place the same person in a new scene and apply the selected pose and composition',
      sheet: 'Create a clean 2x2 character sheet on a plain background: front view upper-left, three-quarter view upper-right, left profile lower-left, and back view lower-right',
      tryon: 'Use the complete outfit shown in the supporting clothing reference',
      replace: 'Replace only the selected object or region using the supporting reference',
      faceSwap: 'Replace only the face of the person in Image One with the face from Image Two',
      headSwap: 'Replace the entire head of the person in Image One with the head from Image Two',
      personSwap: 'Replace the entire person in Image One with the person from Image Two'
    }
    if (kreaModules.identity && identityActions[identityPreset]) {
      const poseInstruction = kreaModules.depth
        ? '. Apply the pose, body orientation, framing, and camera viewpoint from the pose reference'
        : ''
      return `${identityActions[identityPreset]}${poseInstruction}`
    }
    if (kreaModules.identity && kreaModules.depth) {
      return 'Keep the original person and apply the pose, body orientation, framing, and camera viewpoint from the pose reference'
    }
    if (kreaModules.depth) return 'Create a coherent image that follows the supplied pose, depth structure, composition, and camera viewpoint'
    if (kreaModules.vision) return 'Create a coherent image using the subject, content, and composition from the reference images'
    if (kreaModules.styleReference) return 'Create a coherent image using the visual style, color, lighting, and texture from the style reference images'
    if (kreaModules.nk2e) return 'Create a coherent edited image that follows the supplied structure and preserves natural visual detail'
    if (kreaModules.anypaint && kreaAnyPaintMask) return 'Regenerate the masked area naturally and blend it seamlessly with the unchanged original image'
    if (isPureOutpaint()) return 'Extend the original image naturally while preserving its subjects, style, lighting, perspective, and visual continuity'
    return ''
  }

  function identityHasExtraUserPrompt() {
    const entered = imageForm.prompt.trim()
    if (!entered) return false
    // The module's own visible fallback is not an additional edit request.
    // Treating it as one makes Gemma merge the outfit and pose commands twice,
    // which weakens Identity Edit enough to preserve the source clothing.
    return entered !== implicitModulePrompt().trim()
  }

  function rawImagePrompt() {
    let change = imageForm.prompt.trim() || implicitModulePrompt()
    if (!kreaModules.identity) return change
	if (identityPreset === 'tryon') return change
    // Older jobs may contain the former Change/Preserve envelope. Unwrap it,
    // but do not build a new one: Krea Identity Edit follows short, separate
    // natural-language instructions much more reliably than a long policy-like
    // contract.
    while (/^change\s*:/i.test(change)) change = change.replace(/^change\s*:\s*/i, '').trim()
    const preserveAt = change.search(/(?:^|\n)preserve\s*:/i)
    if (preserveAt >= 0) change = change.slice(0, preserveAt).trim()
    const lines = [change]
    if (kreaModules.depth && !/(?:pose|posture|body orientation|자세|포즈|구도)/i.test(change)) {
      lines.push('The person now holds the same pose shown in the pose reference.')
    }
    if (identityPreserveCustom.trim()) lines.push(`Keep ${identityPreserveCustom.trim()} unchanged.`)
    return lines.filter(Boolean).join('\n')
  }

  function setIdentityPreserveItems(items) {
    identityPreserveItems = identityPreserveCatalog.map((item) => item.id).filter((id) => items.includes(id))
    resetImageEnhancement()
  }

  function toggleIdentityPreserveItem(id) {
    if (kreaModules.depth && (id === 'pose' || id === 'composition')) return
    setIdentityPreserveItems(identityPreserveItems.includes(id) ? identityPreserveItems.filter((item) => item !== id) : [...identityPreserveItems, id])
  }

  function identityPreserveDefaults(preset) {
    const defaults = {
      '': defaultIdentityPreserveItems,
      restage: ['identity', 'face', 'hair', 'body', 'clothing', 'untouched'],
      sheet: ['identity', 'face', 'hair', 'body', 'clothing'],
      faceSwap: ['hair', 'body', 'clothing', 'pose', 'background', 'lighting', 'composition', 'untouched'],
      headSwap: ['body', 'clothing', 'pose', 'background', 'lighting', 'composition', 'untouched'],
      personSwap: ['pose', 'background', 'lighting', 'composition', 'untouched'],
      tryon: ['identity', 'face', 'hair', 'body', 'pose', 'background', 'lighting', 'composition', 'untouched'],
      replace: defaultIdentityPreserveItems
    }
    const selected = [...(defaults[preset] || defaults[''])]
    return kreaModules.depth ? selected.filter((id) => id !== 'pose' && id !== 'composition') : selected
  }

  function applyIdentityPreset(value) {
    identityPreset = value
    if (!(identityPresetUI[value] || identityPresetUI['']).showSecondary) setKreaImage('identityReference', null)
    const presets = {
      restage: 'Place the same person in a new scene and pose as described',
      sheet: 'Create a clean 2x2 character sheet on a plain background: front view upper-left, three-quarter view upper-right, left profile lower-left, and back view lower-right',
      tryon: '',
      replace: 'Replace only the selected object or region as described',
      faceSwap: 'Replace only the face of the person in Image One with the face from Image Two',
      headSwap: 'Replace the entire head of the person in Image One with the head from Image Two',
      personSwap: 'Replace the entire person in Image One with the person from Image Two'
    }
    setIdentityPreserveItems(identityPreserveDefaults(value))
    identityPreserveCustom = ''
    if (!(value in presets)) return
    imageForm.prompt = presets[value]
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
    if (!rawImagePrompt().trim()) return '무엇을 만들지 프롬프트를 입력하세요.'
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
    if (maskEditorMode || cannyEditorOpen || imageModal || runtimeInfoOpen || recentImagePickerTarget || presetImagePickerTarget || remoteImageTarget) return
    featureModulesOpen = false
  }

  function looksLikeStructuredPrompt(value = imageForm.prompt) {
    const text = value.trim()
    if (!text || (text[0] !== '{' && text[0] !== '[')) return false
    try { JSON.parse(text); return true } catch { return false }
  }

  function imageEnhancementActive(enabled = imageEnhanceEnabled, prompt = rawImagePrompt()) {
	if (kreaModules.identity && identityPreset === 'tryon' && !identityHasExtraUserPrompt()) return false
    return enabled && prompt.trim() !== '' && !looksLikeStructuredPrompt(prompt)
  }

  function imageEnhancementCurrent(enhanced = imageEnhancedPrompt, source = imageEnhancedSource, current = rawImagePrompt()) {
    return enhanced.trim() !== '' && source === current
  }

  // These values are rendered in the submit controls. Keep their dependencies
  // explicit so nested form bindings immediately update the button state.
  $: imageEnhancementIsActive = imageEnhancementActive(imageEnhanceEnabled, rawImagePrompt())
  $: activeKreaModuleLabels = Object.entries(kreaModules).filter(([, enabled]) => enabled).map(([name]) => kreaModuleLabels[name])
  $: kreaModuleMessage = (
    kreaModules, identityPreset, kreaIdentityImage, kreaIdentityReference, kreaDepthImage, kreaVisionImages, kreaStyleReferenceImages,
    kreaStyleSelections, userLoraSelections, kreaNK2EImage, kreaAnyPaintImage, kreaAnyPaintMask, kreaOptions,
    kreaModuleDisabledReason()
  )
  $: imageEnhancementIsCurrent = (
    imageForm, identityPreserveItems, identityPreserveCustom, kreaModules,
    imageEnhancementCurrent(imageEnhancedPrompt, imageEnhancedSource, rawImagePrompt())
  )
  $: if (imageCheckpointStatus?.identity_runtime && !imageCheckpointStatus.identity_runtime.heretic_ready && kreaOptions.identity_encoder === 'heretic') {
    kreaOptions = { ...kreaOptions, identity_encoder: 'default' }
  }
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

  function resetImageCreation() {
    clearRefs()
    setKreaImage('identity', null)
    setKreaImage('identityReference', null)
    setKreaImage('identityMask', null)
    setKreaImage('strictMask', null)
    setKreaImage('depth', null)
    setKreaImage('nk2e', null)
    setKreaImage('anypaint', null)
    setKreaImage('anypaintMask', null)
    clearKreaRefs('vision')
    clearKreaRefs('styleReference')
    clearImageSequenceMasks()
    imageSequenceBase = null
    imageSequencePrompts = ['', '']
    imageSequenceRegions = ['all', 'all']
    imageSequenceStrength = 0.8
    imageSequenceOpen = false
    kreaModules = { identity: false, depth: false, style: false, userLora: false, vision: false, styleReference: false, nk2e: false, anypaint: false }
    kreaStyleSelections = [{ name: 'retroanime', strength: 1 }]
    userLoraSelections = []
    const checkpoint = config?.image?.default_checkpoint || 'official'
    kreaOptions = {
      checkpoint,
    identity_strength: 1, ref_boost: 4, source_ref_boost: 1, grounding_px: 768, steps: 8,
      identity_model: 'convrot', identity_encoder: 'heretic',
      sampling_preset: samplingPresetForCheckpoint(checkpoint, 'default'),
      depth_strength: 0.8,
      vision_mode: 'descriptor', vision_megapixels: 1, style_reference_strength: 1,
      nk2e_mode: 'edit', nk2e_strength: 0.7, vae_mode: 'default', identity_fit_mode: 'fit',
      strict_mask_grow: 0, strict_mask_feather: 0,
      outpaint_left: 0, outpaint_top: 0, outpaint_right: 0, outpaint_bottom: 0,
      anypaint_strength: 1, anypaint_boundary_redraw_px: 32,
      filter_mode: checkpoint === 'official' ? 'balanced' : 'off', filter_strength: checkpoint === 'official' ? 1 : 0,
      prompt_enhancer: Boolean(config?.image?.default_prompt_enhancer), prompt_enhancer_strength: 1, prompt_text_scale: 1.75
    }
    imageForm = { prompt: '', width: 1024, height: 1024, seed: -1, mode: 'create' }
    imageResolutionMode = 'smart'
    imageAspectRatio = '1:1'
    imageMegapixels = 1
    applySmartResolution()
    imageEnhanceEnabled = config?.prompt_enhancement?.default_enabled ?? true
    filterPromptPreset = ''
    parentImageJobID = ''
    identityPreset = ''
    identityPreserveItems = [...defaultIdentityPreserveItems]
    identityPreserveCustom = ''
    depthPoseID = ''
    nk2ePoseID = ''
    kreaNK2EPreprocessed = false
    imageCloneMessage = ''
    resetImageEnhancement()
  }

  async function enhanceImagePrompt() {
    const original = rawImagePrompt()
    if (!original || looksLikeStructuredPrompt(original)) return
    enhancingPrompt = true; error = ''
    try {
      const form = new FormData()
      form.append('prompt', original)
      form.append('mode', kreaModules.identity && kreaModules.depth ? 'edit_control' : kreaModules.identity ? 'edit' : kreaModules.anypaint ? 'paint' : (kreaModules.depth || kreaModules.nk2e) ? 'control' : 't2i')
	  if (kreaModules.identity) {
		form.append('identity_preset', identityPreset)
		form.append('identity_preserve_items', JSON.stringify(identityPreserveItems))
	  }
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
    const multiple = kreaModules.anypaint ? 16 : 8
    imageForm.width = Math.min(2048, Math.max(256, Math.round(width / multiple) * multiple))
    imageForm.height = Math.min(2048, Math.max(256, Math.round(height / multiple) * multiple))
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
    if (promptExamplesTarget === 'video') {
      const currentPrompt = videoForm.prompt.trimEnd()
      videoPromptPreset = preset.id
      videoForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${preset.prompt}` : preset.prompt
      promptExamplesOpen = false
      resetVideoEnhancement()
      return
    }
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

  function isMoodyCheckpoint(checkpoint) {
    return checkpoint?.startsWith('moody-')
  }

  function checkpointVisible(checkpoint) {
    if (checkpoint === 'official') return true
    const visible = settings?.image?.visible_checkpoints
    return !Array.isArray(visible) || visible.includes(checkpoint)
  }

  function setCheckpointVisible(checkpoint, visible) {
    if (!settings?.image || checkpoint === 'official') return
    const next = new Set(settings.image.visible_checkpoints || checkpointDisplayChoices.map(([id]) => id))
    if (visible) next.add(checkpoint)
    else next.delete(checkpoint)
    settings.image.visible_checkpoints = ['official', ...checkpointDisplayChoices.map(([id]) => id).filter((id) => next.has(id))]
    if (!visible && settings.image.default_checkpoint === checkpoint) settings.image.default_checkpoint = 'official'
    if (!visible && kreaOptions.checkpoint === checkpoint) selectKreaCheckpoint('official')
    settings = { ...settings, image: { ...settings.image } }
  }

  function displayCheckpointReady(checkpoint) {
    if (checkpoint === 'ray-v2-nvfp4' || checkpoint === 'ray-v4-nvfp4') {
      const source = checkpoint.replace('-nvfp4', '')
      return Boolean(imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === source)?.validated)
    }
    return Boolean(imageCheckpointStatus?.variants?.find((item) => item.id === checkpoint)?.ready)
  }

  function samplingPresetForCheckpoint(checkpoint, current = 'default') {
    if (isMoodyCheckpoint(checkpoint)) return 'moody'
    return current === 'moody' ? 'default' : current
  }

  function selectKreaCheckpoint(checkpoint) {
	if (checkpoint === 'identity-convrot') {
		kreaOptions = { ...kreaOptions, identity_model: 'convrot', sampling_preset: 'default', filter_mode: 'off', filter_strength: 0 }
		return
	}
    kreaOptions = {
      ...kreaOptions,
      checkpoint,
	  identity_model: kreaModules.identity ? 'selected' : kreaOptions.identity_model,
      sampling_preset: samplingPresetForCheckpoint(checkpoint, kreaOptions.sampling_preset),
      ...(checkpoint === 'official' ? {} : { filter_mode: 'off', filter_strength: 0 })
    }
  }

  function selectedKreaCheckpoint() {
	return kreaModules.identity && kreaOptions.identity_model === 'convrot' ? 'identity-convrot' : kreaOptions.checkpoint
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
      identityPreset = params.identity_preset || ''
      identityPreserveItems = Array.isArray(params.identity_preserve_items)
        ? identityPreserveCatalog.map((item) => item.id).filter((id) => params.identity_preserve_items.includes(id))
        : identityPreserveDefaults(identityPreset)
      if (kreaModules.depth) identityPreserveItems = identityPreserveItems.filter((id) => id !== 'pose' && id !== 'composition')
      identityPreserveCustom = params.identity_preserve_custom || ''
      kreaOptions = {
        ...kreaOptions,
        checkpoint: params.checkpoint || 'official',
        identity_strength: params.identity_strength !== undefined ? Number(params.identity_strength) : 1,
        identity_model: params.identity_model || 'convrot',
        identity_encoder: params.identity_encoder || 'heretic',
        ref_boost: params.ref_boost !== undefined ? Number(params.ref_boost) : 4,
        source_ref_boost: params.source_ref_boost !== undefined ? Number(params.source_ref_boost) : 1,
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
        ,sampling_preset: params.sampling_preset || (params.sampler === 'er_sde' ? 'detail' : params.sampler === 'euler_ancestral' ? 'moody' : 'default')
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
      else if (input.role === 'identity_reference') addIdentityReferenceObjects([input])
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
    imageForm.prompt = ''; identityPreserveItems = [...defaultIdentityPreserveItems]; identityPreserveCustom = ''
    resetImageEnhancement(); mobileImagePane = 'create'; window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  async function cloneImageJob(job, part) {
    cloningImageJob = `${job.id}:${part}`
    imageCloneMessage = ''
    error = ''
    try {
      if (part === 'all') parentImageJobID = ''
      if (part === 'prompt' || part === 'all') cloneImagePrompt(job)
      if (part === 'settings' || part === 'all') cloneImageSettings(job)
      let inputCount = null
      if (part === 'references' || part === 'all') inputCount = await cloneImageReferences(job)
      const labels = { prompt: '프롬프트', references: '참조 이미지', settings: '설정', all: '전체 작업' }
      imageCloneMessage = inputCount === 0 && part === 'references'
        ? '이 작업에는 불러올 참조 이미지가 없습니다.'
        : `${labels[part]}을 새 작업 작성란으로 불러왔습니다${inputCount ? ` · 이미지 ${inputCount}장` : ''}. 기존 결과는 변경되지 않습니다.`
      mobileImagePane = 'create'
      window.scrollTo({ top: 0, behavior: 'smooth' })
    } catch (e) {
      error = e.message
    } finally {
      cloningImageJob = ''
    }
  }

  function imageSequenceBlockedMessage() {
    if (imageForm.mode !== 'create') return '연속 생성은 새 이미지 생성에서만 사용할 수 있습니다.'
    const incompatible = [
      [kreaModules.identity, '원본 수정'], [kreaModules.depth, '자세·구도'],
      [kreaModules.vision, '내용·구도 참조'], [kreaModules.styleReference, '스타일 참조'],
      [kreaModules.nk2e, '편집·윤곽'], [kreaModules.anypaint, '부분 수정·확장']
    ].filter(([enabled]) => enabled).map(([, label]) => label)
    if (incompatible.length) return `${incompatible.join(' · ')} 모듈을 끈 뒤 사용할 수 있습니다.`
    const moduleReason = kreaModuleDisabledReason()
    if (moduleReason) return moduleReason
    if (imageForm.width * imageForm.height > 2 * 1024 * 1024) return '연속 장면은 Identity Edit를 사용하므로 2MP 이하가 필요합니다.'
    return ''
  }

  function openImageSequence() {
    imageSequencePrompts = [rawImagePrompt(), '']
    imageSequenceRegions = ['all', 'all']
    clearImageSequenceMasks()
    imageSequenceBase = null
    imageSequenceOpen = true
  }

  function applyRobotSequenceExample() {
    imageSequencePrompts = [
      'Wide full-body shot of a small friendly orange robot centered beside a blue armchair in a softly lit modern studio, with clear empty space around its entire body from antenna tips to both feet. It has exactly two arms, one continuous orange metal faceplate, exactly two round black recessed eyes, one small curved black smile, and two thin antennae, with no display screen. Both arms rest naturally at its sides, clean 3D animated film style.',
      "Move the robot's right arm, which is on the left side of the image, from its side into a raised friendly waving pose. Replace the old lowered arm position completely; show this arm exactly once in its new raised position. Preserve the exact face, head, left arm, body, chair, camera, lighting, and background unchanged.",
      "Move the same raised right arm on the left side of the image down to a halfway-lowered position, as the next moment of the wave. Replace its previous raised position completely; show this arm exactly once in the new position. Preserve the exact face, head, left arm, body, chair, camera, lighting, and background unchanged."
    ]
    imageSequenceRegions = ['all', 'left-arm', 'left-arm']
    clearImageSequenceMasks()
    imageSequenceStrength = 0.65
  }

  function addImageSequenceScene() {
    if (imageSequencePrompts.length >= 6) return
    imageSequencePrompts = [...imageSequencePrompts, '']
    imageSequenceRegions = [...imageSequenceRegions, 'all']
    imageSequenceMasks = [...imageSequenceMasks, null]
    imageSequenceMaskPreviews = [...imageSequenceMaskPreviews, '']
  }

  function removeImageSequenceScene(index) {
    if (imageSequencePrompts.length <= 2) return
    imageSequencePrompts = imageSequencePrompts.filter((_, itemIndex) => itemIndex !== index)
    imageSequenceRegions = imageSequenceRegions.filter((_, itemIndex) => itemIndex !== index)
    if (imageSequenceMaskPreviews[index]) URL.revokeObjectURL(imageSequenceMaskPreviews[index])
    imageSequenceMasks = imageSequenceMasks.filter((_, itemIndex) => itemIndex !== index)
    imageSequenceMaskPreviews = imageSequenceMaskPreviews.filter((_, itemIndex) => itemIndex !== index)
  }

  function updateImageSequencePrompt(index, value) {
    imageSequencePrompts = imageSequencePrompts.map((prompt, itemIndex) => itemIndex === index ? value : prompt)
  }

  function updateImageSequenceRegion(index, value) {
    if (imageSequenceMaskPreviews[index]) URL.revokeObjectURL(imageSequenceMaskPreviews[index])
    imageSequenceMasks = imageSequenceMasks.map((mask, itemIndex) => itemIndex === index ? null : mask)
    imageSequenceMaskPreviews = imageSequenceMaskPreviews.map((preview, itemIndex) => itemIndex === index ? '' : preview)
    imageSequenceRegions = imageSequenceRegions.map((region, itemIndex) => itemIndex === index ? value : region)
    imageSequenceRegionPicker = -1
  }

  function clearImageSequenceMasks() {
    imageSequenceMaskPreviews.forEach((preview) => { if (preview) URL.revokeObjectURL(preview) })
    imageSequenceMasks = imageSequencePrompts.map(() => null)
    imageSequenceMaskPreviews = imageSequencePrompts.map(() => '')
    imageSequenceMaskEditorIndex = -1
  }

  function useImageSequenceMask(file) {
    const index = imageSequenceMaskEditorIndex
    if (index < 1 || !file) return
    if (imageSequenceMaskPreviews[index]) URL.revokeObjectURL(imageSequenceMaskPreviews[index])
    imageSequenceMasks = imageSequenceMasks.map((mask, itemIndex) => itemIndex === index ? file : mask)
    imageSequenceMaskPreviews = imageSequenceMaskPreviews.map((preview, itemIndex) => itemIndex === index ? URL.createObjectURL(file) : preview)
    imageSequenceRegions = imageSequenceRegions.map((region, itemIndex) => itemIndex === index ? 'custom' : region)
    imageSequenceMaskEditorIndex = -1
  }

  const imageSequenceRegionOptions = [
    { id: 'all', label: '전체 수정', description: '이미지 전체를 편집' },
    { id: 'left', label: '화면 왼쪽', description: '왼쪽 절반을 넓게 수정' },
    { id: 'right', label: '화면 오른쪽', description: '오른쪽 절반을 넓게 수정' },
    { id: 'upper', label: '화면 상단', description: '위쪽 절반을 넓게 수정' },
    { id: 'lower', label: '화면 하단', description: '아래쪽 절반을 넓게 수정' },
    { id: 'left-arm', label: '화면 왼쪽 팔', description: '왼쪽 팔의 이전·새 위치를 포함' },
    { id: 'right-arm', label: '화면 오른쪽 팔', description: '오른쪽 팔의 이전·새 위치를 포함' }
  ]

  function imageSequenceRegionOption(region) {
    return imageSequenceRegionOptions.find((option) => option.id === region) || imageSequenceRegionOptions[0]
  }

  async function generateImage(sequencePrompts = null) {
    const isSequence = Array.isArray(sequencePrompts)
    if (!isSequence && imageEnhancementActive() && !imageEnhancementCurrent()) {
      await enhanceImagePrompt()
      if (!imageEnhancementCurrent()) return
    }
    busy = true; error = ''
    try {
      const form = new FormData()
      const firstPrompt = isSequence ? sequencePrompts[0].trim() : rawImagePrompt()
      const userPrompt = isSequence
        ? sequencePrompts[0].trim()
        : (imageForm.prompt.trim() || implicitModulePrompt())
      Object.entries(imageForm).forEach(([key, value]) => form.append(key, key === 'prompt' ? (isSequence ? firstPrompt : imageEnhancementActive() ? imageEnhancedPrompt : firstPrompt) : value))
      // Keep the human instruction separate from the generated Change/Preserve
      // contract. Gallery display and prompt cloning must restore this text, not
      // leak the internal Identity Edit envelope back into the prompt field.
      form.append('original_prompt', userPrompt)
      if (isSequence) {
        form.append('sequence_prompts', JSON.stringify(sequencePrompts.map((prompt) => prompt.trim())))
        form.append('sequence_regions', JSON.stringify(imageSequenceRegions))
        form.append('sequence_identity_strength', imageSequenceStrength)
        if (imageSequenceBase?.jobID) form.append('sequence_base_job_id', imageSequenceBase.jobID)
        imageSequenceMasks.forEach((mask, index) => { if (index > 0 && mask) form.append(`sequence_mask_${index}`, mask) })
      }
      if (parentImageJobID) form.append('parent_job_id', parentImageJobID)
      if (imageForm.mode === 'create') {
        form.append('steps', kreaOptions.steps)
        form.append('checkpoint', kreaOptions.checkpoint)
        form.append('filter_mode', kreaOptions.filter_mode)
        form.append('filter_strength', kreaOptions.filter_strength)
        form.append('prompt_enhancer', kreaOptions.prompt_enhancer)
        form.append('prompt_enhancer_strength', kreaOptions.prompt_enhancer_strength)
        form.append('prompt_text_scale', kreaOptions.prompt_text_scale)
        form.append('sampling_preset', kreaOptions.sampling_preset)
        if (kreaModules.identity) {
          form.append('identity_preset', identityPreset)
		  if (identityPreset === 'tryon') {
			form.append('identity_auto_prompt', 'true')
			form.append('identity_user_prompt', identityHasExtraUserPrompt() ? 'true' : 'false')
		  }
          form.append('identity_preserve_items', JSON.stringify(identityPreserveItems))
          form.append('identity_preserve_custom', identityPreserveCustom)
          appendImageInput(form, 'identity_image', 'reuse_identity_image', kreaIdentityImage)
          kreaIdentityReferences.forEach((image) => appendImageInput(form, 'identity_reference', 'reuse_identity_reference', image))
          appendImageInput(form, 'identity_mask', 'reuse_identity_mask', kreaIdentityMask)
          appendImageInput(form, 'strict_mask', 'reuse_strict_mask', kreaStrictMask)
          form.append('identity_strength', kreaOptions.identity_strength)
          form.append('ref_boost', kreaOptions.ref_boost)
          form.append('source_ref_boost', kreaOptions.source_ref_boost)
          form.append('grounding_px', kreaOptions.grounding_px)
          form.append('strict_mask_grow', kreaOptions.strict_mask_grow)
          form.append('strict_mask_feather', kreaOptions.strict_mask_feather)
          form.append('vae_mode', kreaOptions.vae_mode)
          form.append('identity_fit_mode', kreaOptions.identity_fit_mode)
		  form.append('identity_model', kreaOptions.identity_model)
		  form.append('identity_encoder', kreaOptions.identity_encoder)
        }
        if (kreaModules.depth) {
          appendImageInput(form, 'depth_image', 'reuse_depth_image', kreaDepthImage)
          form.append('depth_strength', kreaOptions.depth_strength)
          if (kreaDepthImage?.posePrompt) {
            form.append('depth_pose_prompt', kreaDepthImage.posePrompt)
            // Diagram-style pose presets first need a realistic person reference.
            // User uploads and generated images can go directly into Identity Edit source B.
            form.append('prepare_pose_reference', 'true')
          }
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
      imageSequenceOpen = false
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
    if (!job.output_url || upscalingImageJob) return
    upscalingImageJob = job.id; error = ''
    try {
      await api.upscaleImage(job.id, { scale: 2, seed: -1 })
      showNewestListPage('image')
      await refresh()
    } catch (e) { error = e.message }
    finally { upscalingImageJob = '' }
  }

  async function detailEnhanceImage(job) {
    if (!job.output_url || detailEnhancingImageJob) return
    detailEnhancingImageJob = job.id; error = ''
    try {
      await api.detailEnhanceImage(job.id, { strength: 1, seed: -1, vae: 'wan' })
      showNewestListPage('image')
      await refresh()
    } catch (e) { error = e.message }
    finally { detailEnhancingImageJob = '' }
  }

  function openGarmentExtractor(job = null) {
    garmentExtractorInitialJob = job?.output_url ? job : null
    garmentExtractorOpen = true
  }

  function openGarmentExtractorFromModal(jobID) {
    const job = imageJobs.find((item) => item.id === jobID)
    imageModal = null
    openGarmentExtractor(job)
  }

  async function submitGarmentExtraction(form) {
    error = ''
    await api.garmentExtract(form)
    mobileImagePane = 'results'
    showNewestListPage('image')
    await refresh()
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
      if (recognitionFileInput) recognitionFileInput.value = ''
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

  function updateRecognitionURL(event) {
    recognitionForm.url = event.currentTarget.value
    if (recognitionForm.url.trim()) {
      recognitionForm.source = 'url'
      recognitionFile = null
      if (recognitionFileInput) recognitionFileInput.value = ''
    }
    resetRecognitionOptions()
  }

  function updateRecognitionFile(event) {
    recognitionFile = event.currentTarget.files?.[0] || null
    if (!recognitionFile) return
    recognitionForm.source = 'file'
    recognitionForm.url = ''
    resetRecognitionOptions()
  }

  function clearRecognitionFile() {
    recognitionFile = null
    if (recognitionFileInput) recognitionFileInput.value = ''
    recognitionForm.source = 'url'
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
    busy = true; error = ''
    try {
      let effectivePrompt = videoEnhancedPrompt
      if (videoEnhancementActive() && !videoEnhancementCurrent()) {
        effectivePrompt = await enhanceVideoPrompt()
        if (!effectivePrompt) return
      }
      const form = new FormData()
      Object.entries(videoForm).forEach(([key, value]) => form.append(key, key === 'prompt' && videoEnhancementActive() ? effectivePrompt : value))
      form.append('num_frames', framesForDuration(videoDurationSeconds, videoForm.fps))
      form.append('original_prompt', videoForm.prompt)
      appendImageInput(form, 'start_image', 'reuse_start_image', videoImage)
      appendImageInput(form, 'end_image', 'reuse_end_image', videoEndImage)
      form.append('end_image_strength', videoEndStrength)
      const selectedKeyframes = videoKeyframes.filter((keyframe) => keyframe.image)
      form.append('keyframe_count', selectedKeyframes.length)
      selectedKeyframes.forEach((keyframe, index) => {
        appendImageInput(form, `keyframe_image_${index}`, `reuse_keyframe_image_${index}`, keyframe.image)
        form.append(`keyframe_time_${index}`, keyframe.time)
        form.append(`keyframe_strength_${index}`, keyframe.strength)
      })
      await api.video(form)
      showNewestListPage('video')
      videoForm.prompt = ''
      clearVideoConditioning()
      resetVideoEnhancement()
      await refresh()
      mobileVideoPane = 'results'
    } catch (e) { error = e.message } finally { busy = false }
  }

  function videoInputKey(image) {
    if (!image) return ''
    return image.server ? image.ref : `${image.name}:${image.size}:${image.lastModified}`
  }

  function videoImageKey() {
    return videoInputKey(videoImage)
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

  function normalizedVideoImage(image) {
    if (!image) return null
    if (image.server || image.file) return image
    return { file: image, name: image.name, size: image.size, lastModified: image.lastModified, preview: URL.createObjectURL(image), server: false }
  }

  function releaseVideoImage(image) {
    if (image?.preview?.startsWith('blob:')) URL.revokeObjectURL(image.preview)
  }

  function videoImagePreview(image) {
    return image?.preview || image?.url || ''
  }

  function setVideoConditionImage(target, image) {
    const normalized = normalizedVideoImage(image)
    videoPromptCreationMessage = ''
    if (target === 'start') {
      releaseVideoImage(videoImage)
      videoImage = normalized
      resetVideoEnhancement()
    } else if (target === 'end') {
      releaseVideoImage(videoEndImage)
      videoEndImage = normalized
    } else if (target.startsWith('keyframe:')) {
      const id = Number(target.split(':')[1])
      videoKeyframes = videoKeyframes.map((keyframe) => {
        if (keyframe.id !== id) return keyframe
        releaseVideoImage(keyframe.image)
        return { ...keyframe, image: normalized }
      })
    }
  }

  function addVideoKeyframe() {
    if (videoKeyframes.length >= 8) return
    videoPromptCreationMessage = ''
    const count = videoKeyframes.length + 1
    const time = Math.max(0.1, Math.round((videoDurationSeconds * count / (count + 1)) * 10) / 10)
    videoKeyframes = [...videoKeyframes, { id: nextVideoKeyframeID++, image: null, time, strength: 1 }]
  }

  function removeVideoKeyframe(id) {
    videoPromptCreationMessage = ''
    const removed = videoKeyframes.find((keyframe) => keyframe.id === id)
    releaseVideoImage(removed?.image)
    videoKeyframes = videoKeyframes.filter((keyframe) => keyframe.id !== id)
  }

  function updateVideoKeyframe(id, field, value) {
    videoPromptCreationMessage = ''
    videoKeyframes = videoKeyframes.map((keyframe) => keyframe.id === id ? { ...keyframe, [field]: Number(value) } : keyframe)
  }

  function clearVideoConditioning() {
    videoPromptCreationMessage = ''
    releaseVideoImage(videoImage)
    releaseVideoImage(videoEndImage)
    videoKeyframes.forEach((keyframe) => releaseVideoImage(keyframe.image))
    videoImage = null
    videoEndImage = null
    videoEndStrength = 1
    videoKeyframes = []
  }

  function resetVideoCreation() {
    clearVideoConditioning()
    nextVideoKeyframeID = 1
    videoForm = {
      prompt: '',
      width: config?.video?.default_width || 768,
      height: config?.video?.default_height || 512,
      fps: config?.video?.default_fps || 24,
      seed: -1,
      image_strength: 1
    }
    videoDurationSeconds = durationFromFrames(config?.video?.default_frames || 121, videoForm.fps)
    videoEnhanceEnabled = config?.prompt_enhancement?.default_enabled ?? true
    videoPromptPreset = ''
    resetVideoEnhancement()
  }

  function resetSpeechCreation() {
    speechForm = {
      text: '', instructions: '',
      language: config?.speech?.default_language || 'Korean',
      speaker: config?.speech?.default_speaker || 'Sohee',
      seed: -1
    }
  }

  function resetRecognitionCreation() {
    recognitionFile = null
    if (recognitionFileInput) recognitionFileInput.value = ''
    recognitionForm = {
      source: 'url', url: '', language: config?.recognition?.default_language || 'Auto', context: '',
      output_formats: [...(config?.recognition?.default_output_formats || ['srt', 'txt'])],
      translation_mode: config?.recognition?.default_translation_mode || 'none',
      target_language: config?.recognition?.default_translation_language || 'Korean',
      media_part: '', media_source: ''
    }
    recognitionOptions = null
  }

  function useRecentVideoImage(job) {
    if (!job?.output_url || !videoImagePickerTarget) return
    setVideoConditionImage(videoImagePickerTarget, {
      server: true,
      ref: `${job.id}:output:0`,
      url: job.output_url,
      name: `결과 ${job.id.slice(0, 8)}.png`,
      role: 'output'
    })
    videoImagePickerTarget = ''
  }

  function useRemoteVideoImage(file) {
    if (file && videoRemoteImageTarget) setVideoConditionImage(videoRemoteImageTarget, file)
  }

  function videoConditionTitle(target, suffix = ' 선택') {
    if (target === 'start') return `시작 이미지${suffix}`
    if (target === 'end') return `마지막 이미지${suffix}`
    const id = Number(String(target).split(':')[1])
    const index = videoKeyframes.findIndex((keyframe) => keyframe.id === id)
    return `키프레임 ${index >= 0 ? index + 1 : ''} 이미지${suffix}`
  }

  function videoConditioningDisabledReason() {
    const finalTime = (framesForDuration(videoDurationSeconds, videoForm.fps) - 1) / Number(videoForm.fps || 1)
    const occupied = new Set()
    for (let index = 0; index < videoKeyframes.length; index++) {
      const keyframe = videoKeyframes[index]
      if (!keyframe.image) continue
      const frame = Math.round(Number(keyframe.time) * Number(videoForm.fps))
      if (!(Number(keyframe.time) > 0 && Number(keyframe.time) < finalTime) || frame <= 0 || frame >= framesForDuration(videoDurationSeconds, videoForm.fps) - 1) return `키프레임 ${index + 1} 위치를 시작과 마지막 사이로 지정하세요.`
      if (occupied.has(frame)) return '같은 프레임 위치에 키프레임을 두 개 배치할 수 없습니다.'
      occupied.add(frame)
    }
    return ''
  }

  async function appendEnhancementImage(form, image) {
    if (!image) return
    if (!image.server) {
      form.append('image', image.file || image)
      return
    }
    const response = await fetch(image.url)
    if (!response.ok) throw new Error('시작 이미지를 읽지 못했습니다.')
    const blob = await response.blob()
    form.append('image', new File([blob], image.name || 'start-image.png', { type: blob.type || 'image/png' }))
  }

  async function enhanceVideoPrompt() {
    const original = videoForm.prompt.trim()
    if (!original) return ''
    enhancingPrompt = true; error = ''
    try {
      const form = new FormData()
      form.append('prompt', original)
      form.append('mode', videoImage ? 'i2v' : 't2v')
      await appendEnhancementImage(form, videoImage)
      const result = await api.enhancePrompt(form)
      videoEnhancedPrompt = result.enhanced_prompt
      videoEnhancedSource = original
      videoEnhancedImageKey = videoImageKey()
      return result.enhanced_prompt
    } catch (e) { error = e.message; return '' }
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
    if (!count || !confirm(`완료·실패·취소 작업 ${count}개와 저장 파일을 모두 삭제할까요?`)) return
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
    refreshVideoModelStatus()
    refreshImageCheckpointStatus()
  }

  function loadAssistantImage(src) {
    return new Promise((resolve, reject) => {
      const image = new Image()
      image.onload = () => resolve(image)
      image.onerror = () => reject(new Error('선택한 영상 이미지를 읽지 못했습니다.'))
      image.src = src
    })
  }

  async function videoAssistantVisualContext(message) {
    const text = String(message || '').toLowerCase()
    const cues = ['프롬프트', 'prompt', '시작', '마지막', '키프레임', '장면', '영상', '이어', '전환', '움직']
    if (!cues.some((cue) => text.includes(cue))) return null
    const conditions = [
      ...(videoImage ? [{ label: 'START', detail: '0초', image: videoImage }] : []),
      ...videoKeyframes.filter((item) => item.image).map((item, index) => ({ label: `KEYFRAME ${index + 1}`, detail: `${Number(item.time).toFixed(1)}초`, image: item.image })),
      ...(videoEndImage ? [{ label: 'END', detail: `${((framesForDuration(videoDurationSeconds, videoForm.fps) - 1) / videoForm.fps).toFixed(1)}초`, image: videoEndImage }] : [])
    ]
    if (!conditions.length) return null
    const loaded = []
    for (const condition of conditions) {
      try { loaded.push({ ...condition, bitmap: await loadAssistantImage(videoImagePreview(condition.image)) }) } catch (_) {}
    }
    if (!loaded.length) return null
    const columns = loaded.length === 1 ? 1 : 2
    const cellWidth = loaded.length === 1 ? 640 : 420
    const cellHeight = loaded.length === 1 ? 480 : 315
    const rows = Math.ceil(loaded.length / columns)
    const canvas = document.createElement('canvas')
    canvas.width = cellWidth * columns
    canvas.height = cellHeight * rows
    const context = canvas.getContext('2d')
    context.fillStyle = '#0d1115'
    context.fillRect(0, 0, canvas.width, canvas.height)
    loaded.forEach((item, index) => {
      const left = index % columns * cellWidth
      const top = Math.floor(index / columns) * cellHeight
      const padding = 6
      const sourceWidth = item.bitmap.naturalWidth || item.bitmap.width
      const sourceHeight = item.bitmap.naturalHeight || item.bitmap.height
      const scale = Math.min((cellWidth - padding * 2) / sourceWidth, (cellHeight - padding * 2) / sourceHeight)
      const width = sourceWidth * scale
      const height = sourceHeight * scale
      context.drawImage(item.bitmap, left + (cellWidth - width) / 2, top + (cellHeight - height) / 2, width, height)
      const label = `${item.label} · ${item.detail}`
      context.font = '700 18px sans-serif'
      const labelWidth = context.measureText(label).width + 22
      context.fillStyle = 'rgba(8,12,15,.86)'
      context.fillRect(left + 12, top + 12, labelWidth, 34)
      context.fillStyle = '#d9f5b7'
      context.fillText(label, left + 23, top + 35)
    })
    return {
      kind: 'video_conditioning',
      image_url: canvas.toDataURL('image/jpeg', 0.84),
      labels: loaded.map((item) => `${item.label} ${item.detail}`)
    }
  }

  async function createVideoPromptFromScenes() {
    if (creatingVideoPrompt || (!videoImage && !videoEndImage && !videoKeyframes.some((item) => item.image))) return
    creatingVideoPrompt = true
    videoPromptCreationMessage = ''
    error = ''
    try {
      const request = '현재 선택된 시작·키프레임·마지막 장면을 시간 순서로 모두 보고, 장면 사이를 자연스럽게 연결할 LTX 영상 프롬프트를 만들어줘. 피사체 동작, 카메라 움직임, 환경의 움직임과 장면 연속성을 구체적으로 포함해.'
      const visualContext = await videoAssistantVisualContext(request)
      if (!visualContext) throw new Error('분석할 장면 이미지를 읽지 못했습니다.')
      const result = await api.assistantChat({
        messages: [{ role: 'user', content: request }],
        state: assistantState,
        visual_context: visualContext
      })
      const prompt = result.actions?.find((action) => action.type === 'set_video' && action.prompt?.trim())?.prompt?.trim()
      if (!prompt) throw new Error('장면 분석 결과에서 영상 프롬프트를 얻지 못했습니다.')
      videoForm = { ...videoForm, prompt }
      resetVideoEnhancement()
      videoPromptCreationMessage = '장면 이미지를 분석해 프롬프트에 적용했습니다.'
    } catch (cause) {
      error = cause.message || '장면 프롬프트를 만들지 못했습니다.'
    } finally {
      creatingVideoPrompt = false
    }
  }

  $: assistantState = {
    tab,
    busy,
    image: { ...imageForm, enhance_enabled: imageEnhanceEnabled, modules: activeKreaModuleLabels },
    video: {
      ...videoForm,
      duration: videoDurationSeconds,
      enhance_enabled: videoEnhanceEnabled,
      has_start_image: Boolean(videoImage),
      has_end_image: Boolean(videoEndImage),
      keyframes: videoKeyframes.map((item) => ({ time: item.time, strength: item.strength, has_image: Boolean(item.image) }))
    },
    speech: { ...speechForm },
    recognition: {
      source: recognitionForm.source,
      has_source: Boolean(recognitionFile || recognitionForm.url.trim()),
      language: recognitionForm.language,
      context: recognitionForm.context,
      translation_mode: recognitionForm.translation_mode,
      target_language: recognitionForm.target_language
    },
    recent_images: pagedImageJobs.map((job, index) => ({
      index: (listPages.image - 1) * listPageSizes.image + index + 1,
      job_id: job.id,
      status: job.status,
      prompt: String(job.prompt || '').slice(0, 180)
    }))
  }

  function switchAssistantTab(nextTab, results = false) {
    if (nextTab === 'settings') openSettings()
    else tab = nextTab
    if (nextTab === 'image') mobileImagePane = results ? 'results' : 'create'
    if (nextTab === 'video') mobileVideoPane = results ? 'results' : 'create'
    if (nextTab === 'speech') mobileSpeechPane = results ? 'results' : 'create'
    if (nextTab === 'recognition') mobileRecognitionPane = results ? 'results' : 'create'
  }

  function applyAssistantActions(actions = []) {
    for (const action of actions) {
      if (action.type === 'navigate' && action.tab) switchAssistantTab(action.tab)
      else if (action.type === 'show_results' && action.tab) switchAssistantTab(action.tab, true)
      else if (action.type === 'open_modules') {
        switchAssistantTab('image')
        featureModulesOpen = true
      } else if (action.type === 'set_image') {
        switchAssistantTab('image')
        imageForm = {
          ...imageForm,
          ...(action.prompt != null ? { prompt: action.prompt } : {}),
          ...(action.width >= 256 ? { width: action.width } : {}),
          ...(action.height >= 256 ? { height: action.height } : {}),
          ...(action.seed != null ? { seed: action.seed } : {})
        }
        if (action.enhance_enabled != null) imageEnhanceEnabled = action.enhance_enabled
        imageResolutionMode = 'custom'
        resetImageEnhancement()
      } else if (action.type === 'set_video') {
        switchAssistantTab('video')
        videoForm = {
          ...videoForm,
          ...(action.prompt != null ? { prompt: action.prompt } : {}),
          ...(action.width >= 256 ? { width: action.width } : {}),
          ...(action.height >= 256 ? { height: action.height } : {}),
          ...(action.fps > 0 ? { fps: action.fps } : {}),
          ...(action.seed != null ? { seed: action.seed } : {})
        }
        if (action.duration > 0) videoDurationSeconds = action.duration
        if (action.enhance_enabled != null) videoEnhanceEnabled = action.enhance_enabled
        resetVideoEnhancement()
      } else if (action.type === 'set_speech') {
        switchAssistantTab('speech')
        speechForm = {
          ...speechForm,
          ...(action.text != null ? { text: action.text } : {}),
          ...(action.instructions != null ? { instructions: action.instructions } : {}),
          ...(action.language ? { language: action.language } : {}),
          ...(action.speaker ? { speaker: action.speaker } : {}),
          ...(action.seed != null ? { seed: action.seed } : {})
        }
      } else if (action.type === 'set_recognition') {
        switchAssistantTab('recognition')
        recognitionForm = {
          ...recognitionForm,
          ...(action.context != null ? { context: action.context } : {}),
          ...(action.language ? { language: action.language } : {}),
          ...(action.translation_mode ? { translation_mode: action.translation_mode } : {}),
          ...(action.target_language ? { target_language: action.target_language } : {})
        }
      } else if (action.type === 'set_module' && action.module in kreaModules) {
        switchAssistantTab('image')
        const desired = action.enabled !== false
        if (Boolean(kreaModules[action.module]) !== desired) toggleKreaModule(action.module)
        if (desired && action.module === 'identity' && action.preset != null) applyIdentityPreset(action.preset)
        featureModulesOpen = true
      } else if (action.type === 'set_recent_image') {
        const job = imageJobs[Number(action.image_index) - 1]
        if (!job?.output_url || job.status !== 'completed') continue
        const image = {
          server: true,
          ref: `${job.id}:output:0`,
          url: job.output_url,
          name: `생성 이미지 #${action.image_index}`,
          role: 'output'
        }
        if (action.target === 'vision' || action.target === 'styleReference') addKreaRefObjects(action.target, [image])
        else if (['identity', 'identityReference', 'depth', 'nk2e', 'anypaint'].includes(action.target)) setKreaImage(action.target, image)
        switchAssistantTab('image')
        featureModulesOpen = true
      } else if (action.type === 'set_outpaint') {
        const job = imageJobs[Number(action.image_index) - 1]
        if (!job?.output_url || job.status !== 'completed') continue
        const image = {
          server: true,
          ref: `${job.id}:output:0`,
          url: job.output_url,
          name: `생성 이미지 #${action.image_index}`,
          role: 'output'
        }
        kreaModules = Object.fromEntries(Object.keys(kreaModules).map((name) => [name, name === 'anypaint']))
        setKreaImage('anypaint', image)
        setKreaImage('anypaintMask', null)
        kreaOptions = {
          ...kreaOptions,
          outpaint_left: Number(action.outpaint_left) || 0,
          outpaint_top: Number(action.outpaint_top) || 0,
          outpaint_right: Number(action.outpaint_right) || 0,
          outpaint_bottom: Number(action.outpaint_bottom) || 0
        }
        imageForm = { ...imageForm, mode: 'create', prompt: '' }
        resetImageEnhancement()
        switchAssistantTab('image')
        featureModulesOpen = true
      }
    }
  }

  async function executeAssistantOperation(kind) {
    error = ''
    if (kind === 'image') {
      switchAssistantTab('image')
      const reason = imageDisabledReason()
      if (reason) throw new Error(reason)
      if (imageEnhancementActive() && !imageEnhancementCurrent()) await enhanceImagePrompt()
      if (imageEnhancementActive() && !imageEnhancementCurrent()) throw new Error(error || '프롬프트를 향상하지 못했습니다.')
      await generateImage()
      if (error) throw new Error(error)
      return '이미지 작업을 요청했습니다.'
    }
    if (kind === 'video') {
      switchAssistantTab('video')
      if (!videoForm.prompt.trim()) throw new Error('영상 프롬프트를 입력하세요.')
      if (videoEnhancementActive() && !videoEnhancementCurrent()) await enhanceVideoPrompt()
      if (videoEnhancementActive() && !videoEnhancementCurrent()) throw new Error(error || '영상 프롬프트를 향상하지 못했습니다.')
      await generateVideo()
      if (error) throw new Error(error)
      return '영상 작업을 요청했습니다.'
    }
    if (kind === 'speech') {
      switchAssistantTab('speech')
      if (!speechForm.text.trim()) throw new Error('읽을 문장을 입력하세요.')
      await generateSpeech()
      if (error) throw new Error(error)
      return '음성 작업을 요청했습니다.'
    }
    if (kind === 'recognition') {
      switchAssistantTab('recognition')
      if (!recognitionFile && !recognitionForm.url.trim()) throw new Error('먼저 자막으로 만들 파일이나 URL을 선택하세요.')
      await recognizeSpeech()
      if (error) throw new Error(error)
      return '자막 작업을 요청했습니다.'
    }
    throw new Error('지원하지 않는 작업입니다.')
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
      kreaOptions = {
        ...kreaOptions,
        checkpoint: config.image.default_checkpoint || 'official',
        sampling_preset: samplingPresetForCheckpoint(config.image.default_checkpoint || 'official', kreaOptions.sampling_preset),
        prompt_enhancer: Boolean(config.image.default_prompt_enhancer),
        ...((config.image.default_checkpoint || 'official') === 'official' ? {} : { filter_mode: 'off', filter_strength: 0 })
      }
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
  <div><span class="mark"><SparkBolt label="Spark Media" /></span><h1>Spark Media</h1></div>
  <div class="engine-strip">
    <span class="system-usage" title="5초 간격으로 갱신되는 DGX Spark 사용률"><b>CPU</b> {systemUsage.cpu_percent ?? '–'}% <b>GPU</b> {systemUsage.gpu_percent ?? '–'}% <b>MEM</b> {systemUsage.mem_used_gb == null ? '–' : Number(systemUsage.mem_used_gb).toFixed(1)}/{systemUsage.mem_total_gb == null ? '–' : Number(systemUsage.mem_total_gb).toFixed(1)}GB({systemUsage.mem_percent ?? '–'}%)</span>
    {#if tab === 'image'}
      <span class:running={engineStates[imageModeMeta[imageForm.mode].engine] === 'online'}><i></i>{imageModeMeta[imageForm.mode].short} API<span class="engine-state-text"> · {engineStates[imageModeMeta[imageForm.mode].engine] || 'offline'}</span></span>
      <span class:running={engineStates.prompt === 'online'}><i></i>Enhancer API<span class="engine-state-text"> · {engineStates.prompt || 'offline'}</span></span>
      <span class:running={engineStates.upscale === 'online'}><i></i>Upscale API<span class="engine-state-text"> · {engineStates.upscale || 'offline'}</span></span>
      <span class:running={engineStates.garment === 'online'}><i></i>Garment API<span class="engine-state-text"> · {engineStates.garment || 'offline'}</span></span>
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
    <button class:active={tab === 'image'} onclick={() => { tab = 'image'; refreshUserLoras() }}>이미지</button>
    <button class:active={tab === 'video'} onclick={() => tab = 'video'}>영상</button>
    <button class:active={tab === 'speech'} onclick={() => tab = 'speech'}>음성</button>
    <button class:active={tab === 'recognition'} onclick={() => tab = 'recognition'}>받아쓰기</button>
    <button class:active={tab === 'lora'} onclick={() => tab = 'lora'}>LoRA</button>
    <button class:active={tab === 'history'} onclick={() => tab = 'history'}>기록 <b>{jobs.length}</b></button>
    <button class:active={tab === 'settings'} onclick={openSettings}>설정</button>
  </nav>

  {#if error}<div class="error"><span>{error}</span><button onclick={() => error = ''}>×</button></div>{/if}

  {#if tab === 'image'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 이미지 화면">
      <button type="button" role="tab" aria-selected={mobileImagePane === 'create'} class:active={mobileImagePane === 'create'} onclick={() => mobileImagePane = 'create'}><span>만들기</span><small>설정·기능 모듈</small></button>
      <button type="button" role="tab" aria-selected={mobileImagePane === 'results'} class:active={mobileImagePane === 'results'} onclick={() => mobileImagePane = 'results'}><span>생성 이미지 목록</span><small>{imageJobs.length}개{#if activeJobs().some((job) => job.kind === 'image')} · 생성 중{/if}</small></button>
    </div>
    <section class="workspace image-workspace" class:mobile-results={mobileImagePane === 'results'}>
      <form class="image-create-pane" onsubmit={(e) => { e.preventDefault(); generateImage() }}>
        <div class="section-title"><div><span>01</span><h2>이미지 생성</h2></div><div class="image-title-actions">{#if imageForm.mode === 'create'}<button type="button" class="quiet header-prompt-tool" onclick={() => { promptExamplesTarget = 'image'; promptExamplesOpen = true }}>예제{#if filterPromptPreset}<b>선택됨</b>{/if}</button><PromptComposer compact storageKey="spark-media-prompt-composer-image" activeStyles={kreaModules.style ? kreaStyleSelections.map((style) => style.name) : []} onApply={(prompt, mode) => { const currentPrompt = imageForm.prompt.trimEnd(); imageForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${prompt}` : prompt; filterPromptPreset = ''; resetImageEnhancement() }} />{/if}<a class="quiet portrait-lab-open" href="/tools/portrait-lab/" target="_blank" rel="noreferrer">P Lab ↗</a><button type="button" class="quiet image-create-reset" disabled={busy} title="프롬프트와 이미지 생성 설정을 모두 비웁니다." onclick={resetImageCreation}>초기화</button></div></div>
        {#if imageCloneMessage}<div class="clone-notice"><span>{imageCloneMessage}</span><button type="button" aria-label="불러오기 안내 닫기" onclick={() => imageCloneMessage = ''}>×</button></div>{/if}
        {#if imageForm.mode === 'create'}
          <div class="prompt-tools-row">
            <button type="button" class="prompt-tool-open sequence-tool-open" disabled={busy} onclick={openImageSequence}><span>연속 생성</span></button>
            <button type="button" class="prompt-tool-open feature-tool-open" class:has-warning={Boolean(kreaModuleMessage)} aria-haspopup="dialog" onclick={() => featureModulesOpen = true}><span>기능 모듈</span>{#if activeKreaModuleLabels.length}<b>{activeKreaModuleLabels.length}개</b>{/if}</button>
            <button type="button" class="prompt-tool-open garment-tool-open" aria-haspopup="dialog" onclick={() => openGarmentExtractor()}><span>의상 추출</span></button>
          </div>
          {#if kreaModuleMessage}<small class="feature-module-toolbar-warning">{kreaModuleMessage}</small>{/if}
        {/if}
        <label>{kreaModules.identity ? '변경할 내용' : '프롬프트'}<textarea bind:value={imageForm.prompt} rows="7" placeholder="{kreaModules.identity ? '원본에서 바꿀 내용만 구체적으로 입력하세요.' : isPureOutpaint() ? '선택 사항 · 비워두면 원본을 자연스럽게 이어서 확장합니다.' : '만들고 싶은 장면을 입력하세요.'}"></textarea></label>
        {#if kreaModules.identity}
          <div class="identity-preserve-control">
            <div><strong>유지할 내용</strong><small>켜진 항목은 보존하고, 꺼진 항목은 변경을 허용합니다.{kreaModules.depth ? ' Depth 사용 중에는 자세·구도를 보존하지 않습니다.' : ''}</small></div>
            <div class="identity-preserve-chips">
              {#each identityPreserveCatalog as item}
                <button type="button" class:active={identityPreserveItems.includes(item.id)} disabled={kreaModules.depth && (item.id === 'pose' || item.id === 'composition')} onclick={() => toggleIdentityPreserveItem(item.id)}>{item.label}</button>
              {/each}
            </div>
            <label>추가 유지 조건<input bind:value={identityPreserveCustom} oninput={resetImageEnhancement} placeholder="예: 목걸이와 원본의 한글 문구"></label>
          </div>
        {/if}
        <div class="enhanced-prompt image-enhancer-panel" class:inactive={!imageEnhancementIsActive}>
          <div class="image-enhancer-panel-header">
            <div class="enhancer-panel-title"><strong title="연결된 Gemma 4 12B 모델이 Krea 2용 영어 프롬프트로 정리·확장합니다.">프롬프트 향상</strong><a href={kreaPromptGuideSource} target="_blank" rel="noreferrer">출처 ↗</a></div>
            <div class="enhancer-panel-actions">
              <button type="button" class="quiet enhancer-run" disabled={!imageEnhancementIsActive || enhancingPrompt || !rawImagePrompt().trim()} onclick={enhanceImagePrompt}>{enhancingPrompt ? '처리 중…' : imageEnhancementIsCurrent ? '다시 처리' : '프롬프트 향상'}</button>
            <div class="segmented compact">
              <button type="button" class:active={imageEnhanceEnabled} onclick={() => imageEnhanceEnabled = true}>켜짐</button>
              <button type="button" class:active={!imageEnhanceEnabled} onclick={() => imageEnhanceEnabled = false}>꺼짐</button>
            </div>
            </div>
          </div>
          {#if imageEnhancedPrompt.trim()}
            <textarea bind:value={imageEnhancedPrompt} rows="5" aria-label="Krea 향상 프롬프트"></textarea>
            <small>{looksLikeStructuredPrompt() ? 'JSON 형식은 원문을 유지합니다.' : imageEnhancementIsActive ? '실제 생성에 사용할 문장입니다. 확인하고 직접 수정할 수 있습니다.' : '꺼짐 · 기존 결과는 보존되며 실제 생성에는 원문을 사용합니다.'}</small>
          {:else}
            <small>{imageEnhancementIsActive ? '프롬프트 향상을 누르면 결과를 확인하고 직접 수정할 수 있습니다.' : '꺼짐 · 실제 생성에는 원문을 사용합니다.'}</small>
          {/if}
        </div>
        {#if imageForm.mode === 'create'}
          <section class="krea-runtime-controls" aria-label="Krea 모델 내부 조정">
            <div class="runtime-control-heading"><div><strong>모델 내부 조정</strong><small>필터 벡터와 텍스트 조건 강도를 간단히 조절합니다.</small></div><button type="button" class="runtime-info-button" aria-label="모델 내부 조정 설명" title="설명 보기" onclick={() => runtimeInfoOpen = true}>i</button></div>
            <div class="runtime-control-row">
              <label><span>필터 완화</span><select disabled={kreaOptions.checkpoint !== 'official'} value={kreaOptions.filter_mode} onchange={(event) => { const mode = event.currentTarget.value; kreaOptions = { ...kreaOptions, filter_mode: mode, filter_strength: filterModeDefault(mode) } }}><option value="off">{kreaOptions.checkpoint === 'official' ? '꺼짐 · 원본' : '체크포인트에 내장됨'}</option><option value="adherence">준수 강화 · skc3vo</option><option value="balanced">균형 · 2-vector</option><option value="strong">강함 · 3-vector</option></select></label>
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
          {#if featureModulesOpen}
            <div class="feature-module-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) featureModulesOpen = false }}>
              <section class="feature-module-modal" role="dialog" aria-modal="true" aria-label="기능 모듈">
                <header>
                  <div><strong>기능 모듈</strong><small>필요한 기능만 켜면 내부 연결은 자동으로 구성됩니다. 변경 내용은 즉시 유지됩니다.</small></div>
                  <button type="button" aria-label="닫기" onclick={() => featureModulesOpen = false}>×</button>
                </header>
                <div class="feature-module-content">
                  {#if kreaModuleMessage}<div class="feature-module-warning">{kreaModuleMessage}</div>{/if}
                  <section class="module-panel" aria-label="Krea 생성 모듈">
            <article class="module-card" class:enabled={kreaModules.identity}>
              <button type="button" class="module-toggle" aria-pressed={kreaModules.identity} onclick={() => toggleKreaModule('identity')}>
                <span class="module-icon">REF</span><span><strong>원본 수정</strong><small>Identity Edit · 원본의 인물이나 장면을 유지하면서 원하는 부분 변경</small></span><i></i>
              </button>
              {#if kreaModules.identity}
                <div class="module-body">
                  {#if parentImageJobID}<div class="clone-notice"><span>결과 작업 {parentImageJobID.slice(0, 8)}에서 계속 편집 중</span><button type="button" onclick={() => parentImageJobID = ''}>×</button></div>{/if}
                  <label>무엇을 할까요?<select value={identityPreset} onchange={(event) => applyIdentityPreset(event.currentTarget.value)}><option value="">직접 지시</option><option value="restage">같은 인물로 장면 변경</option><option value="sheet">2×2 캐릭터 시트</option><option value="faceSwap">얼굴 교체</option><option value="headSwap">머리 전체 교체</option><option value="personSwap">인물 교체</option><option value="tryon">의상 교체</option><option value="replace">선택 영역 교체</option></select></label>
                  <div class="module-source-field"><label class="module-file">{identityUI.primary}<small>{identityUI.primaryHint}</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('identity', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaIdentityPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaIdentityPreview} alt={`${identityUI.primary} 미리보기`} title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaIdentityPreview, identityUI.primary)} onkeydown={(event) => showImageOnKey(event, kreaIdentityPreview, identityUI.primary)}>{:else}<i>REF</i>{/if}<b title={kreaIdentityImage?.name || identityUI.primaryHint}>{kreaIdentityImage?.name || identityUI.primaryHint}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'identity'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'identity'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'identity'}>URL</button></div></div>
                  {#if identityUI.showSecondary}
                    <div class="module-source-field"><label class="module-file" class:optional={!identityUI.secondaryRequired}>{identityUI.secondary} · 최대 3장<small>{identityUI.secondaryHint}{identityUI.secondaryRequired ? ' · 1장 이상 필수' : ' · 선택 사항'} · 의상·포즈·소품을 함께 선택 가능</small><input type="file" accept="image/*" multiple onchange={(e) => addIdentityReferences(e.currentTarget.files)}><span class="module-file-display"><i>+REF</i><b>{kreaIdentityReferences.length ? `${kreaIdentityReferences.length}장 선택됨` : identityUI.secondaryHint}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'identityReference'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'identityReference'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'identityReference'}>URL</button></div></div>
                    {#if kreaIdentityReferences.length}<div class="reference-previews identity-reference-previews">{#each kreaIdentityReferences as image, i}<div><button type="button" class="reference-preview-open" onclick={(event) => showImage(event, image.preview || image.url, `${identityUI.secondary} ${i + 1}`)}><img src={image.preview || image.url} alt={`${identityUI.secondary} ${i + 1}`}><span class="reference-preview-index">{i + 1}</span></button><button type="button" class="reference-preview-remove" aria-label={`${identityUI.secondary} ${i + 1} 제거`} onclick={() => removeIdentityReference(i)}>×</button></div>{/each}</div>{/if}
                  {/if}
                  <p class="identity-prompt-guide">{identityUI.guide}</p>
                  <details class="module-advanced">
                    <summary><span>고급 설정</span><small>닮음·참조 해석·마스크</small></summary>
                    <div class="module-advanced-body">
                      <div class="module-controls">
                        <label><span>편집 LoRA 강도 <b>{Number(kreaOptions.identity_strength).toFixed(2)}</b></span><input type="range" min="0" max="2" step="0.05" bind:value={kreaOptions.identity_strength}></label>
                        <label><span>보조 참조 강도 <b>{kreaOptions.ref_boost}</b></span><input type="range" min="0" max="10" step="0.5" bind:value={kreaOptions.ref_boost}></label>
                        <label><span>원본 유지 강도 <b>{kreaOptions.source_ref_boost}</b></span><input type="range" min="0" max="10" step="0.5" bind:value={kreaOptions.source_ref_boost}></label>
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
                  <div class="module-source-field depth-source-field"><label class="module-file">구도 참조 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('depth', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaDepthPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaDepthPreview} alt="구도 참조 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaDepthPreview, 'Depth 구도 참조')} onkeydown={(event) => showImageOnKey(event, kreaDepthPreview, 'Depth 구도 참조')}>{:else}<i>3D</i>{/if}<b title={kreaDepthImage?.name || '원하는 자세와 구도의 이미지 선택'}>{kreaDepthImage?.name || '원하는 자세와 구도의 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'depth'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'depth'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'depth'}>URL</button></div></div>
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
                  <div class="module-source-field"><label class="module-file">참조 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('nk2e', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaNK2EPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaNK2EPreview} alt="NK2E 참조 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaNK2EPreview, 'NK2E 참조')} onkeydown={(event) => showImageOnKey(event, kreaNK2EPreview, 'NK2E 참조')}>{:else}<i>N2</i>{/if}<b title={kreaNK2EImage?.name || '편집하거나 윤곽을 가져올 이미지 선택'}>{kreaNK2EImage?.name || '편집하거나 윤곽을 가져올 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'nk2e'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'nk2e'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'nk2e'}>URL</button></div></div>
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
                    <div class="module-source-field"><label class="module-file">원본 이미지 <input type="file" accept="image/*" onchange={(e) => setKreaImage('anypaint', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaAnyPaintPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaAnyPaintPreview} alt="부분 수정 원본 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaAnyPaintPreview, '부분 수정·확장 원본')} onkeydown={(event) => showImageOnKey(event, kreaAnyPaintPreview, '부분 수정·확장 원본')}>{:else}<i>IMG</i>{/if}<b title={kreaAnyPaintImage?.name || '수정하거나 확장할 이미지 선택'}>{kreaAnyPaintImage?.name || '수정하거나 확장할 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'anypaint'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'anypaint'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'anypaint'}>URL</button></div></div>
                    <label class="module-file optional">수정 마스크 <small>선택 사항 · 흰 영역을 새로 생성</small><input type="file" accept="image/*" onchange={(e) => setKreaImage('anypaintMask', e.currentTarget.files?.[0] || null)}><span class="module-file-display">{#if kreaAnyPaintMaskPreview}<img role="button" tabindex="0" class="zoomable-source" src={kreaAnyPaintMaskPreview} alt="부분 수정 마스크 미리보기" title="클릭하여 크게 보기" onclick={(event) => showImage(event, kreaAnyPaintMaskPreview, '수정 마스크')} onkeydown={(event) => showImageOnKey(event, kreaAnyPaintMaskPreview, '수정 마스크')}>{:else}<i>MASK</i>{/if}<b title={kreaAnyPaintMask?.name || '확장만 할 때는 비워두기'}>{kreaAnyPaintMask?.name || '확장만 할 때는 비워두기'}</b></span></label>
                  </div>
                  <button type="button" class="mask-editor-open" disabled={!kreaAnyPaintPreview} onclick={() => maskEditorMode = 'anypaint'}>원본 위에서 수정 영역 칠하기</button>
                  <div class="outpaint-controls">
                    <strong>이미지 확장</strong><small>원본 크기에 선택한 픽셀만큼 더합니다.</small>
                    <div>
                      <label>왼쪽<select bind:value={kreaOptions.outpaint_left}><option value={0}>없음</option><option value={64}>64px</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                      <label>위쪽<select bind:value={kreaOptions.outpaint_top}><option value={0}>없음</option><option value={64}>64px</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                      <label>오른쪽<select bind:value={kreaOptions.outpaint_right}><option value={0}>없음</option><option value={64}>64px</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
                      <label>아래쪽<select bind:value={kreaOptions.outpaint_bottom}><option value={0}>없음</option><option value={64}>64px</option><option value={128}>128px</option><option value={256}>256px</option><option value={384}>384px</option><option value={512}>512px</option></select></label>
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
                <span class="module-icon">MY</span><span><strong>사용자 LoRA</strong><small>LoRA 관리에서 등록한 인물·캐릭터·스타일</small></span><i></i>
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
                            <label><input type="range" min="-2" max="2" step="0.01" value={selection.strength} oninput={(event) => updateUserLoraStrength(selection.filename, event.currentTarget.value)}><b>{Number(selection.strength).toFixed(2)}</b></label>
                            <button type="button" aria-label={`${selection.filename} 제거`} onclick={() => toggleUserLora(selection.filename)}>×</button>
                          </div>
                        {/each}
                      </div>
                    {/if}
                  {:else}
                    <small class="module-caution">등록된 LoRA가 없습니다. 상단 LoRA 탭에서 먼저 추가하세요.</small>
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
                  <div class="module-source-field"><label class="module-file">스타일 이미지 · 최대 2장<input type="file" accept="image/*" multiple onchange={(e) => addKreaRefs('styleReference', e.currentTarget.files)}><span class="module-file-display"><i>REF</i><b>{kreaStyleReferenceImages.length ? `${kreaStyleReferenceImages.length}장 선택됨` : '화풍을 가져올 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'styleReference'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'styleReference'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'styleReference'}>URL</button></div></div>
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
                  <div class="module-source-field"><label class="module-file">참조 이미지 · 최대 4장<input type="file" accept="image/*" multiple onchange={(e) => addKreaRefs('vision', e.currentTarget.files)}><span class="module-file-display"><i>VL</i><b>{kreaVisionImages.length ? `${kreaVisionImages.length}장 선택됨` : '내용을 참고할 이미지 선택'}</b></span></label><div class="module-source-actions"><button type="button" class="recent-result-open" onclick={() => recentImagePickerTarget = 'vision'}>최근 결과</button><button type="button" class="recent-result-open" onclick={() => presetImagePickerTarget = 'vision'}>프리셋</button><button type="button" class="recent-result-open" onclick={() => remoteImageTarget = 'vision'}>URL</button></div></div>
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
              <label>너비<input type="number" min="256" max="2048" step="8" bind:value={imageForm.width}></label>
              <label>높이<input type="number" min="256" max="2048" step="8" bind:value={imageForm.height}></label>
            </div>
          {/if}
        </div>
        {#if imageForm.mode === 'create'}
          <section class="image-generation-controls" aria-label="이미지 생성 설정">
            <div class="generation-control-heading"><strong>생성 설정</strong><small>{kreaOptions.sampling_preset === 'detail' ? 'ER-SDE / Simple' : kreaOptions.sampling_preset === 'moody' ? 'Euler A / Beta' : 'Euler / Simple'} · {kreaOptions.steps} steps</small></div>
            <div class="generation-control-grid">
              <label><span>체크포인트</span><select value={selectedKreaCheckpoint()} onchange={(event) => selectKreaCheckpoint(event.currentTarget.value)}>{#if kreaModules.identity}<option value="identity-convrot">Identity 전용 · ConvRot INT8</option>{/if}<option value="official">Krea 2 Turbo · 공식 NVFP4</option>{#if checkpointVisible('chriscole-edit-v1.1')}<option value="chriscole-edit-v1.1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'chriscole-edit-v1.1')?.ready}>Krea 2 Turbo Edit v1.1 · FP8</option>{/if}{#if checkpointVisible('moody-v7')}<option value="moody-v7" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-v7')?.ready}>Moody Krea 2 Mix V7 · NVFP4</option>{/if}{#if checkpointVisible('moody-cutie-v4')}<option value="moody-cutie-v4" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-cutie-v4')?.ready}>Moody Cutie Mix V4 · NVFP4</option>{/if}{#if checkpointVisible('moody-amateur-v1')}<option value="moody-amateur-v1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-amateur-v1')?.ready}>Moody Amateur Mix V1 · NVFP4</option>{/if}{#if checkpointVisible('ray-v1')}<option value="ray-v1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v1')?.ready}>Ray Artshoot V1 · FP8</option>{/if}{#if checkpointVisible('ray-v2')}<option value="ray-v2" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v2')?.ready}>Ray Artshoot V2 · FP8</option>{/if}{#if checkpointVisible('ray-v2-nvfp4')}<option value="ray-v2-nvfp4" disabled={!imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === 'ray-v2')?.validated}>Ray Artshoot V2 · NVFP4</option>{/if}{#if checkpointVisible('ray-v3')}<option value="ray-v3" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v3')?.ready}>Ray Artshoot V3 · INT8</option>{/if}{#if checkpointVisible('ray-v4')}<option value="ray-v4" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v4')?.ready}>Ray Artshoot V4 · INT8</option>{/if}{#if checkpointVisible('ray-v4-nvfp4')}<option value="ray-v4-nvfp4" disabled={!imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === 'ray-v4')?.validated}>Ray Artshoot V4 · NVFP4</option>{/if}</select></label>
              <label class="sampling-field"><span>샘플링 프리셋</span><select bind:value={kreaOptions.sampling_preset}><option value="default">기본 · Euler / Simple</option><option value="detail">디테일 · ER-SDE / Simple</option><option value="moody">Moody 권장 · Euler A / Beta</option></select></label>
              <label><span>스텝</span><select bind:value={kreaOptions.steps}><option value={8}>8 · 기본</option><option value={10}>10 · 균형</option><option value={12}>12 · 디테일</option></select></label>
              {#if kreaModules.identity}<label><span>텍스트 인코더</span><select bind:value={kreaOptions.identity_encoder}><option value="heretic" disabled={imageCheckpointStatus?.identity_runtime && !imageCheckpointStatus.identity_runtime.heretic_ready}>Heretic · INT8 ConvRot</option><option value="default">기본 · Qwen3-VL FP8</option></select></label>{/if}
              <label><span>시드 <small>-1 무작위</small></span><input type="number" min="-1" bind:value={imageForm.seed}></label>
            </div>
          </section>
        {:else}
          <div class="fields"><label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={imageForm.seed}></label></div>
        {/if}
        <button class="primary" disabled={Boolean(imageDisabledMessage) || enhancingPrompt}>{enhancingPrompt ? '프롬프트 향상 중…' : busy ? '요청 중…' : imageEnhancementIsActive && !imageEnhancementIsCurrent ? '향상 및 생성 시작' : activeJobs().some((j) => j.kind === 'image' || j.kind === 'video' || j.kind === 'speech') ? '이미지 큐에 추가' : '생성 시작'}</button>
        {#if imageDisabledMessage}<small class="submit-hint">{imageDisabledMessage}</small>{/if}
      </form>
      <aside class="image-results-pane">
        <div class="results-heading">
          <h3>생성 이미지 목록</h3>
          <div class="view-switch" aria-label="생성 이미지 목록 보기 방식">
            <button type="button" class:active={imageView === 'gallery'} onclick={() => setImageView('gallery')}>갤러리</button>
            <button type="button" class:active={imageView === 'list'} onclick={() => setImageView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="생성 이미지 목록" total={imageJobs.length} page={listPages.image} pageSize={listPageSizes.image} pageSizes={imagePageSizeOptions} sortOrder={listSortOrders.image} onPageChange={(page) => setListPage('image', page)} onPageSizeChange={(size) => setListPageSize('image', size)} onSortOrderChange={(order) => setListSortOrder('image', order)} />
        <div class="gallery image-results" class:list-view={imageView === 'list'}>
        {#each pagedImageJobs as job, imageIndex (job.id)}
          {@const generationProgress = job.status === 'queued' || job.status === 'running' ? imageGenerationProgress(job) : null}
          {@const visibleImageIndex = (listPages.image - 1) * listPageSizes.image + imageIndex + 1}
          <article class:pending={job.status !== 'completed'}>
            <span class="image-index-badge" title={`대화창에서 ${visibleImageIndex}번 이미지로 지칭`}>#{visibleImageIndex}</span>
            {#if imageView === 'list'}
              {#if job.output_url}<button type="button" class="image-list-thumb image-zoom" aria-label="생성 이미지 크게 보기" onclick={(event) => showImage(event, job.output_url, '생성 이미지', job.prompt, job.id)}><img src={job.output_url} alt={job.prompt}></button>{:else}<div class="image-list-thumb placeholder">{#if generationProgress}<div class="image-generation-status compact"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small></div>{:else}<span>{job.status}</span>{/if}</div>{/if}
              <div class="image-list-content">
                <span>{imageModeMeta[job.params?.mode]?.label || '이미지'}{imageModuleSummary(job)} · {job.params?.width || '—'}×{job.params?.height || '—'}{#if imageSamplingSummary(job)} · {imageSamplingSummary(job)}{/if}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</span>
                <button type="button" class="image-prompt" title="클릭하여 전체 프롬프트 보기" onclick={() => promptModal = { title: '전체 프롬프트', detail: `${imageModeMeta[job.params?.mode]?.label || '이미지'} · ${job.params?.width || '—'}×${job.params?.height || '—'}${imageSamplingSummary(job) ? ` · ${imageSamplingSummary(job)}` : ''}`, text: imagePromptModalText(job) }}>{job.prompt}</button>
                {#if job.error}<em>{job.error}</em>{/if}
                {#if job.status === 'failed'}<button type="button" class="job-retry image-retry" disabled={retryingJob === job.id} onclick={() => retryJob(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>{/if}
                <div class="image-clone-actions" aria-label="이 작업에서 불러오기">
                  <span>불러오기:</span>
                  <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'prompt')}>프롬프트</button>
                  <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'references')}>참조</button>
                  <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'settings')}>설정</button>
                  <button type="button" class="clone-all" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'all')}>{cloningImageJob === `${job.id}:all` ? '불러오는 중…' : '전체'}</button>
                </div>
                {#if job.status === 'completed'}<div class="image-post-actions"><span>후처리:</span><button type="button" title="이 결과를 Identity 원본으로 불러와 계속 편집" onclick={() => continueEditing(job)}>편집</button>{#if job.params?.mode === 'garment_extract' && job.outputs?.mask}<button type="button" title="저장된 의상 마스크 보기" onclick={(event) => showImage(event, job.outputs.mask, '의상 마스크', job.prompt, job.id)}>마스크</button>{:else}<button type="button" title="이 이미지에서 의상만 투명 PNG로 추출" disabled={engineStates.garment !== 'online'} onclick={() => openGarmentExtractor(job)}>의상</button>{/if}<button type="button" title="Ostris Edit LoRA로 다시 그립니다. 얼굴·색·글자·구도가 달라질 수 있습니다." disabled={Boolean(detailEnhancingImageJob) || Boolean(upscalingImageJob) || engineStates.image_create !== 'online'} onclick={() => detailEnhanceImage(job)}>{detailEnhancingImageJob === job.id ? '처리 중…' : '디테일'}</button><button type="button" title="SeedVR2로 복원하고 2배 확대" disabled={Boolean(detailEnhancingImageJob) || Boolean(upscalingImageJob) || engineStates.upscale !== 'online'} onclick={() => upscaleImage(job)}>{upscalingImageJob === job.id ? '처리 중…' : '고화질'}</button></div>{/if}
              </div>
            {:else}
              {#if job.output_url}<button type="button" class="gallery-image image-zoom" aria-label="생성 이미지 크게 보기" onclick={(event) => showImage(event, job.output_url, '생성 이미지', job.prompt, job.id)}><img src={job.output_url} alt={job.prompt}></button>{:else}<div class="placeholder">{#if generationProgress}<div class="image-generation-status"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small>{#if generationProgress.eta}<small>{generationProgress.eta}</small>{/if}</div>{:else}<span>{job.status}</span>{/if}</div>{/if}<span class="image-mode-badge" title={`${imageModeMeta[job.params?.mode]?.label || '이미지'}${imageModuleSummary(job)}`}>{imageModeMeta[job.params?.mode]?.label || '이미지'}{imageModuleSummary(job)}</span>
              <button type="button" class="image-prompt" title="클릭하여 전체 프롬프트 보기" onclick={() => promptModal = { title: '전체 프롬프트', detail: `${imageModeMeta[job.params?.mode]?.label || '이미지'} · ${job.params?.width || '—'}×${job.params?.height || '—'}${imageSamplingSummary(job) ? ` · ${imageSamplingSummary(job)}` : ''}`, text: imagePromptModalText(job) }}>{job.prompt}</button>
              {#if job.error}<em>{job.error}</em>{/if}
              {#if job.status === 'failed'}<button type="button" class="job-retry image-retry" disabled={retryingJob === job.id} onclick={() => retryJob(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>{/if}
              <div class="image-clone-actions" aria-label="이 작업에서 불러오기">
                <span>불러오기:</span>
                <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'prompt')}>프롬프트</button>
                <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'references')}>참조</button>
                <button type="button" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'settings')}>설정</button>
                <button type="button" class="clone-all" disabled={Boolean(cloningImageJob)} onclick={() => cloneImageJob(job, 'all')}>{cloningImageJob === `${job.id}:all` ? '불러오는 중…' : '전체'}</button>
              </div>
              {#if job.status === 'completed'}<div class="image-post-actions"><span>후처리:</span><button type="button" title="이 결과를 Identity 원본으로 불러와 계속 편집" onclick={() => continueEditing(job)}>편집</button>{#if job.params?.mode === 'garment_extract' && job.outputs?.mask}<button type="button" title="저장된 의상 마스크 보기" onclick={(event) => showImage(event, job.outputs.mask, '의상 마스크', job.prompt, job.id)}>마스크</button>{:else}<button type="button" title="이 이미지에서 의상만 투명 PNG로 추출" disabled={engineStates.garment !== 'online'} onclick={() => openGarmentExtractor(job)}>의상</button>{/if}<button type="button" title="Ostris Edit LoRA로 다시 그립니다. 얼굴·색·글자·구도가 달라질 수 있습니다." disabled={Boolean(detailEnhancingImageJob) || Boolean(upscalingImageJob) || engineStates.image_create !== 'online'} onclick={() => detailEnhanceImage(job)}>{detailEnhancingImageJob === job.id ? '처리 중…' : '디테일'}</button><button type="button" title="SeedVR2로 복원하고 2배 확대" disabled={Boolean(detailEnhancingImageJob) || Boolean(upscalingImageJob) || engineStates.upscale !== 'online'} onclick={() => upscaleImage(job)}>{upscalingImageJob === job.id ? '처리 중…' : '고화질'}</button></div>{/if}
            {/if}
            {#if job.status === 'queued'}<button class="job-stop" disabled={cancellingJob === job.id} onclick={() => cancelJob(job)}>{cancellingJob === job.id ? '취소 중…' : '대기 취소'}</button>{:else if job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}
          </article>
        {:else}<div class="empty">첫 이미지가 여기에 나타납니다.</div>{/each}
      </div>
      <ResultPagination compact label="생성 이미지 목록" total={imageJobs.length} page={listPages.image} pageSize={listPageSizes.image} onPageChange={(page) => setListPage('image', page)} />
      </aside>
    </section>
  {:else if tab === 'video'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 영상 화면">
      <button type="button" role="tab" aria-selected={mobileVideoPane === 'create'} class:active={mobileVideoPane === 'create'} onclick={() => mobileVideoPane = 'create'}><span>만들기</span><small>영상 생성 설정</small></button>
      <button type="button" role="tab" aria-selected={mobileVideoPane === 'results'} class:active={mobileVideoPane === 'results'} onclick={() => mobileVideoPane = 'results'}><span>생성 영상 목록</span><small>{videoJobs.length}개{#if activeJobs().some((job) => job.kind === 'video')} · 생성 중{/if}</small></button>
    </div>
    <section class="workspace mobile-media-workspace" class:mobile-results={mobileVideoPane === 'results'}>
      <form class="mobile-create-pane" onsubmit={(e) => { e.preventDefault(); generateVideo() }}>
        <div class="section-title"><div><span>02</span><h2>영상 생성</h2></div><div class="image-title-actions"><button type="button" class="quiet header-prompt-tool" onclick={() => { promptExamplesTarget = 'video'; promptExamplesOpen = true }}>예제{#if videoPromptPreset}<b>선택됨</b>{/if}</button><PromptComposer compact storageKey="spark-media-prompt-composer-video" onApply={(prompt, mode) => { const currentPrompt = videoForm.prompt.trimEnd(); videoForm.prompt = mode === 'append' && currentPrompt ? `${currentPrompt}\n${prompt}` : prompt; videoPromptPreset = ''; resetVideoEnhancement() }} /><a class="quiet portrait-lab-open" href="/tools/portrait-lab/" target="_blank" rel="noreferrer">P Lab ↗</a><button type="button" class="quiet image-create-reset" disabled={busy} title="영상 생성 설정을 모두 비웁니다." onclick={resetVideoCreation}>초기화</button></div></div>
        <label>원본 프롬프트<textarea bind:value={videoForm.prompt} rows="5" placeholder="장면과 움직임을 자연스럽게 입력하세요." required></textarea></label>
        <section class="video-conditioning">
          <div class="video-conditioning-heading"><div><strong>장면 이미지</strong><small>시작·마지막 이미지와 중간 키프레임을 필요한 만큼 조합합니다.</small></div><div class="video-conditioning-heading-actions"><button type="button" title="선택한 장면 이미지를 보고 LTX 영상 프롬프트 만들기" disabled={creatingVideoPrompt || (!videoImage && !videoEndImage && !videoKeyframes.some((item) => item.image))} onclick={createVideoPromptFromScenes}>{creatingVideoPrompt ? '분석 중…' : '프롬프트 만들기'}</button><button type="button" disabled={videoKeyframes.length >= 8} onclick={addVideoKeyframe}>+ 키프레임</button></div></div>
          {#if videoPromptCreationMessage}<small class="video-prompt-creation-message">{videoPromptCreationMessage}</small>{/if}
          <div class="video-boundary-images">
            <article class:has-image={Boolean(videoImage)}>
              <div class="video-condition-heading"><strong>시작 이미지</strong><small>0초 · 선택 사항</small></div>
              {#if videoImage}
                <button type="button" class="video-condition-preview" title="클릭하여 크게 보기" onclick={(event) => showImage(event, videoImagePreview(videoImage), '영상 시작 이미지')}><img src={videoImagePreview(videoImage)} alt="영상 시작 이미지"></button>
                <span class="video-condition-name" title={videoImage.name}>{videoImage.name}</span>
              {:else}<div class="video-condition-empty">첫 장면을 고정하려면 이미지를 선택하세요.</div>{/if}
              <div class="video-condition-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => setVideoConditionImage('start', e.currentTarget.files?.[0] || null)}><span>파일</span></label><button type="button" onclick={() => videoImagePickerTarget = 'start'}>최근 결과</button><button type="button" onclick={() => videoRemoteImageTarget = 'start'}>URL</button>{#if videoImage}<button type="button" class="danger" onclick={() => setVideoConditionImage('start', null)}>제거</button>{/if}</div>
              {#if videoImage}<label class="video-condition-strength">반영 강도<input type="number" min="0" max="1" step="0.05" bind:value={videoForm.image_strength}></label>{/if}
            </article>
            <article class:has-image={Boolean(videoEndImage)}>
              <div class="video-condition-heading"><strong>마지막 이미지</strong><small>{((framesForDuration(videoDurationSeconds, videoForm.fps) - 1) / videoForm.fps).toFixed(1)}초 · 선택 사항</small></div>
              {#if videoEndImage}
                <button type="button" class="video-condition-preview" title="클릭하여 크게 보기" onclick={(event) => showImage(event, videoImagePreview(videoEndImage), '영상 마지막 이미지')}><img src={videoImagePreview(videoEndImage)} alt="영상 마지막 이미지"></button>
                <span class="video-condition-name" title={videoEndImage.name}>{videoEndImage.name}</span>
              {:else}<div class="video-condition-empty">도착 장면을 고정하려면 이미지를 선택하세요.</div>{/if}
              <div class="video-condition-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => setVideoConditionImage('end', e.currentTarget.files?.[0] || null)}><span>파일</span></label><button type="button" onclick={() => videoImagePickerTarget = 'end'}>최근 결과</button><button type="button" onclick={() => videoRemoteImageTarget = 'end'}>URL</button>{#if videoEndImage}<button type="button" class="danger" onclick={() => setVideoConditionImage('end', null)}>제거</button>{/if}</div>
              {#if videoEndImage}<label class="video-condition-strength">반영 강도<input type="number" min="0" max="1" step="0.05" bind:value={videoEndStrength}></label>{/if}
            </article>
          </div>
          {#if videoKeyframes.length}
            <div class="video-keyframes">
              {#each videoKeyframes as keyframe, index (keyframe.id)}
                <article>
                  <div class="video-condition-heading"><strong>키프레임 {index + 1}</strong><button type="button" aria-label="키프레임 제거" onclick={() => removeVideoKeyframe(keyframe.id)}>×</button></div>
                  {#if keyframe.image}<button type="button" class="video-keyframe-preview" title="클릭하여 크게 보기" onclick={(event) => showImage(event, videoImagePreview(keyframe.image), `영상 키프레임 ${index + 1}`)}><img src={videoImagePreview(keyframe.image)} alt="영상 키프레임 {index + 1}"></button>{:else}<div class="video-keyframe-empty">IMG</div>{/if}
                  <div class="video-keyframe-controls">
                    <span class="video-condition-name" title={keyframe.image?.name || '이미지 미선택'}>{keyframe.image?.name || '이미지 미선택'}</span>
                    <div class="video-condition-actions"><label><input type="file" accept="image/png,image/jpeg,image/webp" onchange={(e) => setVideoConditionImage(`keyframe:${keyframe.id}`, e.currentTarget.files?.[0] || null)}><span>파일</span></label><button type="button" onclick={() => videoImagePickerTarget = `keyframe:${keyframe.id}`}>최근 결과</button><button type="button" onclick={() => videoRemoteImageTarget = `keyframe:${keyframe.id}`}>URL</button></div>
                    <div class="video-keyframe-numbers"><label>위치 (초)<input type="number" min="0.01" max={Math.max(0.01, (framesForDuration(videoDurationSeconds, videoForm.fps) - 2) / videoForm.fps)} step="0.01" value={keyframe.time} onchange={(event) => updateVideoKeyframe(keyframe.id, 'time', event.currentTarget.value)}></label><label>반영 강도<input type="number" min="0" max="1" step="0.05" value={keyframe.strength} onchange={(event) => updateVideoKeyframe(keyframe.id, 'strength', event.currentTarget.value)}></label></div>
                  </div>
                </article>
              {/each}
            </div>
          {/if}
        </section>
        <div class="enhanced-prompt image-enhancer-panel" class:inactive={!videoEnhancementIsActive}>
          <div class="image-enhancer-panel-header">
            <div class="enhancer-panel-title"><strong title="연결된 Gemma 4 12B 모델이 LTX 캡션 형식의 영어 프롬프트로 정리·확장합니다.">프롬프트 향상</strong></div>
            <div class="enhancer-panel-actions">
              <button type="button" class="quiet enhancer-run" disabled={!videoEnhancementIsActive || enhancingPrompt || !videoForm.prompt.trim()} onclick={enhanceVideoPrompt}>{enhancingPrompt ? '향상 중…' : videoEnhancementIsCurrent ? '다시 향상' : '미리 향상'}</button>
              <div class="segmented compact">
                <button type="button" class:active={videoEnhanceEnabled} onclick={() => videoEnhanceEnabled = true}>켜짐</button>
                <button type="button" class:active={!videoEnhanceEnabled} onclick={() => videoEnhanceEnabled = false}>꺼짐</button>
              </div>
            </div>
          </div>
          {#if videoEnhancedPrompt.trim()}
            <textarea bind:value={videoEnhancedPrompt} rows="8" aria-label="향상된 영상 프롬프트"></textarea>
            <small>{videoEnhancementIsActive ? `${videoImage ? '시작 이미지를 분석한' : '텍스트 기반'} 실제 생성 프롬프트입니다. 생성 전에 직접 수정할 수 있습니다.` : '꺼짐 · 기존 결과는 보존되며 실제 생성에는 원문을 사용합니다.'}</small>
          {:else}
            <small>{videoEnhancementIsActive ? '생성 시작 시 자동으로 향상합니다. 먼저 확인하려면 미리 향상을 누르세요.' : videoImage && !config?.prompt_enhancement.vision_enabled ? '현재 향상 모델은 이미지를 볼 수 없어 I2V에서는 원문을 사용합니다.' : '꺼짐 · 실제 생성에는 원문을 사용합니다.'}</small>
          {/if}
        </div>
        <div class="fields three">
          <label>너비<input type="number" min="256" max="1920" step="64" bind:value={videoForm.width}></label>
          <label>높이<input type="number" min="256" max="1920" step="64" bind:value={videoForm.height}></label>
          <label class="duration-field"><span>길이 (초) <small>{framesForDuration(videoDurationSeconds, videoForm.fps)} 프레임 · 8k+1</small></span><input aria-label="영상 길이 초" type="number" min="0.1" step="0.1" bind:value={videoDurationSeconds}></label>
          <label>FPS<input type="number" min="1" max="60" step="1" bind:value={videoForm.fps}></label>
          <label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={videoForm.seed}></label>
        </div>
        {#if videoConditioningDisabledReason()}<small class="video-conditioning-error">{videoConditioningDisabledReason()}</small>{/if}
        <button class="primary" disabled={busy || enhancingPrompt || Boolean(videoConditioningDisabledReason())}>{enhancingPrompt ? '프롬프트 처리 중…' : busy ? '요청 중…' : activeJobs().some((j) => j.kind === 'image' || j.kind === 'video' || j.kind === 'speech') ? '영상 큐에 추가' : '생성 시작'}</button>
      </form>
      <aside class="video-results-pane mobile-results-pane">
        <div class="results-heading">
          <h3>생성 영상 목록</h3>
          <div class="view-switch" aria-label="생성 영상 목록 보기 방식">
            <button type="button" class:active={videoView === 'gallery'} onclick={() => setVideoView('gallery')}>갤러리</button>
            <button type="button" class:active={videoView === 'list'} onclick={() => setVideoView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="생성 영상 목록" total={videoJobs.length} page={listPages.video} pageSize={listPageSizes.video} pageSizes={imagePageSizeOptions} sortOrder={listSortOrders.video} onPageChange={(page) => setListPage('video', page)} onPageSizeChange={(size) => setListPageSize('video', size)} onSortOrderChange={(order) => setListSortOrder('video', order)} />
        <div class="video-list" class:list-view={videoView === 'list'}>
        {#each pagedVideoJobs as job (job.id)}
          {@const generationProgress = job.status === 'queued' || job.status === 'running' ? videoGenerationProgress(job) : null}
          <article class:pending={job.status !== 'completed'}>
            {#if videoView === 'list'}
              {#if job.output_url}<button type="button" class="video-list-thumb" aria-label="영상 크게 보기" onclick={() => showVideo(job)}><!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.output_url}></video></button>{:else}<div class="video-list-thumb empty-thumb">{#if generationProgress}<div class="image-generation-status compact"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small></div>{:else}<span>{job.status}</span>{/if}</div>{/if}
              <div class="video-list-content"><small>{job.params?.width}×{job.params?.height} · {formatDuration(videoJobDuration(job))} · {job.params?.fps} fps{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small><button type="button" class="image-prompt" title={job.prompt} onclick={() => promptModal = { title: '전체 프롬프트', detail: `영상 · ${job.params?.width}×${job.params?.height} · ${formatDuration(videoJobDuration(job))} · ${job.params?.fps} fps`, text: videoPromptModalText(job) }}>{job.prompt}</button>{#if job.error}<em>{job.error}</em>{/if}</div>
            {:else}
              {#if job.output_url}<button type="button" class="video-gallery-thumb" aria-label="영상 크게 보기" title="클릭하여 크게 보기" onclick={() => showVideo(job)}><!-- svelte-ignore a11y_media_has_caption --><video preload="metadata" muted playsinline src={job.output_url}></video></button>{:else}<div class="video-placeholder">{#if generationProgress}<div class="image-generation-status"><strong>{generationProgress.label}</strong><div class="image-generation-bar"><i style={`width:${generationProgress.percent}%`}></i></div><small>{generationProgress.elapsed}</small>{#if generationProgress.eta}<small>{generationProgress.eta}</small>{/if}</div>{:else}<span>{job.status}</span>{/if}</div>{/if}<button type="button" class="image-prompt" title={job.prompt} onclick={() => promptModal = { title: '전체 프롬프트', detail: `영상 · ${job.params?.width}×${job.params?.height} · ${formatDuration(videoJobDuration(job))} · ${job.params?.fps} fps`, text: videoPromptModalText(job) }}>{job.prompt}</button><small>{job.params?.width}×{job.params?.height} · {formatDuration(videoJobDuration(job))} · {job.params?.fps} fps{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</small>{#if job.error}<em>{job.error}</em>{/if}
            {/if}
            {#if job.status === 'queued'}<div class="video-job-actions"><button class="job-stop" disabled={cancellingJob === job.id} onclick={() => cancelJob(job)}>{cancellingJob === job.id ? '취소 중…' : '대기 취소'}</button></div>{:else if job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'}<div class="video-job-actions">{#if job.status === 'failed'}<button type="button" class="job-retry" disabled={retryingJob === job.id} onclick={() => retryJob(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>{/if}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button></div>{/if}
          </article>
        {:else}<div class="empty">첫 영상이 여기에 나타납니다.</div>{/each}
      </div>
      <ResultPagination compact label="생성 영상 목록" total={videoJobs.length} page={listPages.video} pageSize={listPageSizes.video} onPageChange={(page) => setListPage('video', page)} />
      </aside>
    </section>
  {:else if tab === 'speech'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 음성 화면">
      <button type="button" role="tab" aria-selected={mobileSpeechPane === 'create'} class:active={mobileSpeechPane === 'create'} onclick={() => mobileSpeechPane = 'create'}><span>만들기</span><small>음성 생성 설정</small></button>
      <button type="button" role="tab" aria-selected={mobileSpeechPane === 'results'} class:active={mobileSpeechPane === 'results'} onclick={() => mobileSpeechPane = 'results'}><span>생성 음성 목록</span><small>{speechJobs.length}개{#if activeJobs().some((job) => job.kind === 'speech')} · 생성 중{/if}</small></button>
    </div>
    <section class="workspace mobile-media-workspace" class:mobile-results={mobileSpeechPane === 'results'}>
      <form class="mobile-create-pane" onsubmit={(e) => { e.preventDefault(); generateSpeech() }}>
        <div class="section-title"><div><span>03</span><h2>음성 생성</h2></div><div class="image-title-actions"><button type="button" class="quiet image-create-reset" disabled={busy} title="음성 생성 설정을 모두 비웁니다." onclick={resetSpeechCreation}>초기화</button></div></div>
        <label>읽을 문장<textarea bind:value={speechForm.text} rows="7" placeholder="음성으로 변환할 문장을 입력하세요." required></textarea></label>
        <label>연기 지시 <small>선택 사항 · 1.7B instruction control</small><textarea bind:value={speechForm.instructions} rows="3" placeholder="예: 기쁘고 활기찬 목소리로, 중요한 단어는 힘주어 말해 주세요."></textarea></label>
        <div class="fields three">
          <label>언어<select bind:value={speechForm.language}><option>Korean</option><option>English</option><option>Chinese</option><option>Japanese</option><option>Auto</option></select></label>
          <label>화자<select bind:value={speechForm.speaker}><option>Sohee</option><option>Vivian</option><option>Serena</option><option>Ryan</option><option>Aiden</option><option>Ono_Anna</option></select></label>
          <label><span>시드 <small>-1은 무작위</small></span><input type="number" min="-1" bind:value={speechForm.seed}></label>
        </div>
        <button class="primary" disabled={busy}>{busy ? '요청 중…' : activeJobs().some((j) => j.kind === 'image' || j.kind === 'video' || j.kind === 'speech') ? '음성 큐에 추가' : '음성 만들기'}</button>
      </form>
      <aside class="mobile-results-pane"><div class="results-heading"><h3>생성 음성 목록</h3></div>
        <ResultPagination label="생성 음성 목록" total={speechJobs.length} page={listPages.speech} pageSize={listPageSizes.speech} pageSizes={pageSizeOptions} sortOrder={listSortOrders.speech} onPageChange={(page) => setListPage('speech', page)} onPageSizeChange={(size) => setListPageSize('speech', size)} onSortOrderChange={(order) => setListSortOrder('speech', order)} />
        <div class="audio-list">
        {#each pagedSpeechJobs as job (job.id)}<article><div><span>{job.params?.speaker}{#if job.params?.seed >= 0} · seed {job.params.seed}{/if}</span><p>{job.prompt}</p>{#if job.output_url}<button type="button" class="audio-modal-open" onclick={() => showAudio(job)}>크게 보기</button>{/if}</div>{#if job.params?.instructions}<small class="instruction">지시 · {job.params.instructions}</small>{/if}{#if job.output_url}<audio controls src={job.output_url}></audio>{:else}<small>{job.status === 'queued' ? `대기 ${generationQueuePosition(job)}번째` : statusLabels[job.status] || job.status}</small>{/if}{#if job.error}<em>{job.error}</em>{/if}{#if job.status === 'queued'}<button class="job-stop" disabled={cancellingJob === job.id} onclick={() => cancelJob(job)}>{cancellingJob === job.id ? '취소 중…' : '대기 취소'}</button>{:else if job.status === 'failed'}<button class="job-retry" disabled={retryingJob === job.id} onclick={() => retryJob(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>{:else if job.status === 'completed' || job.status === 'cancelled'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</article>{:else}<div class="empty">첫 음성이 여기에 나타납니다.</div>{/each}
      </div>
      <ResultPagination compact label="생성 음성 목록" total={speechJobs.length} page={listPages.speech} pageSize={listPageSizes.speech} onPageChange={(page) => setListPage('speech', page)} />
      </aside>
    </section>
  {:else if tab === 'recognition'}
    <div class="mobile-image-nav" role="tablist" aria-label="모바일 받아쓰기 화면">
      <button type="button" role="tab" aria-selected={mobileRecognitionPane === 'create'} class:active={mobileRecognitionPane === 'create'} onclick={() => mobileRecognitionPane = 'create'}><span>만들기</span><small>받아쓰기 설정</small></button>
      <button type="button" role="tab" aria-selected={mobileRecognitionPane === 'results'} class:active={mobileRecognitionPane === 'results'} onclick={() => mobileRecognitionPane = 'results'}><span>생성 자막 목록</span><small>{recognitionJobs.length}개{#if activeJobs().some((job) => job.kind === 'recognition')} · 처리 중{/if}</small></button>
    </div>
    <section class="workspace mobile-media-workspace" class:mobile-results={mobileRecognitionPane === 'results'}>
      <form class="mobile-create-pane" onsubmit={(e) => { e.preventDefault(); recognizeSpeech() }}>
        <div class="section-title"><div><span>04</span><h2>자막 받아쓰기</h2></div><div class="image-title-actions"><button type="button" class="quiet image-create-reset" disabled={busy} title="받아쓰기 설정을 모두 비웁니다." onclick={resetRecognitionCreation}>초기화</button></div></div>
        <section class="recognition-source-panel">
          <div class="recognition-source-heading"><div><strong>입력 소스</strong><small>링크 또는 로컬 파일 중 편한 방법을 사용하세요.</small></div></div>
          <div class="recognition-source-bar">
            <div class="recognition-url-input" class:active={recognitionForm.source === 'url'}><i>URL</i><input aria-label="영상 링크" type="url" bind:value={recognitionForm.url} oninput={updateRecognitionURL} placeholder="영상 페이지 URL"></div>
            <button type="button" class="quiet media-options-load" disabled={loadingRecognitionOptions || !recognitionForm.url.trim()} onclick={loadRecognitionOptions}>{loadingRecognitionOptions ? '조회 중…' : '조회'}</button>
            <label class="recognition-file-button" class:active={recognitionForm.source === 'file'} title={recognitionFile?.name || '영상·음성 파일 선택'}><input bind:this={recognitionFileInput} type="file" accept="audio/*,video/*,.mkv,.mp4,.webm,.mov,.m4v,.avi,.wav,.flac,.ogg,.mp3,.m4a,.aac" onchange={updateRecognitionFile}><i>FILE</i><span>{recognitionFile?.name || '파일 선택'}</span></label>
            {#if recognitionFile}<button type="button" class="recognition-file-clear" aria-label="선택 파일 해제" title="선택 파일 해제" onclick={clearRecognitionFile}>×</button>{/if}
          </div>
          <small class="recognition-source-note">{recognitionForm.source === 'file' && recognitionFile ? '선택한 파일을 작업 폴더로 바로 전송합니다.' : '링크 영상을 보관하고 음성을 분리합니다. 필요하면 브라우저 해석기를 사용합니다.'}</small>
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
                <small>별도 파트나 출처 선택 없이 기본 방식으로 처리합니다.</small>
              {/if}
            </div>
          {/if}
        </section>
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
          <h3>생성 자막 목록</h3>
          <div class="view-switch" aria-label="생성 자막 목록 보기 방식">
            <button type="button" class:active={subtitleView === 'gallery'} onclick={() => setSubtitleView('gallery')}>갤러리</button>
            <button type="button" class:active={subtitleView === 'list'} onclick={() => setSubtitleView('list')}>리스트</button>
          </div>
        </div>
        <ResultPagination label="생성 자막 목록" total={recognitionJobs.length} page={listPages.recognition} pageSize={listPageSizes.recognition} pageSizes={imagePageSizeOptions} sortOrder={listSortOrders.recognition} onPageChange={(page) => setListPage('recognition', page)} onPageSizeChange={(size) => setListPageSize('recognition', size)} onSortOrderChange={(order) => setListSortOrder('recognition', order)} />
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
      <ResultPagination compact label="생성 자막 목록" total={recognitionJobs.length} page={listPages.recognition} pageSize={listPageSizes.recognition} onPageChange={(page) => setListPage('recognition', page)} />
      </aside>
    </section>
  {:else if tab === 'lora'}
    <LoraStudio {imageJobs} onChanged={refreshUserLoras} onOpenSettings={openSettings} />
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
          {#each [['video', 'LTX 영상'], ['speech', 'Qwen3 TTS'], ['recognition', 'Qwen3 ASR'], ['prompt', '프롬프트·번역'], ['upscale', 'SeedVR2 고화질'], ['media', '미디어 접근·FFmpeg']] as item}
            <label><span>{item[1]} <small class:online={engineStates[item[0]] === 'online'}>{engineStates[item[0]] || 'offline'}</small></span><input type="url" bind:value={settings.engines[item[0]].endpoint} required></label>
          {/each}
        </div>
      </section>

      <section class="settings-card">
        <h3>다운로드 인증</h3>
        <p>Civitai와 Hugging Face 인증 정보를 한 번 저장하면 Krea 체크포인트·LoRA·LTX 모델 다운로드에서 함께 사용합니다. 저장된 값은 화면으로 다시 전송되지 않습니다.</p>
        <div class="fields">
          <label>Civitai API 키
            <input type="password" autocomplete="new-password" bind:value={civitaiToken} placeholder={imageCheckpointStatus?.token_configured ? '저장됨 · 변경할 때만 입력' : 'API key'}>
          </label>
          <label>Hugging Face read 토큰
            <input type="password" autocomplete="new-password" bind:value={hfToken} placeholder={videoModelStatus?.token_configured ? '저장됨 · 변경할 때만 입력' : 'hf_…'}>
          </label>
        </div>
        <button type="button" class="primary" disabled={savingDownloadCredentials || (!civitaiToken.trim() && !hfToken.trim())} onclick={saveDownloadCredentials}>{savingDownloadCredentials ? '저장 중…' : '인증 정보 저장'}</button>
      </section>

      <section class="settings-card">
        <h3>Krea 모델 준비</h3>
        <p>저장된 Civitai 키와 Hugging Face 토큰을 사용해 Identity Edit용 인코더와 Ray·Moody·Turbo Edit 체크포인트를 영구 모델 캐시에 준비합니다.</p>
        {#if imageCheckpointStatus}
          <div class="storage-stats">
            <span><small>상태</small><strong>{imageCheckpointStatus.ready ? '모두 준비됨' : imageCheckpointStatus.preparing ? '다운로드 중' : '준비 필요'}</strong></span>
            <span><small>완료</small><strong>{imageCheckpointStatus.variants?.filter((item) => item.ready).length || 0}/{imageCheckpointStatus.variants?.length || 4}</strong></span>
            <span><small>현재</small><strong>{imageCheckpointStatus.current || '대기'}</strong></span>
          </div>
          {#if imageCheckpointStatus.preparing && imageCheckpointStatus.total_bytes > 0}
            <div class="progress-track"><span style={`width:${Math.min(100, imageCheckpointStatus.downloaded_bytes / imageCheckpointStatus.total_bytes * 100)}%`}></span></div>
            <small>{formatBytes(imageCheckpointStatus.downloaded_bytes)} / {formatBytes(imageCheckpointStatus.total_bytes)}</small>
          {/if}
          {#if imageCheckpointStatus.error}<small class="model-setup-error">{imageCheckpointStatus.error}</small>{/if}
          {#if imageCheckpointStatus.identity_runtime}
            <div class="model-variant-list">
              <label>
                <input type="checkbox" checked disabled>
                <span>Identity · ConvRot INT8<small>{imageCheckpointStatus.identity_runtime.convrot_ready ? '준비됨' : '컨테이너 시작 시 Hugging Face에서 자동 준비'}</small></span>
                <a href={imageCheckpointStatus.identity_runtime.convrot_source} target="_blank" rel="noreferrer">출처</a>
              </label>
              <label>
                <input type="checkbox" checked disabled>
                <span>Identity · Heretic INT8 ConvRot<small>{imageCheckpointStatus.identity_runtime.heretic_ready ? '준비됨' : imageCheckpointStatus.identity_runtime.heretic_downloaded_bytes ? `${formatBytes(imageCheckpointStatus.identity_runtime.heretic_downloaded_bytes)} / ${formatBytes(imageCheckpointStatus.identity_runtime.heretic_size_bytes)}` : formatBytes(imageCheckpointStatus.identity_runtime.heretic_size_bytes)}</small></span>
                <a href={imageCheckpointStatus.identity_runtime.heretic_source} target="_blank" rel="noreferrer">출처</a>
              </label>
            </div>
          {/if}
          <h4>이미지 탭 모델 표시</h4>
          <p>체크한 모델만 이미지 생성의 체크포인트 목록에 표시됩니다. 공식 NVFP4는 항상 표시됩니다.</p>
          <div class="model-variant-list checkpoint-visibility-list">
            <label><input type="checkbox" checked disabled><span>Krea 2 Turbo · 공식 NVFP4<small>항상 표시</small></span></label>
            {#each checkpointDisplayChoices as choice}
              <label class:unavailable={!displayCheckpointReady(choice[0])}><input type="checkbox" checked={checkpointVisible(choice[0])} onchange={(event) => setCheckpointVisible(choice[0], event.currentTarget.checked)}><span>{choice[1]}<small>{displayCheckpointReady(choice[0]) ? '준비됨' : '모델 준비 필요'}</small></span></label>
            {/each}
          </div>
          <hr>
          <h4>다운로드 대상</h4>
          <p>아래에서 선택한 원본 파일을 `선택 모델 준비` 버튼으로 다운로드합니다.</p>
          <div class="model-variant-list">
            {#each imageCheckpointStatus.variants || [] as variant}
              <label>
                <input type="checkbox" checked={checkpointSelection[variant.id]} onchange={(event) => checkpointSelection = { ...checkpointSelection, [variant.id]: event.currentTarget.checked }}>
                <span>{variant.label}<small>{variant.ready ? '준비됨' : variant.downloaded_bytes ? `${formatBytes(variant.downloaded_bytes)} / ${formatBytes(variant.size_bytes)}` : formatBytes(variant.size_bytes)}</small></span>
                <a href={variant.source} target="_blank" rel="noreferrer">출처</a>
              </label>
            {/each}
          </div>
        {/if}
        <button type="button" class="primary" disabled={preparingImageCheckpoints || imageCheckpointStatus?.preparing || (!civitaiToken.trim() && !imageCheckpointStatus?.token_configured)} onclick={prepareImageCheckpoints}>
          {imageCheckpointStatus?.preparing ? '모델 준비 중…' : imageCheckpointStatus?.ready && imageCheckpointStatus?.identity_runtime?.heretic_ready ? '파일 다시 확인' : '선택 모델 준비'}
        </button>
        {#if imageCheckpointStatus?.nvfp4_conversion}
          {@const conversion = imageCheckpointStatus.nvfp4_conversion}
          <hr>
          <h4>V2·V4 NVFP4 변환</h4>
          <p>BF16 원본을 받은 뒤 GB10 네이티브 NVFP4로 변환하고, 실제 512px 생성을 통과한 파일만 선택 가능하게 만듭니다.</p>
          <div class="model-variant-list">
            {#each conversion.variants || [] as variant}
              <label>
                <input type="checkbox" checked={nvfp4Selection[variant.id]} onchange={(event) => nvfp4Selection = { ...nvfp4Selection, [variant.id]: event.currentTarget.checked }}>
                <span>{variant.id === 'ray-v2' ? 'Ray Artshoot V2' : 'Ray Artshoot V4'}<small>{variant.validated ? `검증 완료 · ${formatBytes(variant.converted_size_bytes)}` : variant.converted_ready ? '변환됨 · 생성 검증 필요' : variant.source_ready ? 'BF16 준비됨 · 변환 대기' : `BF16 ${formatBytes(variant.source_size_bytes)}`}</small></span>
                <a href={variant.source} target="_blank" rel="noreferrer">출처</a>
              </label>
            {/each}
          </div>
          {#if conversion.preparing}
            <div class="progress-track"><span style={`width:${conversion.total ? Math.min(100, conversion.done / conversion.total * 100) : 0}%`}></span></div>
            <small>{conversion.current} · {conversion.stage === 'download' ? 'BF16 다운로드' : conversion.stage === 'unload' ? '메모리 정리' : conversion.stage === 'convert' ? 'NVFP4 변환' : conversion.stage === 'validate' ? '생성 검증' : '준비'} · {conversion.stage === 'download' ? `${formatBytes(conversion.done)} / ${formatBytes(conversion.total)}` : `${conversion.done}/${conversion.total}`}</small>
          {/if}
          {#if conversion.error}<small class="model-setup-error">{conversion.error}</small>{/if}
          <label class="inline-check"><input type="checkbox" bind:checked={removeBF16Sources}> <span>검증 성공 후 BF16 원본 삭제</span></label>
          <button type="button" class="primary" disabled={convertingImageCheckpoints || conversion.preparing || imageCheckpointStatus?.preparing || (!civitaiToken.trim() && !imageCheckpointStatus?.token_configured)} onclick={convertImageCheckpointsNVFP4}>
            {conversion.preparing ? 'NVFP4 준비 중…' : '선택 모델 NVFP4 준비'}
          </button>
          <small>변환 프로필: <a href={conversion.profile_source} target="_blank" rel="noreferrer">출처</a> · 커밋 {conversion.profile_commit?.slice(0, 8)}</small>
        {/if}
      </section>

      <section class="settings-card">
        <h3>LTX 영상 모델 준비</h3>
        <p>저장된 Hugging Face 토큰을 사용해 SSH나 Compose 재시작 없이 공식 모델과 공개 Motion LoRA를 내려받습니다.</p>
        {#if videoModelStatus}
          <div class="storage-stats">
            <span><small>상태</small><strong>{videoModelStatus.ready ? '준비 완료' : videoModelStatus.preparing ? '다운로드 중' : '준비 필요'}</strong></span>
            <span><small>파일</small><strong>{videoModelStatus.ready_files}/{videoModelStatus.required_files}</strong></span>
            <span><small>Motion LoRA</small><strong>{videoModelStatus.motion_lora_ready ? '준비됨' : '대기'}</strong></span>
          </div>
          {#if videoModelStatus.error}<small class="model-setup-error">{videoModelStatus.error}</small>{/if}
        {/if}
        <small><a href="https://huggingface.co/Lightricks/LTX-2.5" target="_blank" rel="noreferrer">LTX-2.5 라이선스 동의 ↗</a> 후 다운로드 인증에 같은 계정의 read 토큰을 저장하세요.</small>
        <button type="button" class="primary" disabled={preparingVideoModels || videoModelStatus?.preparing || (!hfToken.trim() && !videoModelStatus?.token_configured && !videoModelStatus?.ready)} onclick={prepareVideoModels}>
          {videoModelStatus?.preparing ? '모델 준비 중…' : videoModelStatus?.ready ? '파일 다시 확인' : '모델 준비 시작'}
        </button>
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
          <label>기본 체크포인트<select bind:value={settings.image.default_checkpoint}><option value="official">Krea 2 Turbo · 공식 NVFP4</option><option value="chriscole-edit-v1.1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'chriscole-edit-v1.1')?.ready}>Krea 2 Turbo Edit v1.1 · FP8</option><option value="moody-v7" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-v7')?.ready}>Moody Krea 2 Mix V7 · NVFP4</option><option value="moody-cutie-v4" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-cutie-v4')?.ready}>Moody Cutie Mix V4 · NVFP4</option><option value="moody-amateur-v1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'moody-amateur-v1')?.ready}>Moody Amateur Mix V1 · NVFP4</option><option value="ray-v1" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v1')?.ready}>Ray Artshoot V1 · FP8</option><option value="ray-v2" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v2')?.ready}>Ray Artshoot V2 · FP8</option><option value="ray-v2-nvfp4" disabled={!imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === 'ray-v2')?.validated}>Ray Artshoot V2 · NVFP4</option><option value="ray-v3" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v3')?.ready}>Ray Artshoot V3 · INT8</option><option value="ray-v4" disabled={!imageCheckpointStatus?.variants?.find((item) => item.id === 'ray-v4')?.ready}>Ray Artshoot V4 · INT8</option><option value="ray-v4-nvfp4" disabled={!imageCheckpointStatus?.nvfp4_conversion?.variants?.find((item) => item.id === 'ray-v4')?.validated}>Ray Artshoot V4 · NVFP4</option></select><small>준비·검증된 체크포인트만 선택할 수 있습니다.</small></label>
          {#each imageModeChoices as mode}
            <div class="backend-setting"><strong>{imageModeMeta[mode].label}</strong><label>Endpoint<input type="url" bind:value={settings.image.backends[mode].endpoint} required></label><label>모델<input bind:value={settings.image.backends[mode].model} required></label></div>
          {/each}
          <div class="fields three">
            <label>기본 너비<input type="number" min="256" step="8" bind:value={settings.image.default_width}></label>
            <label>기본 높이<input type="number" min="256" step="8" bind:value={settings.image.default_height}></label>
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
            <label>Motion LoRA 기본값<select bind:value={settings.video.default_motion_lora_enabled}><option value={false}>꺼짐</option><option value={true}>켜짐</option></select></label>
            <label>Motion LoRA 강도<input type="number" min="0" max="1" step="0.05" disabled={!settings.video.default_motion_lora_enabled} bind:value={settings.video.default_motion_lora_strength}><small>권장 0.35~0.70 · 제안 0.50</small></label>
          </div>
          <small>설정을 저장하면 이후 영상 작업부터 즉시 적용됩니다. 모델 전환이 필요한 첫 작업에서만 파이프라인을 자동으로 다시 적재합니다.</small>
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
    <section class="history"><div class="section-title"><div><span>06</span><h2>생성 기록</h2></div>{#if jobs.some((job) => job.status !== 'queued' && job.status !== 'running')}<button class="quiet danger" disabled={deletingJob === 'all'} onclick={clearFinishedJobs}>모두 비우기</button>{/if}</div>
      <ResultPagination label="생성 기록" total={jobs.length} page={listPages.history} pageSize={listPageSizes.history} pageSizes={pageSizeOptions} sortOrder={listSortOrders.history} onPageChange={(page) => setListPage('history', page)} onPageSizeChange={(size) => setListPageSize('history', size)} onSortOrderChange={(order) => setListSortOrder('history', order)} />
      {#each pagedHistoryJobs as job (job.id)}<article><span class="kind">{kindLabels[job.kind] || job.kind}</span><div><button type="button" class="history-prompt" title="전체 내용 보기" onclick={() => promptModal = { title: `${kindLabels[job.kind] || job.kind} 작업`, detail: `${new Date(job.created_at).toLocaleString()} · ${statusLabels[job.status] || job.status}`, text: job.prompt }}>{job.prompt}</button><small>{new Date(job.created_at).toLocaleString()} · {statusLabels[job.status] || job.status}</small>{#if job.error}<em>{job.error}</em>{/if}</div><div class="job-actions">{#if job.output_url}<a href={job.output_url} target="_blank">열기 ↗</a>{/if}{#if job.kind === 'recognition' && (job.status === 'failed' || job.status === 'cancelled')}<button class="job-retry" disabled={retryingJob === job.id} onclick={() => retryJob(job)}>{retryingJob === job.id ? '재개 중…' : '재개'}</button>{:else if (job.kind === 'image' || job.kind === 'video') && job.status === 'failed'}<button class="job-retry" disabled={retryingJob === job.id || activeJobs().some((item) => item.kind === 'image' || item.kind === 'video')} onclick={() => retryJob(job)}>{retryingJob === job.id ? '재시도 중…' : '재시도'}</button>{/if}{#if job.status !== 'queued' && job.status !== 'running'}<button class="job-delete" disabled={deletingJob === job.id} onclick={() => deleteJob(job)}>삭제</button>{/if}</div></article>{:else}<div class="empty">아직 생성 기록이 없습니다.</div>{/each}
      <ResultPagination compact label="생성 기록" total={jobs.length} page={listPages.history} pageSize={listPageSizes.history} onPageChange={(page) => setListPage('history', page)} />
    </section>
  {/if}
</main>

{#if imageSequenceOpen}
  <div class="image-sequence-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget && !busy) imageSequenceOpen = false }}>
    <section class="image-sequence-modal" role="dialog" aria-modal="true" aria-label="연속 이미지 생성">
      <header>
        <div><strong>연속 이미지 생성</strong><small>첫 장을 만든 뒤, 직전 결과를 원본으로 사용해 다음 장면을 순서대로 만듭니다.</small></div>
        <div class="image-sequence-header-actions"><button type="button" class="image-sequence-example" disabled={busy} onclick={applyRobotSequenceExample}>로봇 3장 예제</button><button type="button" aria-label="닫기" disabled={busy} onclick={() => imageSequenceOpen = false}>×</button></div>
      </header>
      <div class="image-sequence-content">
        <div class="image-sequence-notice"><strong>현재 생성 설정 유지</strong><span>{imageForm.width}×{imageForm.height} · {kreaOptions.checkpoint} · {kreaOptions.steps} steps{#if kreaModules.style || kreaModules.userLora} · 선택한 LoRA 적용{/if}</span></div>
        <div class="image-sequence-base" class:ready={Boolean(imageSequenceBase)}>
          {#if imageSequenceBase}
            <img src={imageSequenceBase.url} alt="연속 생성 첫 장면"><span><small>첫 장면 준비됨</small><b title={imageSequenceBase.prompt}>{imageSequenceBase.name}</b><em>이 이미지 위에 장면별 마스크를 칠할 수 있습니다.</em></span><button type="button" onclick={() => { clearImageSequenceMasks(); imageSequenceBase = null }}>새로 생성</button>
          {:else}
            <span><small>첫 장면</small><b>프롬프트로 새로 생성</b><em>실제 이미지 위에 영역을 칠하려면 기존 결과를 첫 장으로 선택하세요.</em></span><button type="button" onclick={() => recentImagePickerTarget = 'sequenceBase'}>생성 이미지 선택</button>
          {/if}
        </div>
        {#if imageSequenceBlockedMessage()}<div class="image-sequence-warning">{imageSequenceBlockedMessage()}</div>{/if}
        <label class="image-sequence-strength"><span><strong>장면 연속성</strong><small>낮추면 동작·구도 변화가 커지고, 높이면 직전 장면을 더 강하게 유지합니다.</small></span><input type="range" min="0.4" max="1.2" step="0.05" bind:value={imageSequenceStrength}><b>{Number(imageSequenceStrength).toFixed(2)}</b></label>
        <ol class="image-sequence-scenes">
          {#each imageSequencePrompts as prompt, index}
            <li>
              <div class="image-sequence-scene-heading"><span>{index + 1}</span><label for={`sequence-scene-${index}`}>{index === 0 ? '첫 장면 · 전체 묘사' : `장면 ${index + 1} · 직전 장면에서 바꿀 내용`}</label>{#if index > 1}<button type="button" aria-label={`장면 ${index + 1} 제거`} onclick={() => removeImageSequenceScene(index)}>×</button>{/if}</div>
              <textarea id={`sequence-scene-${index}`} rows="3" value={prompt} placeholder={index === 0 ? '인물·장소·조명·구도를 포함한 첫 장면' : '예: 같은 인물이 창가로 걸어가며 카메라가 옆으로 이동한다'} oninput={(event) => updateImageSequencePrompt(index, event.currentTarget.value)}></textarea>
              {#if index > 0}
                <div class="image-sequence-scene-tools">
                  <span>{#if imageSequenceMaskPreviews[index]}<img src={imageSequenceMaskPreviews[index]} alt={`장면 ${index + 1} 마스크`}>{:else}<i class={`image-sequence-region-preview region-${imageSequenceRegions[index] || 'all'}`}><span></span></i>{/if}<b>{imageSequenceRegions[index] === 'custom' ? '직접 칠한 영역' : imageSequenceRegionOption(imageSequenceRegions[index]).label}</b></span>
                  <button type="button" class="paint" disabled={!imageSequenceBase} title={imageSequenceBase ? '첫 장면 위에 변경 영역을 직접 칠합니다.' : '먼저 생성 이미지를 첫 장으로 선택하세요.'} onclick={() => imageSequenceMaskEditorIndex = index}>영역 칠하기</button>
                  <button type="button" onclick={() => imageSequenceRegionPicker = index}>빠른 영역</button>
                </div>
                <small>{imageSequenceRegions[index] === 'all' ? '전체 이미지를 프롬프트로 편집합니다.' : '마스크 밖은 직전 장면을 그대로 보존합니다.'}</small>
              {/if}
            </li>
          {/each}
        </ol>
        {#if imageSequencePrompts.length < 6}<button type="button" class="image-sequence-add" onclick={addImageSequenceScene}>+ 장면 추가</button>{/if}
      </div>
      <footer>
        <small>2~6장 · 영상 탭에는 자동 배치하지 않습니다.</small>
        <button type="button" class="quiet" disabled={busy} onclick={() => imageSequenceOpen = false}>닫기</button>
        <button type="button" class="primary" disabled={busy || Boolean(imageSequenceBlockedMessage()) || imageSequencePrompts.some((prompt) => !prompt.trim())} onclick={() => generateImage(imageSequencePrompts)}>{busy ? '큐에 추가 중…' : imageSequenceBase ? `나머지 ${imageSequencePrompts.length - 1}장 생성` : `${imageSequencePrompts.length}장 생성`}</button>
      </footer>
    </section>
  </div>
{/if}

{#if imageSequenceRegionPicker >= 0}
  <div class="image-sequence-region-backdrop" role="presentation" onclick={(event) => { if (event.target === event.currentTarget) imageSequenceRegionPicker = -1 }}>
    <section class="image-sequence-region-modal" role="dialog" aria-modal="true" aria-label="변경 허용 영역 선택">
      <header><div><strong>변경 허용 영역</strong><small>초록색 영역만 새로 그립니다 · 왼쪽·오른쪽은 화면 기준</small></div><button type="button" aria-label="닫기" onclick={() => imageSequenceRegionPicker = -1}>×</button></header>
      <div class="image-sequence-region-grid">
        {#each imageSequenceRegionOptions as option}
          <button type="button" class:selected={imageSequenceRegions[imageSequenceRegionPicker] === option.id} onclick={() => updateImageSequenceRegion(imageSequenceRegionPicker, option.id)}>
            <i class={`image-sequence-region-map region-${option.id}`}><span></span></i>
            <b>{option.label}</b><small>{option.description}</small>
          </button>
        {/each}
      </div>
    </section>
  </div>
{/if}

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
<MaskEditor open={imageSequenceMaskEditorIndex >= 1} source={imageSequenceBase?.url || ''} existingMask={imageSequenceMaskPreviews[imageSequenceMaskEditorIndex] || ''} title={`장면 ${imageSequenceMaskEditorIndex + 1} 변경 허용 영역`} description="빨간 영역만 다음 장면에서 새로 그립니다. 움직이기 전 위치와 이동할 위치를 함께 넉넉히 칠하세요." outputName={`sequence-scene-${imageSequenceMaskEditorIndex + 1}-mask.png`} onApply={useImageSequenceMask} onClose={() => imageSequenceMaskEditorIndex = -1} />
<CannyEditor open={cannyEditorOpen} source={kreaNK2EPreview} preprocessed={kreaNK2EPreprocessed} onApply={useCannyMap} onClose={() => cannyEditorOpen = false} />
<ImageModal image={imageModal} onGarmentExtract={openGarmentExtractorFromModal} onClose={() => imageModal = null} />
<GarmentExtractorModal open={garmentExtractorOpen} jobs={imageJobs} initialJob={garmentExtractorInitialJob} onSubmit={submitGarmentExtraction} onClose={() => { garmentExtractorOpen = false; garmentExtractorInitialJob = null }} />
<VideoModal video={videoModal} onClose={() => videoModal = null} />
<SubtitleModal result={subtitleModal} onClose={() => subtitleModal = null} />
<AudioModal audio={audioModal} onClose={() => audioModal = null} />
<PromptModal prompt={promptModal} onClose={() => promptModal = null} />
<AssistantChat state={assistantState} onActions={applyAssistantActions} onExecute={executeAssistantOperation} getVisualContext={videoAssistantVisualContext} />
<PromptExamplesModal open={promptExamplesOpen} examples={filterPromptPresets} selectedID={promptExamplesTarget === 'video' ? videoPromptPreset : filterPromptPreset} officialSource={kreaPromptGuideSource} communitySource={filterPromptSource} vibeSource={vibePromptGuideSource} wildcardSource={wildcardPromptSource} onApply={applyPromptExample} onClose={() => promptExamplesOpen = false} />
<RecentImagePicker open={Boolean(recentImagePickerTarget)} title={recentImagePickerTarget === 'sequenceBase' ? '연속 생성 첫 장면 선택' : recentImagePickerTarget === 'identityReference' ? `${identityUI.secondary} 선택` : recentImagePickerTarget === 'depth' ? '자세·구도 이미지 선택' : recentImagePickerTarget === 'nk2e' ? '편집·윤곽 이미지 선택' : recentImagePickerTarget === 'anypaint' ? '부분 수정·확장 원본 선택' : recentImagePickerTarget === 'styleReference' ? '스타일 참조 이미지 추가' : recentImagePickerTarget === 'vision' ? '내용·구도 참조 이미지 추가' : `${identityUI.primary} 선택`} jobs={imageJobs} selectedRef={recentImagePickerTarget === 'sequenceBase' ? (imageSequenceBase?.ref || '') : recentImagePickerTarget === 'identityReference' ? (kreaIdentityReference?.ref || '') : recentImagePickerTarget === 'depth' ? (kreaDepthImage?.ref || '') : recentImagePickerTarget === 'nk2e' ? (kreaNK2EImage?.ref || '') : recentImagePickerTarget === 'anypaint' ? (kreaAnyPaintImage?.ref || '') : recentImagePickerTarget === 'identity' ? (kreaIdentityImage?.ref || '') : ''} onSelect={useRecentModuleImage} onClose={() => recentImagePickerTarget = ''} />
<PresetImagePicker open={Boolean(presetImagePickerTarget)} title={presetImagePickerTarget === 'identityReference' ? `${identityUI.secondary} 프리셋 선택` : presetImagePickerTarget === 'depth' ? '자세·구도 프리셋 선택' : presetImagePickerTarget === 'nk2e' ? '편집·윤곽 프리셋 선택' : presetImagePickerTarget === 'anypaint' ? '부분 수정·확장 원본 프리셋' : presetImagePickerTarget === 'styleReference' ? '스타일 참조 프리셋 추가' : presetImagePickerTarget === 'vision' ? '내용·구도 참조 프리셋 추가' : `${identityUI.primary} 프리셋 선택`} examples={filterPromptPresets} initialTab={presetImagePickerTarget === 'depth' || presetImagePickerTarget === 'nk2e' ? 'pose' : 'example'} onSelect={usePresetModuleImage} onClose={() => presetImagePickerTarget = ''} />
<RemoteImageModal open={Boolean(remoteImageTarget)} title={remoteImageTitles[remoteImageTarget] || 'URL 이미지 가져오기'} append={remoteImageTarget === 'vision' || remoteImageTarget === 'styleReference' || remoteImageTarget === 'identityReference'} onImport={useRemoteModuleImage} onClose={() => remoteImageTarget = ''} />
<RecentImagePicker open={Boolean(videoImagePickerTarget)} title={videoConditionTitle(videoImagePickerTarget)} jobs={imageJobs} selectedRef="" onSelect={useRecentVideoImage} onClose={() => videoImagePickerTarget = ''} />
<RemoteImageModal open={Boolean(videoRemoteImageTarget)} title={videoConditionTitle(videoRemoteImageTarget, ' URL 가져오기')} onImport={useRemoteVideoImage} onClose={() => videoRemoteImageTarget = ''} />
