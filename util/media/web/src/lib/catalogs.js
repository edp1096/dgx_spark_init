import { sogniPromptPresets } from '../sogniPromptPresets.js'



export const identityPreserveCatalog = [
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

export const defaultIdentityPreserveItems = ['identity', 'face', 'hair', 'body', 'clothing', 'pose', 'background', 'lighting', 'composition', 'untouched']

export const checkpointDisplayChoices = [
    ['chriscole-edit-v1.1', 'Krea 2 Turbo Edit v1.1 · FP8'],
    ['moody-v7', 'Moody Krea 2 Mix V7 · NVFP4'], ['moody-cutie-v4', 'Moody Cutie Mix V4 · NVFP4'], ['moody-amateur-v1', 'Moody Amateur Mix V1 · NVFP4'],
    ['ray-v1', 'Ray Artshoot V1 · FP8'], ['ray-v2', 'Ray Artshoot V2 · FP8'], ['ray-v2-nvfp4', 'Ray Artshoot V2 · NVFP4'],
    ['ray-v3', 'Ray Artshoot V3 · INT8'], ['ray-v4', 'Ray Artshoot V4 · INT8'], ['ray-v4-nvfp4', 'Ray Artshoot V4 · NVFP4']
  ]

export const pageSizeOptions = [8, 10, 20, 50, 100]

export const imagePageSizeOptions = [8, 10, 12, 16, 20, 24, 28, 50, 100]

export const engineMeta = {
    video: ['video', 'LTX'],
    speech: ['speech', 'TTS'],
    recognition: ['media', 'Media'],
    lora: ['image_create', 'LoRA 관리']
  }

export const engineStatusCatalog = [
    ['image_create', 'Krea 2 이미지'], ['video', 'LTX 영상'], ['speech', 'Qwen3 TTS'],
    ['recognition', 'Qwen3 ASR'], ['prompt', '프롬프트·번역'], ['upscale', 'SeedVR2 고화질'], ['garment', '의상 추출'], ['faceswap', 'ReActor 얼굴 교체'],
    ['media', '미디어·FFmpeg']
  ]

export const imageModeMeta = {
    create: { label: 'Krea 2 Turbo', short: '생성·고급', engine: 'image_create', help: '새 이미지 생성과 Identity·Depth·LoRA·부분 수정 등의 기능을 조합합니다.' },
    edit: { label: 'FLUX.2 Klein 4B', short: '원본 수정', engine: 'image_edit', help: '하나 이상의 참조 이미지를 바탕으로 내용과 스타일을 변경합니다.' },
    detail_enhance: { label: '디테일 재해석', short: 'Krea Detail', engine: 'image_create', help: 'Ostris Edit LoRA로 원본을 다시 그려 세부 묘사를 강화합니다.' },
    upscale: { label: '고화질', short: 'SeedVR2', engine: 'upscale', help: '완성된 이미지를 SeedVR2로 복원하고 확대합니다.' },
    garment_extract: { label: '의상 추출', short: 'Garment', engine: 'garment', help: '의상만 투명 PNG와 마스크로 분리합니다.' },
    face_swap: { label: 'ReActor 얼굴 교체', short: 'ReActor', engine: 'faceswap', help: 'INSWapper가 생성 모델의 재해석 없이 얼굴 영역을 직접 교체합니다.' }
  }

export const imageModeChoices = ['create']

export const kreaModuleLabels = {
    identity: '원본 수정', depth: '자세·구도', nk2e: '실험 편집·윤곽', anypaint: '부분 수정·확장',
    style: '스타일 LoRA', userLora: '사용자 LoRA', styleReference: '스타일 이미지 참조', vision: '내용·구도 참조'
  }

export const identityPresetUI = {
    '': { primary: '편집할 원본', primaryHint: '변경할 인물이나 장면', secondary: '보조 참조', secondaryHint: '얼굴·인물·의상·사물 제공', showSecondary: true, guide: '메인 프롬프트에 바꿀 내용을 직접 입력하세요.' },
    restage: { primary: '인물 원본', primaryHint: '다른 장면에 배치할 인물', showSecondary: false, guide: '메인 프롬프트에 새로운 자세와 장면을 입력하세요.' },
    sheet: { primary: '인물 원본', primaryHint: '시트로 만들 인물', secondary: '추가 외형 참조', secondaryHint: '다른 각도나 복장 자료 · 선택 사항', showSecondary: true, guide: '같은 인물의 2×2 시트를 자동으로 구성합니다.' },
    faceSwap: { primary: '편집할 원본', primaryHint: '몸·장면을 유지할 이미지', secondary: '가져올 얼굴', secondaryHint: '교체할 얼굴이 선명한 이미지', secondaryRequired: true, showSecondary: true, guide: '첫 이미지의 얼굴만 두 번째 이미지의 얼굴로 교체합니다.' },
    headSwap: { primary: '편집할 원본', primaryHint: '몸·장면을 유지할 이미지', secondary: '가져올 머리', secondaryHint: '머리 위주의 근접 사진 · 옷은 최대한 제외', secondaryRequired: true, showSecondary: true, guide: 'BFS Head Swap V1.1이 얼굴·머리카락·두상 전체를 교체합니다. 전신 참조는 의상까지 섞일 수 있으므로 머리 중심의 사진을 사용하세요.' },
    personSwap: { primary: '배경·장면 원본', primaryHint: '배경과 구도를 유지할 이미지', secondary: '가져올 인물', secondaryHint: '장면에 넣을 인물 이미지', secondaryRequired: true, showSecondary: true, guide: '첫 이미지의 장면에 두 번째 이미지의 인물을 배치합니다.' },
    tryon: { primary: '편집할 인물 원본', primaryHint: '옷을 바꿀 인물 이미지', secondary: '참고할 의상', secondaryHint: '입힐 옷이나 착장 이미지', secondaryRequired: true, showSecondary: true, guide: '두 번째 이미지의 의상을 참고해 첫 인물의 옷을 변경합니다.' },
    replace: { primary: '편집할 원본', primaryHint: '일부를 교체할 이미지', secondary: '교체 요소 참조', secondaryHint: '새로 넣을 사물·소재 · 선택 사항', showSecondary: true, guide: '메인 프롬프트와 변경 허용 영역으로 교체할 부분을 지정하세요.' }
  }

export const imageAspectRatios = [
    ['1:1', 1, '정사각'], ['3:4', 3 / 4, '세로'], ['4:3', 4 / 3, '가로'],
    ['2:3', 2 / 3, '세로 사진'], ['3:2', 3 / 2, '가로 사진'], ['9:16', 9 / 16, '세로 화면'], ['16:9', 16 / 9, '가로 화면']
  ]

export const outputLabels = { srt: 'SRT', vtt: 'VTT', timestamped_txt: '타임코드 TXT', txt: '일반 TXT' }

export const kindLabels = { image: '이미지', video: '영상', speech: '음성', recognition: '자막' }

export const statusLabels = { queued: '대기 중', running: '처리 중', completed: '완료', failed: '실패', cancelled: '중지됨' }

export const languageCodes = { Korean: 'ko', Japanese: 'ja', English: 'en', Chinese: 'zh' }

export const translationLanguages = [
    'Korean', 'Japanese', 'English', 'Chinese', 'Traditional Chinese',
    'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Russian',
    'Arabic', 'Hindi', 'Vietnamese', 'Thai', 'Indonesian', 'Turkish',
    'Dutch', 'Polish', 'Ukrainian'
  ]

export const filterPromptSource = 'https://www.sogni.ai/loras/krea2-filter-bypass-2#examples'

export const kreaPromptGuideSource = 'https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md'

export const vibePromptGuideSource = 'https://vibeart.app/blog/z-image-turbo-prompt-guide'

export const wildcardPromptSource = 'https://huggingface.co/datasets/Crocody/mymuse/tree/main/Wildcards'

export const officialPromptPresets = [
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

export const communityPromptPresets = [
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

export const vibePromptPresets = [
    { id: 'vibe-hanbok', label: '한복 인물 · 복합 지시', category: 'portrait', source: vibePromptGuideSource, image: 'vibe-hanbok.png', prompt: `A young Korean woman in pastel pink traditional Joseon hanbok, with intricate embroidery. Perfect makeup. Delicate indigo braided hair, elegantly adorned with red flowers and beads in exquisite detail. A woman with a hairpin tucked behind her head, holding a round folding fan with a tree and a bird. Neon lightning lamp, a bright yellow light floats above with the left hand open. A softly lit outdoor night background, with the silhouette in the draft visible over the vast, drifting Gyeongbokgung Palace, and a variety of distant lights faintly blurred. Realistic, cinematic in feel, as shown in the photos, with ultra-high-resolution detail.` },
    { id: 'vibe-skincare', label: '이중언어 화장품 포스터', category: 'graphic', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/h3d38jbc0m21y5rl25w3mx5t-ce22254b3fa2c97f5173315c04ed754b.png', prompt: `Luxury skincare poster. Frosted glass serum bottle on a cream stone pedestal, soft gold rim light, premium beauty campaign composition, highly realistic product photography. The poster contains exactly four readable text elements only: Chinese "晨光精华", English "Morning Serum", Chinese "轻盈修护", English "Light Repair". Elegant high-end typography, balanced spacing, no extra words, no logo, no watermark.` },
    { id: 'vibe-coffee', label: '이중언어 커피 패키지', category: 'photo', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/s4s75fl0u9cvenhpe99ojruv-c5d5d71e040b79e7034036615951ed41.png', prompt: `Photorealistic premium coffee bag packaging on a neutral warm-gray studio background, matte paper bag, subtle valve, realistic shadows. The front label contains only four readable text elements: Chinese "云南咖啡", English "Yunnan Coffee", Chinese "日晒处理", English "Natural Process". Accurate printed typography on the bag surface, no extra text, no logo, high-end packaging photography.` },
    { id: 'vibe-storefront', label: '이중언어 매장 간판', category: 'photo', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/or1ibt5sjhvzuhzgvcufnqmv-f055ec731b4a6e393f6456ae6a8b3978.png', prompt: `Photorealistic modern tea bar storefront at dusk, clean glass facade, warm interior lighting, elegant urban street scene. The storefront signage contains only short readable bilingual text: Chinese "山茶" and English "Mountain Tea". Menu board visible through the window contains only two short readable items: Chinese "乌龙" and English "Oolong". No other text, no logo clutter, premium branding photography.` },
    { id: 'vibe-mid-autumn', label: '문화적으로 일관된 명절 정물', category: 'photo', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/n98edrhzzbm8guoc8odj0pln-37d2d13fce2ef6b07891c9a85d1551a3.png', prompt: `Culturally coherent Mid-Autumn Festival still life in an elegant Chinese home interior: mooncakes on a porcelain plate, a small white tea set, osmanthus blossoms, rabbit paper-cut decoration, warm lantern glow, and a full moon visible through a round window. The arrangement should feel authentic, harmonious, and logically composed, with no random clutter, no text, no watermark. Photorealistic editorial photography.` },
    { id: 'vibe-seven-objects', label: '정확히 7개 · 지정 위치', category: 'photo', source: vibePromptGuideSource, image: 'https://cdn.vibeart.app/canvas/w160lbagvsshc8jzomd8xpvz-b51bd7db25de008cc5b99d609d481009.png', prompt: `Top-down studio tabletop on a charcoal surface. Exactly seven objects and nothing else: a blue notebook in the top left, silver fountain pen in the top center, black camera in the top right, green ceramic tea cup in the middle left, white earbuds case in the middle center, red passport in the middle right, and a yellow keychain centered below them. Clean shadows, precise spacing, photorealistic, no text, no logo.` }
  ]

export const filterPromptPresets = [...officialPromptPresets, ...communityPromptPresets, ...vibePromptPresets]

export const kreaStyleCatalog = [
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

export const recognitionLanguages = [
    ['Auto', 'Auto · 단일 언어'],
    ['AutoMultilingual', 'Auto · 다중 언어'],
    ['Korean', 'Korean'], ['English', 'English'], ['Chinese', 'Chinese'], ['Japanese', 'Japanese']
  ]

export const videoResolutionPresets = [
    { id: 'fast', label: '빠르게', width: 768, height: 512, hint: 'Dense' },
    { id: 'standard', label: '기본', width: 1280, height: 704, hint: '자동' },
    { id: 'high', label: '업스케일', width: 1920, height: 1088, hint: 'SOL Attn' }
  ]

export const remoteImageTitles = {
    identity: '원본 이미지를 URL에서 가져오기',
    identityReference: '보조 참조를 URL에서 가져오기',
    depth: '자세·구도 이미지를 URL에서 가져오기',
    nk2e: '편집·윤곽 이미지를 URL에서 가져오기',
    anypaint: '부분 수정·확장 원본을 URL에서 가져오기',
    styleReference: '스타일 참조를 URL에서 추가',
    vision: '내용·구도 참조를 URL에서 추가'
  }

export const imageSequenceRegionOptions = [
    { id: 'all', label: '전체 수정', description: '이미지 전체를 편집' },
    { id: 'left', label: '화면 왼쪽', description: '왼쪽 절반을 넓게 수정' },
    { id: 'right', label: '화면 오른쪽', description: '오른쪽 절반을 넓게 수정' },
    { id: 'upper', label: '화면 상단', description: '위쪽 절반을 넓게 수정' },
    { id: 'lower', label: '화면 하단', description: '아래쪽 절반을 넓게 수정' },
    { id: 'left-arm', label: '화면 왼쪽 팔', description: '왼쪽 팔의 이전·새 위치를 포함' },
    { id: 'right-arm', label: '화면 오른쪽 팔', description: '오른쪽 팔의 이전·새 위치를 포함' }
  ]
