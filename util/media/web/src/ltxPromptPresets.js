const sources = {
  official: 'https://github.com/Lightricks/LTX-Video',
  blog: 'https://ltx.io/blog/ltx-2-3-prompt-guide',
  discussion: 'https://huggingface.co/Lightricks/LTX-Video/discussions/7',
  maxvideo: 'https://maxvideoai.com/examples/ltx',
  replicate: 'https://replicate.com/lightricks/ltx-video/examples',
  rundiffusion: 'https://www.rundiffusion.com/ltx-2-prompt-guide'
}

// Only the official repository is Apache-2.0. The remaining prompts below are
// independently written from general shot/pacing ideas; none copy source prose.
export const ltxPromptSources = [
  { key: 'ltx-official', label: 'LTX 공식 · Apache-2.0', shortLabel: 'LTX 공식', url: sources.official, licensed: true },
  { key: 'ltx-blog', label: 'LTX 2.3 가이드 · 재작성', shortLabel: 'LTX 가이드', url: sources.blog },
  { key: 'ltx-discussion', label: 'HF 토론 · 재작성', shortLabel: 'HF 토론', url: sources.discussion },
  { key: 'ltx-maxvideo', label: 'MaxVideoAI · 재작성', shortLabel: 'MaxVideo', url: sources.maxvideo },
  { key: 'ltx-replicate', label: 'Replicate · 재작성', shortLabel: 'Replicate', url: sources.replicate },
  { key: 'ltx-rundiffusion', label: 'RunDiffusion · 재작성', shortLabel: 'RunDiffusion', url: sources.rundiffusion },
  { key: 'ltx-wildcard', label: 'Crocody Muse × Style', shortLabel: 'Muse × Style', url: 'https://huggingface.co/datasets/Crocody/mymuse/tree/main/Wildcards', linkOnly: true }
]

export const ltxPromptPresets = [
  {
    id: 'ltx-official-night-market', label: '야시장 · 연속 트래킹', category: 'photo',
    sourceKey: 'ltx-official', sourceLabel: 'LTX 공식 · Apache-2.0', source: sources.official,
    previewIcon: '🏮', previewTone: 'amber',
    prompt: `A food vendor slides a steaming bowl across a crowded night-market counter while customers move through the narrow aisle behind her. The camera begins beside the bubbling pot in a medium close-up, tracks smoothly along the counter with the bowl, then settles on the waiting customer as he lifts it with both hands. Red paper lanterns sway above the stalls, steam catches the warm light, and rainwater reflects passing silhouettes on the pavement. Metal utensils clink, broth simmers, vendors call from farther down the lane, and a scooter hum fades past the market entrance.`
  },
  {
    id: 'ltx-official-cliff-i2v', label: '절벽 여행자 · I2V', category: 'photo',
    sourceKey: 'ltx-official', sourceLabel: 'LTX 공식 · Apache-2.0', source: sources.official,
    previewIcon: '🏔️', previewTone: 'blue',
    prompt: `The traveler shifts her weight forward and takes two careful steps toward the overlook as wind pulls at her jacket and loose hair. The camera follows from behind at waist height, then arcs gently to her left to reveal the valley beyond her shoulder. Low clouds drift between the ridges and nearby grass bends in uneven gusts. Keep the person, clothing, terrain, and lighting established by the first frame consistent. Wind moves across the microphone with distant birds and a faint echo from the valley.`
  },
  {
    id: 'ltx-official-keyframe-gallery', label: '전시장 · 키프레임 연결', category: 'photo',
    sourceKey: 'ltx-official', sourceLabel: 'LTX 공식 · Apache-2.0', source: sources.official,
    previewIcon: '🖼️', previewTone: 'violet',
    prompt: `A visitor walks at an even pace through the gallery while the camera glides backward in front of her, preserving her identity and the spatial continuity of the room. As she reaches the next artwork, she turns her head toward it and the camera eases sideways into the framing established by the final image. Other visitors cross softly out of focus without blocking her face. Ceiling lights remain stable across the transition, footsteps echo on the polished floor, and quiet room tone continues without a cut.`
  },
  {
    id: 'ltx-blog-rainy-tram', label: '빗속 전차 · 대사와 환경음', category: 'photo',
    sourceKey: 'ltx-blog', sourceLabel: 'LTX 2.3 가이드 · 독자 재작성', source: sources.blog,
    previewIcon: '🚋', previewTone: 'teal',
    prompt: `A tired conductor leans from the doorway of a stopped tram as rain runs down its painted metal sides. In a single handheld medium shot, the camera approaches through the wet street while he checks the empty platform, raises one gloved hand, and says in Korean, "마지막 전차입니다." The doors begin to close and he steps back inside. Cool streetlight reflections slide across the windows, the overhead cable crackles once, rain taps steadily on the roof, and the bell rings above the low electric motor.`
  },
  {
    id: 'ltx-blog-clay-repair', label: '클레이 애니 수리점', category: '3d',
    sourceKey: 'ltx-blog', sourceLabel: 'LTX 2.3 가이드 · 독자 재작성', source: sources.blog,
    previewIcon: '🧰', previewTone: 'orange',
    prompt: `In a handmade claymation repair shop, a palm-sized mechanic tightens the final screw on a dented toy robot. The mechanic braces both feet, turns the oversized wrench, and rocks backward when the robot suddenly lights up and waves. The camera holds a close tabletop view, then makes a short stop-motion push toward the blinking robot. Fingerprints remain visible in the clay, a desk lamp flickers warmly, tiny gears chatter, the wrench taps the bench, and the mechanic gives a surprised little laugh.`
  },
  {
    id: 'ltx-discussion-greenhouse', label: '온실 로봇 · 순차적 동작', category: 'fantasy',
    sourceKey: 'ltx-discussion', sourceLabel: 'HF 토론 · 독자 재작성', source: sources.discussion,
    previewIcon: '🤖', previewTone: 'green',
    prompt: `A compact gardening robot rolls between rows of plants in a glass greenhouse, stops beside a drooping seedling, and unfolds a narrow watering arm. It releases a brief spray, scans the leaves with a blue light, then retracts the arm and continues down the aisle. The camera starts overhead to show the orderly rows, descends behind the robot, and follows at wheel height. Condensation beads on the glass, leaves tremble under ventilation fans, water patters onto soil, servos click, and morning birds sound faintly outside.`
  },
  {
    id: 'ltx-maxvideo-skater-anchor', label: '스케이터 · 시작 프레임 유지', category: 'photo',
    sourceKey: 'ltx-maxvideo', sourceLabel: 'MaxVideoAI · 독자 재작성', source: sources.maxvideo,
    previewIcon: '🛹', previewTone: 'red',
    prompt: `Treat the supplied image as the exact opening frame and preserve the skater's face, clothes, board, and street layout. He pushes off once, gathers speed, and rolls over a shallow puddle while the camera tracks parallel at knee height. He crouches and performs one clean ollie over the curb, landing in the same direction of travel without a cut. Storefront reflections stretch across the wet asphalt, wheels rattle over paving seams, the board snaps against the ground, and passing traffic remains softly blurred in the distance.`
  },
  {
    id: 'ltx-maxvideo-pottery-macro', label: '도자기 유약 · 매크로', category: 'photo',
    sourceKey: 'ltx-maxvideo', sourceLabel: 'MaxVideoAI · 독자 재작성', source: sources.maxvideo,
    previewIcon: '🏺', previewTone: 'rose',
    prompt: `An extreme macro shot follows a brush loaded with cobalt glaze as it touches a pale ceramic bowl and leaves a glossy blue trail around the rim. The potter rotates the bowl slowly with the other hand while the camera circles in the opposite direction, keeping the brush tip in sharp focus. Window light travels across the wet glaze, fine clay dust remains visible on the fingers, bristles whisper over ceramic, and the wheel produces a soft mechanical hum. Motion stays precise and unhurried with no jump cuts.`
  },
  {
    id: 'ltx-replicate-library', label: '야간 도서관 · 돌리 인', category: 'photo',
    sourceKey: 'ltx-replicate', sourceLabel: 'Replicate · 독자 재작성', source: sources.replicate,
    previewIcon: '📚', previewTone: 'gold',
    prompt: `A librarian wheels a ladder along towering shelves after closing time, stops beneath a high row, and climbs three rungs to retrieve a weathered book. The camera begins in a wide symmetrical aisle, slowly dollies inward, then tilts upward with her hand as she pulls the volume free and a veil of dust falls through the lamplight. Her cardigan and hair move only from her climbing motion. Wooden wheels creak, pages rustle, an old clock ticks, and distant thunder rolls beyond the tall windows.`
  },
  {
    id: 'ltx-replicate-snow-fox', label: '설원 여우 · 느린 추적', category: 'animal',
    sourceKey: 'ltx-replicate', sourceLabel: 'Replicate · 독자 재작성', source: sources.replicate,
    previewIcon: '🦊', previewTone: 'ice',
    prompt: `A red fox walks cautiously across fresh snow at blue hour, pauses when it hears movement beneath the surface, and turns both ears toward the sound. The camera tracks low beside the fox with a long-lens wildlife look, then comes to a complete stop as the animal lowers its nose and exhales a visible cloud. Fine snow drifts between dark pine trunks, paws compress the powder naturally, branches knock softly in the wind, and the distant forest remains still.`
  },
  {
    id: 'ltx-rundiffusion-watchmaker', label: '시계 수리공 · 원테이크', category: 'photo',
    sourceKey: 'ltx-rundiffusion', sourceLabel: 'RunDiffusion · 독자 재작성', source: sources.rundiffusion,
    previewIcon: '⌚', previewTone: 'amber',
    prompt: `A watchmaker peers through a loupe while fitting a tiny balance wheel into an open pocket watch. Without cutting, the camera slides from a close view of brass tools across the bench to his steady hands, pushes closer as the mechanism begins to oscillate, then racks focus to his eye as he smiles almost imperceptibly. A green-shaded lamp casts one coherent pool of warm light, suspended dust moves through the beam, gears tick delicately, tweezers touch metal, and the quiet workshop hum remains constant.`
  },
  {
    id: 'ltx-rundiffusion-orbital-dock', label: '우주 도킹 · 스린러 아크', category: 'fantasy',
    sourceKey: 'ltx-rundiffusion', sourceLabel: 'RunDiffusion · 독자 재작성', source: sources.rundiffusion,
    previewIcon: '🛰️', previewTone: 'violet',
    prompt: `A small inspection craft approaches the rotating docking ring of an orbital station above the night side of Earth. The camera begins behind and above the craft, follows its curved flight path, then arcs underneath as maneuvering thrusters fire in short controlled bursts and the vehicle aligns with the illuminated port. The planet turns slowly below, navigation lights pulse at regular intervals, blue exhaust scatters through ice particles, radio static precedes a calm clearance tone, and the hull settles into the dock with a muted mechanical lock.`
  }
]
