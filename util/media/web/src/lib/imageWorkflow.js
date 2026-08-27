export function isPureOutpaint({ modules, anypaintImage, anypaintMask, options }) {
  return modules.anypaint
    && Boolean(anypaintImage)
    && !anypaintMask
    && ['outpaint_left', 'outpaint_top', 'outpaint_right', 'outpaint_bottom'].some((key) => Number(options[key]) > 0)
}

export function implicitModulePrompt(input) {
  const { modules, identityPreset, anypaintMask } = input
  const identityActions = {
    restage: 'Place the same person in a new scene and apply the selected pose and composition',
    sheet: 'Create a clean 2x2 character sheet on a plain background: front view upper-left, three-quarter view upper-right, left profile lower-left, and back view lower-right',
    tryon: 'Use the complete outfit shown in the supporting clothing reference',
    replace: 'Replace only the selected object or region using the supporting reference',
    faceSwap: 'Replace only the face of the person in Image One with the face from Image Two',
    headSwap: 'Replace the entire head of the person in Image One with the head from Image Two',
    personSwap: 'Replace the entire person in Image One with the person from Image Two'
  }
  if (modules.identity && identityActions[identityPreset]) {
    const poseInstruction = modules.depth ? '. Apply the pose, body orientation, framing, and camera viewpoint from the pose reference' : ''
    return `${identityActions[identityPreset]}${poseInstruction}`
  }
  if (modules.identity && modules.depth) return 'Keep the original person and apply the pose, body orientation, framing, and camera viewpoint from the pose reference'
  if (modules.depth) return 'Create a coherent image that follows the supplied pose, depth structure, composition, and camera viewpoint'
  if (modules.vision) return 'Create a coherent image using the subject, content, and composition from the reference images'
  if (modules.styleReference) return 'Create a coherent image using the visual style, color, lighting, and texture from the style reference images'
  if (modules.nk2e) return 'Create a coherent edited image that follows the supplied structure and preserves natural visual detail'
  if (modules.anypaint && anypaintMask) return 'Regenerate the masked area naturally and blend it seamlessly with the unchanged original image'
  if (isPureOutpaint(input)) return 'Extend the original image naturally while preserving its subjects, style, lighting, perspective, and visual continuity'
  return ''
}

export function identityHasExtraUserPrompt({ enteredPrompt, implicitPrompt }) {
  const entered = enteredPrompt.trim()
  return Boolean(entered && entered !== implicitPrompt.trim())
}

export function rawImagePrompt({ enteredPrompt, implicitPrompt, modules, identityPreset, identityPreserveCustom }) {
  let change = enteredPrompt.trim() || implicitPrompt
  if (!modules.identity || identityPreset === 'tryon') return change
  while (/^change\s*:/i.test(change)) change = change.replace(/^change\s*:\s*/i, '').trim()
  const preserveAt = change.search(/(?:^|\n)preserve\s*:/i)
  if (preserveAt >= 0) change = change.slice(0, preserveAt).trim()
  const lines = [change]
  if (modules.depth && !/(?:pose|posture|body orientation|자세|포즈|구도)/i.test(change)) {
    lines.push('The person now holds the same pose shown in the pose reference.')
  }
  if (identityPreserveCustom.trim()) lines.push(`Keep ${identityPreserveCustom.trim()} unchanged.`)
  return lines.filter(Boolean).join('\n')
}

export function identityPreserveDefaults(preset, defaultItems, depthEnabled) {
  const defaults = {
    '': defaultItems,
    restage: ['identity', 'face', 'hair', 'body', 'clothing', 'untouched'],
    sheet: ['identity', 'face', 'hair', 'body', 'clothing'],
    faceSwap: ['hair', 'body', 'clothing', 'pose', 'background', 'lighting', 'composition', 'untouched'],
    headSwap: ['body', 'clothing', 'pose', 'background', 'lighting', 'composition', 'untouched'],
    personSwap: ['pose', 'background', 'lighting', 'composition', 'untouched'],
    tryon: ['identity', 'face', 'hair', 'body', 'pose', 'background', 'lighting', 'composition', 'untouched'],
    replace: defaultItems
  }
  const selected = [...(defaults[preset] || defaults[''])]
  return depthEnabled ? selected.filter((id) => id !== 'pose' && id !== 'composition') : selected
}

export function kreaModuleDisabledReason(input) {
  const { modules, identityUI, identityImage, identityReference, depthImage, visionImages, styleReferenceImages, styleSelections, userLoraSelections, nk2eImage, anypaintImage, anypaintMask, options } = input
  if (modules.identity && !identityImage) return `원본 수정의 ${identityUI.primary} 이미지를 선택하세요.`
  if (modules.identity && identityUI.secondaryRequired && !identityReference) return `원본 수정의 ${identityUI.secondary} 이미지를 선택하세요.`
  if (modules.depth && !depthImage) return '자세·구도 모듈의 구도 참조 이미지를 선택하세요.'
  if (modules.vision && visionImages.length === 0) return '내용·구도 참조 이미지를 선택하세요.'
  if (modules.styleReference && styleReferenceImages.length === 0) return '스타일 참조 이미지를 선택하세요.'
  if (modules.style && styleSelections.length === 0) return '적용할 스타일 LoRA를 하나 이상 선택하세요.'
  if (modules.userLora && userLoraSelections.length === 0) return '적용할 사용자 LoRA를 하나 이상 선택하세요.'
  if (modules.nk2e && !nk2eImage) return 'NK2E 편집·윤곽 모듈의 참조 이미지를 선택하세요.'
  if (modules.anypaint && !anypaintImage) return '부분 수정·확장 모듈의 원본 이미지를 선택하세요.'
  if (modules.anypaint && !anypaintMask && !['outpaint_left', 'outpaint_top', 'outpaint_right', 'outpaint_bottom'].some((key) => Number(options[key]) > 0)) return '수정 마스크를 선택하거나 확장할 방향을 지정하세요.'
  if (modules.vision && modules.identity) return '내용·구도 참조와 Identity는 아직 함께 사용할 수 없습니다.'
  if (modules.styleReference && Object.entries(modules).some(([name, enabled]) => name !== 'styleReference' && enabled)) return '스타일 이미지 참조는 현재 단독으로 사용하세요.'
  if (modules.nk2e && Object.entries(modules).some(([name, enabled]) => name !== 'nk2e' && enabled)) return 'NK2E 편집·윤곽은 현재 다른 Krea 모듈과 함께 사용할 수 없습니다.'
  if (modules.anypaint && Object.entries(modules).some(([name, enabled]) => name !== 'anypaint' && enabled)) return '부분 수정·확장은 현재 다른 Krea 모듈과 함께 사용할 수 없습니다.'
  return ''
}

export function imageDisabledReason({ busy, prompt, imageForm, references, moduleReason }) {
  if (busy) return '요청을 전송하고 있습니다.'
  if (!prompt.trim()) return '무엇을 만들지 프롬프트를 입력하세요.'
  if (imageForm.mode === 'edit' && references.length === 0) return '편집할 참조 이미지를 추가하세요.'
  if (imageForm.mode === 'control' && references.length !== 1) return 'Canny 제어 이미지 1장을 추가하세요.'
  return moduleReason
}

export function imageEnhancementActive({ enabled, prompt, structured, identityTryonWithoutUserPrompt }) {
  if (identityTryonWithoutUserPrompt) return false
  return enabled && prompt.trim() !== '' && !structured
}

export function imageEnhancementCurrent({ enhanced, source, current }) {
  return enhanced.trim() !== '' && source === current
}
