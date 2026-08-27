import { writable } from 'svelte/store'

const robotExample = {
  prompts: [
    'Wide full-body shot of a small friendly orange robot centered beside a blue armchair in a softly lit modern studio, with clear empty space around its entire body from antenna tips to both feet. It has exactly two arms, one continuous orange metal faceplate, exactly two round black recessed eyes, one small curved black smile, and two thin antennae, with no display screen. Both arms rest naturally at its sides, clean 3D animated film style.',
    "Move the robot's right arm, which is on the left side of the image, from its side into a raised friendly waving pose. Replace the old lowered arm position completely; show this arm exactly once in its new raised position. Preserve the exact face, head, left arm, body, chair, camera, lighting, and background unchanged.",
    "Move the same raised right arm on the left side of the image down to a halfway-lowered position, as the next moment of the wave. Replace its previous raised position completely; show this arm exactly once in the new position. Preserve the exact face, head, left arm, body, chair, camera, lighting, and background unchanged."
  ],
  regions: ['all', 'left-arm', 'left-arm'],
  strength: 0.65
}

function initialState() {
  return {
    prompts: ['', ''],
    regions: ['all', 'all'],
    masks: [null, null],
    maskPreviews: ['', ''],
    base: null,
    strength: 0.8
  }
}

export class ImageSequenceController {
  constructor(urls = globalThis.URL) {
    this.urls = urls
    this.current = initialState()
    this.state = writable(this.current)
    this.unsubscribe = this.state.subscribe((value) => this.current = value)
  }

  setState(patch) {
    this.state.update((value) => ({ ...value, ...patch }))
  }

  setPrompts(prompts) { this.setState({ prompts: [...prompts] }) }
  setBase(base) { this.setState({ base }) }
  setStrength(strength) { this.setState({ strength: Number(strength) }) }

  reset(prompts = ['', '']) {
    this.clearMasks()
    this.setState({
      ...initialState(),
      prompts: [...prompts],
      regions: prompts.map(() => 'all'),
      masks: prompts.map(() => null),
      maskPreviews: prompts.map(() => '')
    })
  }

  applyRobotExample() {
    this.clearMasks()
    this.setState({
      prompts: [...robotExample.prompts],
      regions: [...robotExample.regions],
      masks: robotExample.prompts.map(() => null),
      maskPreviews: robotExample.prompts.map(() => ''),
      strength: robotExample.strength
    })
  }

  addScene() {
    if (this.current.prompts.length >= 6) return
    this.setState({
      prompts: [...this.current.prompts, ''],
      regions: [...this.current.regions, 'all'],
      masks: [...this.current.masks, null],
      maskPreviews: [...this.current.maskPreviews, '']
    })
  }

  removeScene(index) {
    if (this.current.prompts.length <= 2) return
    this.releasePreview(this.current.maskPreviews[index])
    this.setState({
      prompts: this.current.prompts.filter((_, itemIndex) => itemIndex !== index),
      regions: this.current.regions.filter((_, itemIndex) => itemIndex !== index),
      masks: this.current.masks.filter((_, itemIndex) => itemIndex !== index),
      maskPreviews: this.current.maskPreviews.filter((_, itemIndex) => itemIndex !== index)
    })
  }

  updatePrompt(index, prompt) {
    this.setState({ prompts: this.current.prompts.map((value, itemIndex) => itemIndex === index ? prompt : value) })
  }

  updateRegion(index, region) {
    this.releasePreview(this.current.maskPreviews[index])
    this.setState({
      regions: this.current.regions.map((value, itemIndex) => itemIndex === index ? region : value),
      masks: this.current.masks.map((value, itemIndex) => itemIndex === index ? null : value),
      maskPreviews: this.current.maskPreviews.map((value, itemIndex) => itemIndex === index ? '' : value)
    })
  }

  clearMasks() {
    this.current.maskPreviews.forEach((preview) => this.releasePreview(preview))
    this.setState({
      masks: this.current.prompts.map(() => null),
      maskPreviews: this.current.prompts.map(() => '')
    })
  }

  useMask(index, file) {
    if (index < 1 || !file) return false
    this.releasePreview(this.current.maskPreviews[index])
    this.setState({
      masks: this.current.masks.map((value, itemIndex) => itemIndex === index ? file : value),
      maskPreviews: this.current.maskPreviews.map((value, itemIndex) => itemIndex === index ? this.urls.createObjectURL(file) : value),
      regions: this.current.regions.map((value, itemIndex) => itemIndex === index ? 'custom' : value)
    })
    return true
  }

  releasePreview(preview) {
    if (preview) this.urls.revokeObjectURL(preview)
  }

  destroy() {
    this.current.maskPreviews.forEach((preview) => this.releasePreview(preview))
    this.unsubscribe?.()
  }
}

export function imageSequenceBlockedMessage({ mode, modules, moduleReason, width, height }) {
  if (mode !== 'create') return '연속 생성은 새 이미지 생성에서만 사용할 수 있습니다.'
  const incompatible = [
    [modules.identity, '원본 수정'], [modules.depth, '자세·구도'],
    [modules.vision, '내용·구도 참조'], [modules.styleReference, '스타일 참조'],
    [modules.nk2e, '편집·윤곽'], [modules.anypaint, '부분 수정·확장']
  ].filter(([enabled]) => enabled).map(([, label]) => label)
  if (incompatible.length) return `${incompatible.join(' · ')} 모듈을 끈 뒤 사용할 수 있습니다.`
  if (moduleReason) return moduleReason
  if (Number(width) * Number(height) > 2 * 1024 * 1024) return '연속 장면은 Identity Edit를 사용하므로 2MP 이하가 필요합니다.'
  return ''
}
