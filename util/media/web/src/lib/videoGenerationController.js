import { buildVideoGenerationForm } from './generationRequests.js'
import { videoEnhancementActive, videoEnhancementCurrent, videoInputKey } from './videoWorkflow.js'
import { framesForDuration, durationFromFrames, snapDimension } from './videoTiming.js'

export class VideoGenerationController {
  constructor({ api, actions }) {
    this.api = api
    this.actions = actions
  }

  state() { return this.actions.getState() }
  imageKey(image = this.state().startImage) { return videoInputKey(image) }

  enhancementActive(state = this.state()) {
    return videoEnhancementActive({ enabled: state.enhanceEnabled, image: state.startImage, visionEnabled: state.config?.prompt_enhancement.vision_enabled })
  }

  enhancementCurrent(state = this.state()) {
    return videoEnhancementCurrent({ enhanced: state.enhancedPrompt, source: state.enhancedSource, prompt: state.form.prompt, imageKey: state.enhancedImageKey, currentImageKey: this.imageKey(state.startImage) })
  }

  resetEnhancement() {
    this.actions.patch({ enhancedPrompt: '', enhancedSource: '', enhancedImageKey: '' })
  }

  async appendEnhancementImage(form, image) {
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

  async enhance() {
    const state = this.state()
    const original = state.form.prompt.trim()
    if (!original) return ''
    this.actions.patch({ enhancing: true, error: '' })
    try {
      const form = new FormData()
      form.append('prompt', original)
      form.append('mode', state.startImage ? 'i2v' : 't2v')
      await this.appendEnhancementImage(form, state.startImage)
      const result = await this.api.enhancePrompt(form)
      this.actions.patch({ enhancedPrompt: result.enhanced_prompt, enhancedSource: original, enhancedImageKey: this.imageKey(state.startImage) })
      return result.enhanced_prompt
    } catch (error) {
      this.actions.patch({ error: error.message })
      return ''
    } finally {
      this.actions.patch({ enhancing: false })
    }
  }

  reset() {
    const state = this.state()
    this.actions.clearConditioning()
    this.actions.clearAudio()
    const form = {
      prompt: '', width: state.config?.video?.default_width || 768, height: state.config?.video?.default_height || 512,
      fps: state.config?.video?.default_fps || 24, seed: -1, image_strength: 1
    }
    this.actions.patch({
      form,
      duration: durationFromFrames(state.config?.video?.default_frames || 121, form.fps),
      enhanceEnabled: state.config?.prompt_enhancement?.default_enabled ?? true,
      promptPreset: '', nextKeyframeID: 1, audioPickerOpen: false, advancedOpen: false,
      enhancedPrompt: '', enhancedSource: '', enhancedImageKey: ''
    })
  }

  async createPromptFromScenes(automatic = false) {
    const state = this.state()
    const hasScenes = Boolean(state.startImage || state.endImage || state.keyframes.some((item) => item.image))
    if (state.creatingPrompt || (!hasScenes && !state.audioClips.length)) return ''
    this.actions.patch({ creatingPrompt: true, promptCreationMessage: '', error: '' })
    try {
      const audioDetails = state.audioClips.map((clip, index) =>
        `\n음성 ${index + 1} (${Number(clip.start).toFixed(2)}초) 원문: ${clip.job.prompt || '(원문 없음)'}${clip.job.params?.instructions ? `\n음성 ${index + 1} 연기 지시: ${clip.job.params.instructions}` : ''}${clip.job.params?.speaker ? `\n음성 ${index + 1} 화자: ${clip.job.params.speaker}` : ''}`
      ).join('')
      const request = hasScenes
        ? `현재 선택된 시작·키프레임·마지막 장면을 시간 순서로 모두 보고, 장면 사이를 자연스럽게 연결할 LTX 영상 프롬프트를 만들어서 영상 설정에 적용해줘. 피사체 동작, 카메라 움직임, 환경의 움직임과 장면 연속성을 구체적으로 포함하고, 연결된 음성이 있으면 그 내용·감정·말하는 흐름과 동작을 자연스럽게 맞춰줘.${audioDetails}`
        : `연결된 생성 음성의 내용·감정·말하는 흐름에 어울리는 LTX 오디오 구동 영상 프롬프트를 만들어서 영상 설정에 적용해줘. 피사체 동작, 표정, 카메라 움직임과 환경의 움직임을 구체적으로 쓰되 음성에 없는 극단적인 분위기나 색상은 임의로 만들지 마.${audioDetails}`
      const visualContext = hasScenes ? await this.actions.visualContext(request) : null
      if (hasScenes && !visualContext) throw new Error('분석할 장면 이미지를 읽지 못했습니다.')
      const result = await this.api.assistantChat({ messages: [{ role: 'user', content: request }], state: state.assistantState, visual_context: visualContext })
      const prompt = result.actions?.find((action) => action.type === 'set_video' && action.prompt?.trim())?.prompt?.trim()
      if (!prompt) throw new Error('장면 분석 결과에서 영상 프롬프트를 얻지 못했습니다.')
      this.actions.patch({
        form: { ...this.state().form, prompt }, enhancedPrompt: '', enhancedSource: '', enhancedImageKey: '',
        promptCreationMessage: hasScenes && state.audioClips.length ? '장면 이미지와 음성 내용을 분석해 프롬프트에 적용했습니다.' : hasScenes ? '장면 이미지를 분석해 프롬프트에 적용했습니다.' : '음성 내용을 바탕으로 프롬프트를 적용했습니다.'
      })
      return prompt
    } catch (error) {
      this.actions.patch({ error: error.message || `${automatic ? '자동 ' : ''}영상 프롬프트를 만들지 못했습니다.` })
      return ''
    } finally {
      this.actions.patch({ creatingPrompt: false })
    }
  }

  async generate() {
    this.actions.patch({ busy: true, error: '' })
    try {
      let state = this.state()
      this.actions.patch({ form: { ...state.form, width: snapDimension(state.form.width, 64, 256, 1920), height: snapDimension(state.form.height, 64, 256, 1920) } })
      this.actions.normalizeTiming()
      state = this.state()
      if (!state.form.prompt.trim()) {
        if (!state.audioClips.length && !state.startImage && !state.endImage && !state.keyframes.some((item) => item.image)) throw new Error('프롬프트를 입력하거나 음성·장면 이미지를 선택하세요.')
        const prompt = await this.createPromptFromScenes(true)
        if (!prompt) return
        state = this.state()
      }
      let effectivePrompt = state.enhancedPrompt
      if (this.enhancementActive(state) && !this.enhancementCurrent(state)) {
        effectivePrompt = await this.enhance()
        if (!effectivePrompt) return
        state = this.state()
      }
      const form = buildVideoGenerationForm({
        videoForm: state.form,
        prompt: this.enhancementActive(state) ? effectivePrompt : state.form.prompt,
        originalPrompt: state.form.prompt,
        numFrames: framesForDuration(state.duration, state.form.fps),
        startImage: state.startImage, endImage: state.endImage, endStrength: state.endStrength,
        audioClips: state.audioClips, keyframes: state.keyframes
      })
      await this.api.video(form)
      this.actions.showNewest()
      this.actions.clearConditioning()
      this.actions.clearAudio()
      this.actions.patch({ form: { ...this.state().form, prompt: '' }, enhancedPrompt: '', enhancedSource: '', enhancedImageKey: '', mobilePane: 'results' })
      await this.actions.refresh()
    } catch (error) {
      this.actions.patch({ error: error.message })
    } finally {
      this.actions.patch({ busy: false })
    }
  }
}
