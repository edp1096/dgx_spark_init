import { writable } from 'svelte/store'

const initialState = {
  upscalingJob: '',
  detailEnhancingJob: '',
  cloningJob: '',
  cloneMessage: ''
}

export class ImageResultController {
  constructor({ api, actions }) {
    this.api = api
    this.actions = actions
    this.current = { ...initialState }
    this.state = writable(this.current)
    this.unsubscribe = this.state.subscribe((value) => this.current = value)
  }

  setState(patch) { this.state.update((value) => ({ ...value, ...patch })) }

  async upscale(job) {
    if (!job.output_url || this.current.upscalingJob) return false
    this.setState({ upscalingJob: job.id })
    this.actions.setError('')
    try {
      await this.api.upscaleImage(job.id, { scale: 2, seed: -1 })
      this.actions.showNewest()
      await this.actions.refresh()
      return true
    } catch (cause) {
      this.actions.setError(cause.message)
      return false
    } finally {
      this.setState({ upscalingJob: '' })
    }
  }

  async detailEnhance(job) {
    if (!job.output_url || this.current.detailEnhancingJob) return false
    this.setState({ detailEnhancingJob: job.id })
    this.actions.setError('')
    try {
      await this.api.detailEnhanceImage(job.id, { strength: 1, seed: -1, vae: 'wan' })
      this.actions.showNewest()
      await this.actions.refresh()
      return true
    } catch (cause) {
      this.actions.setError(cause.message)
      return false
    } finally {
      this.setState({ detailEnhancingJob: '' })
    }
  }

  async submitGarment(form) {
    this.actions.setError('')
    await this.api.garmentExtract(form)
    this.actions.showResults()
    this.actions.showNewest()
    await this.actions.refresh()
  }

  async clone(job, part) {
    this.setState({ cloningJob: `${job.id}:${part}`, cloneMessage: '' })
    this.actions.setError('')
    try {
      if (part === 'all') this.actions.clearParentJob()
      if (part === 'prompt' || part === 'all') this.actions.clonePrompt(job)
      if (part === 'settings' || part === 'all') this.actions.cloneSettings(job)
      let inputCount = null
      if (part === 'references' || part === 'all') inputCount = await this.actions.cloneReferences(job)
      const labels = { prompt: '프롬프트', references: '참조 이미지', settings: '설정', all: '전체 작업' }
      const cloneMessage = inputCount === 0 && part === 'references'
        ? '이 작업에는 불러올 참조 이미지가 없습니다.'
        : `${labels[part]}을 새 작업 작성란으로 불러왔습니다${inputCount ? ` · 이미지 ${inputCount}장` : ''}. 기존 결과는 변경되지 않습니다.`
      this.setState({ cloneMessage })
      this.actions.showCreate()
      this.actions.scrollTop()
      return true
    } catch (cause) {
      this.actions.setError(cause.message)
      return false
    } finally {
      this.setState({ cloningJob: '' })
    }
  }

  destroy() { this.unsubscribe?.() }
}
