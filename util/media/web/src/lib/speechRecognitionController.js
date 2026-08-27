export class SpeechRecognitionController {
  constructor({ api, actions }) {
    this.api = api
    this.actions = actions
  }

  state() { return this.actions.getState() }

  resetOptions() {
    const state = this.state()
    this.actions.patch({ recognitionOptions: null, recognitionForm: { ...state.recognitionForm, media_part: '', media_source: '' } })
  }

  resetSpeech() {
    const state = this.state()
    this.actions.patch({ speechForm: {
      text: '', instructions: '', language: state.config?.speech?.default_language || 'Korean',
      speaker: state.config?.speech?.default_speaker || 'Sohee', seed: -1
    } })
  }

  resetRecognition() {
    const state = this.state()
    this.actions.clearRecognitionFileInput()
    this.actions.closeRecognitionPicker()
    this.actions.patch({
      recognitionFile: null, recognitionSourceVideoJob: null, recognitionOptions: null,
      recognitionForm: {
        source: 'url', url: '', language: state.config?.recognition?.default_language || 'Auto', context: '',
        output_formats: [...(state.config?.recognition?.default_output_formats || ['srt', 'txt'])],
        translation_mode: state.config?.recognition?.default_translation_mode || 'none',
        target_language: state.config?.recognition?.default_translation_language || 'Korean', media_part: '', media_source: ''
      }
    })
  }

  async generateSpeech() {
    const state = this.state()
    this.actions.patch({ busy: true, error: '' })
    try {
      const form = new FormData()
      form.append('text', state.speechForm.text)
      form.append('instructions', state.speechForm.instructions)
      form.append('language', state.speechForm.language)
      form.append('speaker', state.speechForm.speaker)
      form.append('seed', state.speechForm.seed)
      await this.api.speech(form)
      this.actions.patch({ speechForm: { ...this.state().speechForm, text: '' }, mobileSpeechPane: 'results' })
      this.actions.showNewest('speech')
      await this.actions.refresh()
    } catch (error) {
      this.actions.patch({ error: error.message })
    } finally {
      this.actions.patch({ busy: false })
    }
  }

  async recognize() {
    const state = this.state()
    const source = state.recognitionForm.source
    if ((source === 'file' && !state.recognitionFile) || (source === 'url' && !state.recognitionForm.url.trim()) || (source === 'video_job' && !state.recognitionSourceVideoJob)) return
    this.actions.patch({ busy: true, error: '' })
    try {
      const form = new FormData()
      if (source === 'file') form.append('media', state.recognitionFile)
      else if (source === 'video_job') form.append('reuse_video_job', state.recognitionSourceVideoJob.id)
      else form.append('url', state.recognitionForm.url.trim())
      if (source === 'url' && state.recognitionForm.media_part) form.append('media_part', state.recognitionForm.media_part)
      if (source === 'url' && state.recognitionForm.media_source) form.append('media_source', state.recognitionForm.media_source)
      form.append('language', state.recognitionForm.language)
      form.append('context', state.recognitionForm.context)
      form.append('output_formats', state.recognitionForm.output_formats.join(','))
      form.append('translation_mode', state.recognitionForm.translation_mode)
      form.append('target_language', state.recognitionForm.target_language)
      await this.api.recognition(form)
      this.actions.showNewest('recognition')
      this.actions.clearRecognitionFileInput()
      this.actions.patch({
        recognitionFile: null, recognitionSourceVideoJob: null, recognitionOptions: null,
        recognitionForm: { ...this.state().recognitionForm, url: '', media_part: '', media_source: '' },
        mobileRecognitionPane: 'results'
      })
      await this.actions.refresh()
    } catch (error) {
      this.actions.patch({ error: error.message })
    } finally {
      this.actions.patch({ busy: false })
    }
  }

  updateURL(value) {
    const state = this.state()
    const form = { ...state.recognitionForm, url: value, media_part: '', media_source: '' }
    const patch = { recognitionForm: form, recognitionOptions: null }
    if (value.trim()) {
      patch.recognitionForm.source = 'url'
      patch.recognitionFile = null
      patch.recognitionSourceVideoJob = null
      this.actions.clearRecognitionFileInput()
    }
    this.actions.patch(patch)
  }

  updateFile(file) {
    if (!file) return
    const state = this.state()
    this.actions.patch({
      recognitionFile: file, recognitionSourceVideoJob: null, recognitionOptions: null,
      recognitionForm: { ...state.recognitionForm, source: 'file', url: '', media_part: '', media_source: '' }
    })
  }

  clearFile() {
    const state = this.state()
    this.actions.clearRecognitionFileInput()
    this.actions.patch({ recognitionFile: null, recognitionForm: { ...state.recognitionForm, source: 'url' } })
  }

  selectedPart() {
    const state = this.state()
    return state.recognitionOptions?.parts?.find((part) => part.id === state.recognitionForm.media_part) || state.recognitionOptions?.parts?.[0]
  }

  selectPart(partID) {
    const state = this.state()
    this.actions.patch({ recognitionForm: { ...state.recognitionForm, media_part: partID, media_source: '' } })
  }

  async loadOptions() {
    const state = this.state()
    const url = state.recognitionForm.url.trim()
    if (!url) return
    this.actions.patch({ loadingRecognitionOptions: true, error: '' })
    try {
      const options = await this.api.mediaOptions(url)
      if (url !== this.state().recognitionForm.url.trim()) return
      this.actions.patch({ recognitionOptions: options, recognitionForm: { ...this.state().recognitionForm, media_part: options.parts?.[0]?.id || '', media_source: '' } })
    } catch (error) {
      this.actions.patch({ error: error.message, recognitionOptions: null, recognitionForm: { ...this.state().recognitionForm, media_part: '', media_source: '' } })
    } finally {
      this.actions.patch({ loadingRecognitionOptions: false })
    }
  }
}
