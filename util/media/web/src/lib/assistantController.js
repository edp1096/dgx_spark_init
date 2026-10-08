import { applyAssistantActionList } from './assistantActions.js'

export class AssistantController {
  constructor(actions) {
    this.actions = actions
  }

  switchTab(nextTab, results = false) {
    if (nextTab === 'settings') this.actions.openSettings()
    else this.actions.setTab(nextTab)
    this.actions.setMobilePane(nextTab, results ? 'results' : 'create')
  }

  applyActions(actions = []) {
    return applyAssistantActionList(actions, {
      ...this.actions.actionContext(),
      switchTab: (tab, results = false) => this.switchTab(tab, results)
    })
  }

  async execute(kind) {
    this.actions.setError('')
    if (kind === 'image') {
      this.switchTab('image')
      const reason = this.actions.imageDisabledReason()
      if (reason) throw new Error(reason)
      if (this.actions.imageEnhancementActive() && !this.actions.imageEnhancementCurrent()) await this.actions.enhanceImagePrompt()
      if (this.actions.imageEnhancementActive() && !this.actions.imageEnhancementCurrent()) throw new Error(this.actions.getError() || '프롬프트를 향상하지 못했습니다.')
      await this.actions.generateImage()
      if (this.actions.getError()) throw new Error(this.actions.getError())
      return '이미지 작업을 요청했습니다.'
    }
    if (kind === 'video') {
      this.switchTab('video')
      const form = this.actions.getVideoForm()
      const hasCondition = this.actions.getVideoAudioJob()
        || this.actions.getVideoImage()
        || this.actions.getVideoEndImage()
        || this.actions.getVideoKeyframes().some((item) => item.image)
      if (!form.prompt.trim()) {
        if (!hasCondition) throw new Error('프롬프트를 입력하거나 음성·장면 이미지를 선택하세요.')
        if (!await this.actions.createVideoPromptFromScenes(true)) throw new Error(this.actions.getError() || '영상 프롬프트를 자동 작성하지 못했습니다.')
      }
      if (this.actions.videoEnhancementActive() && !this.actions.videoEnhancementCurrent()) await this.actions.enhanceVideoPrompt()
      if (this.actions.videoEnhancementActive() && !this.actions.videoEnhancementCurrent()) throw new Error(this.actions.getError() || '영상 프롬프트를 향상하지 못했습니다.')
      await this.actions.generateVideo()
      if (this.actions.getError()) throw new Error(this.actions.getError())
      return '영상 작업을 요청했습니다.'
    }
    if (kind === 'speech') {
      this.switchTab('speech')
      if (!this.actions.getSpeechForm().text.trim()) throw new Error('읽을 문장을 입력하세요.')
      await this.actions.generateSpeech()
      if (this.actions.getError()) throw new Error(this.actions.getError())
      return '음성 작업을 요청했습니다.'
    }
    if (kind === 'recognition') {
      this.switchTab('recognition')
      const form = this.actions.getRecognitionForm()
      if (!this.actions.getRecognitionFile() && !this.actions.getRecognitionSourceVideoJob() && !form.url.trim()) {
        throw new Error('먼저 받아쓸 영상, 파일 또는 URL을 선택하세요.')
      }
      await this.actions.recognizeSpeech()
      if (this.actions.getError()) throw new Error(this.actions.getError())
      return '자막 작업을 요청했습니다.'
    }
    throw new Error('지원하지 않는 작업입니다.')
  }
}
