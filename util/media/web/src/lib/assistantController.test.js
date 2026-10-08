import test from 'node:test'
import assert from 'node:assert/strict'
import { AssistantController } from './assistantController.js'

function harness() {
  const state = { tab: '', panes: {}, error: '', speech: { text: '' }, video: { prompt: '' }, recognition: { url: '' }, calls: [] }
  const actions = {
    openSettings: () => state.tab = 'settings',
    setTab: (tab) => state.tab = tab,
    setMobilePane: (tab, pane) => state.panes[tab] = pane,
    actionContext: () => ({ getImageForm: () => ({ prompt: '' }), setImageForm: () => {}, resetImageEnhancement: () => {} }),
    setError: (value) => state.error = value,
    getError: () => state.error,
    imageDisabledReason: () => '',
    imageEnhancementActive: () => false,
    imageEnhancementCurrent: () => false,
    enhanceImagePrompt: async () => {},
    generateImage: async () => state.calls.push('image'),
    getVideoForm: () => state.video,
    getVideoAudioJob: () => null,
    getVideoImage: () => null,
    getVideoEndImage: () => null,
    getVideoKeyframes: () => [],
    createVideoPromptFromScenes: async () => '',
    videoEnhancementActive: () => false,
    videoEnhancementCurrent: () => false,
    enhanceVideoPrompt: async () => {},
    generateVideo: async () => state.calls.push('video'),
    getSpeechForm: () => state.speech,
    generateSpeech: async () => state.calls.push('speech'),
    getRecognitionForm: () => state.recognition,
    getRecognitionFile: () => null,
    getRecognitionSourceVideoJob: () => null,
    recognizeSpeech: async () => state.calls.push('recognition')
  }
  return { state, actions }
}

test('assistant controller owns tab and mobile pane navigation', () => {
  const { state, actions } = harness()
  const controller = new AssistantController(actions)
  controller.switchTab('video', true)
  assert.equal(state.tab, 'video')
  assert.equal(state.panes.video, 'results')
  controller.switchTab('settings')
  assert.equal(state.tab, 'settings')
})

test('assistant execution validates media inputs before dispatch', async () => {
  const { state, actions } = harness()
  const controller = new AssistantController(actions)
  await assert.rejects(() => controller.execute('video'), /프롬프트를 입력/)
  state.speech.text = '안녕하세요'
  assert.equal(await controller.execute('speech'), '음성 작업을 요청했습니다.')
  assert.deepEqual(state.calls, ['speech'])
})
