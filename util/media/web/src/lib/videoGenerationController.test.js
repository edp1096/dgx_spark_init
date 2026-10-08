import test from 'node:test'
import assert from 'node:assert/strict'
import { VideoGenerationController } from './videoGenerationController.js'

test('video creation reset keeps configured LTX defaults and clears conditioning', () => {
  let state = {
    config: { video: { default_width: 1024, default_height: 576, default_fps: 24, default_frames: 121 }, prompt_enhancement: { default_enabled: false } },
    form: { prompt: 'old', width: 768, height: 512, fps: 24 }, duration: 3, enhancedPrompt: 'old', enhancedSource: 'old',
    startImage: null, endImage: null, keyframes: [], audioClips: []
  }
  let conditioningCleared = 0, audioCleared = 0
  const controller = new VideoGenerationController({ api: {}, actions: {
    getState: () => state, patch: (patch) => state = { ...state, ...patch },
    clearConditioning: () => conditioningCleared++, clearAudio: () => audioCleared++
  } })
  controller.reset()
  assert.equal(state.form.width, 1024)
  assert.equal(state.form.prompt, '')
  assert.equal(state.enhanceEnabled, false)
  assert.equal(conditioningCleared, 1)
  assert.equal(audioCleared, 1)
})
