import test from 'node:test'
import assert from 'node:assert/strict'
import { SpeechRecognitionController } from './speechRecognitionController.js'

test('recognition source changes clear mutually exclusive inputs and stale options', () => {
  let state = {
    recognitionForm: { source: 'file', url: '', media_part: 'old', media_source: 'old' }, recognitionFile: { name: 'old.mp4' },
    recognitionSourceVideoJob: { id: 'video' }, recognitionOptions: { parts: [] }
  }
  let cleared = 0
  const controller = new SpeechRecognitionController({ api: {}, actions: {
    getState: () => state, patch: (patch) => state = { ...state, ...patch }, clearRecognitionFileInput: () => cleared++
  } })
  controller.updateURL('https://example.com/video')
  assert.equal(state.recognitionForm.source, 'url')
  assert.equal(state.recognitionFile, null)
  assert.equal(state.recognitionSourceVideoJob, null)
  assert.equal(state.recognitionOptions, null)
  assert.equal(cleared, 1)
})
