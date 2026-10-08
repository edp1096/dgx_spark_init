import test from 'node:test'
import assert from 'node:assert/strict'
import { buildAssistantState } from './assistantState.js'

test('assistant state exposes only the active media controls and visible image indexes', () => {
  const state = buildAssistantState({
    tab: 'video',
    busy: false,
    imageForm: { prompt: 'image' },
    imageEnhanceEnabled: true,
    activeKreaModuleLabels: ['Identity'],
    videoForm: { prompt: 'video', fps: 24 },
    videoDurationSeconds: 4,
    videoEnhanceEnabled: false,
    videoImage: {},
    videoEndImage: null,
    videoAudioJob: { id: 'audio-1' },
    videoAudioClips: [{ job: { id: 'audio-1', prompt: 'hello', params: { instructions: 'calm' } }, start: 1, duration: 2 }],
    videoKeyframes: [{ time: 2, strength: 0.8, image: {} }],
    speechForm: { text: 'speech' },
    recognitionForm: { source: 'url', url: 'https://example.test/video', language: 'ja', context: '', translation_mode: 'translated', target_language: 'Korean' },
    recognitionFile: null,
    recognitionSourceVideoJob: null,
    pagedImageJobs: [{ id: 'image-9', status: 'completed', prompt: 'x'.repeat(240) }],
    imagePage: 2,
    imagePageSize: 8,
  })

  assert.equal(state.video.has_start_image, true)
  assert.equal(state.video.has_audio, true)
  assert.deepEqual(state.video.audio_clips, [{ source_job_id: 'audio-1', start: 1, duration: 2 }])
  assert.equal(state.recognition.has_source, true)
  assert.equal(state.recent_images[0].index, 9)
  assert.equal(state.recent_images[0].prompt.length, 180)
})
