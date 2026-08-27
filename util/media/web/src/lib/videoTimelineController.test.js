import test from 'node:test'
import assert from 'node:assert/strict'
import { VideoTimelineController } from './videoTimelineController.js'

function fixture() {
  const state = { duration: 5, fps: 24, start: null, end: null, endStrength: 1, keyframes: [], audio: [], message: 'old', next: 1 }
  const revoked = []
  const controller = new VideoTimelineController({
    getDuration: () => state.duration, setDuration: (value) => state.duration = value,
    getFPS: () => state.fps,
    getStartImage: () => state.start, setStartImage: (value) => state.start = value,
    getEndImage: () => state.end, setEndImage: (value) => state.end = value,
    setEndStrength: (value) => state.endStrength = value,
    getKeyframes: () => state.keyframes, setKeyframes: (value) => state.keyframes = value,
    getAudioClips: () => state.audio, setAudioClips: (value) => state.audio = value,
    setPromptMessage: (value) => state.message = value,
    resetEnhancement: () => {}, allocateKeyframeID: () => state.next++, sendAudioToVideo: () => {}
  }, { createObjectURL: () => 'blob:preview', revokeObjectURL: (value) => revoked.push(value) })
  return { controller, state, revoked }
}

test('video timeline owns image and keyframe preview lifetime', () => {
  const { controller, state, revoked } = fixture()
  controller.setConditionImage('start', { name: 'one.png' })
  controller.setConditionImage('start', { name: 'two.png' })
  const keyframe = controller.addKeyframe()
  controller.setConditionImage(`keyframe:${keyframe.id}`, { name: 'key.png' })
  controller.removeKeyframe(keyframe.id)
  assert.equal(state.start.name, 'two.png')
  assert.deepEqual(revoked, ['blob:preview', 'blob:preview'])
  controller.clearConditioning()
  assert.equal(state.start, null)
  assert.equal(state.endStrength, 1)
})

test('video audio movement prevents overlap and stays inside duration', () => {
  const { controller, state } = fixture()
  state.audio = [
    { id: 1, start: 0, duration: 2, job: { id: 'one' } },
    { id: 2, start: 3, duration: 2, job: { id: 'two' } }
  ]
  controller.moveAudio(2, 1)
  assert.equal(state.audio[1].start, 2)
  controller.moveAudio(2, 99)
  assert.equal(state.audio[1].start, 3)
})

test('video timing normalizes keyframes through the shared 8n+1 rule', () => {
  const { controller, state } = fixture()
  state.keyframes = [{ id: 1, image: null, time: 0.875, strength: 1 }]
  controller.normalizeTiming()
  assert.equal((Math.round(state.duration * state.fps) + 1) % 8, 1)
  assert.equal(state.keyframes[0].time * state.fps, Math.round(state.keyframes[0].time * state.fps))
})
