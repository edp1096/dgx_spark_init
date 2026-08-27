import test from 'node:test'
import assert from 'node:assert/strict'
import { nearestAvailableVideoKeyframeFrame, normalizeVideoTiming, videoConditioningDisabledReason, videoStage2TokenCount } from './videoWorkflow.js'

test('video timing snaps duplicate keyframes and clamps audio', () => {
  const result = normalizeVideoTiming({ seconds: 5, fps: 24, keyframes: [{ id: 1, time: 1 }, { id: 2, time: 1 }], audioClips: [{ id: 1, start: 9, duration: 2 }] })
  assert.notEqual(result.keyframes[0].time, result.keyframes[1].time)
  assert.equal(result.audioClips[0].start, 3)
})

test('video keyframe selection and validation use frame positions', () => {
  const frame = nearestAvailableVideoKeyframeFrame({ rawFrame: 24, seconds: 5, fps: 24, keyframes: [{ id: 1, time: 1 }] })
  assert.equal(frame, 25)
  assert.match(videoConditioningDisabledReason({ audioSelected: false, a2vReady: true, seconds: 5, fps: 24, keyframes: [{ image: {}, time: 0 }] }), /시작과 마지막 사이/)
})

test('stage two token count remains stable for known dimensions', () => {
  assert.equal(videoStage2TokenCount(768, 512, 5, 24), 6144)
})
