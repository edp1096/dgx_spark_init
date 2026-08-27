import assert from 'node:assert/strict'
import test from 'node:test'

import { durationFromFrames, formatDuration, framesForDuration, snapDimension } from './videoTiming.js'

test('LTX durations round-trip through 8n+1 frame counts', () => {
  for (const seconds of [1, 2.24, 5, 10]) {
    const frames = framesForDuration(seconds, 24)
    assert.equal((frames - 1) % 8, 0)
    assert.ok(frames >= 9)
    assert.equal(durationFromFrames(frames, 24), Math.round(((frames - 1) / 24) * 1000) / 1000)
  }
})

test('dimensions and display durations retain their existing UI contract', () => {
  assert.equal(snapDimension(1001, 32, 256, 2048), 992)
  assert.equal(snapDimension(10, 32, 256, 2048), 256)
  assert.equal(formatDuration(65.5), '1:05.5')
  assert.equal(formatDuration(3661), '1:01:01')
})
