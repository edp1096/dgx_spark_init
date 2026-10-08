import test from 'node:test'
import assert from 'node:assert/strict'
import { upscaleFrameWork, videoUpscaleEstimate, videoUpscaleEstimateSeconds } from './videoUpscaleEta.js'

const cleanGB10Sample = {
  source_width: 768,
  source_height: 512,
  width: 1632,
  height: 1088,
  duration: 121 / 24,
  fps: 24,
  batch_size: 5,
  temporal_overlap: 1,
  upscale_scale: 2.125
}

test('calculates temporal overlap work procedurally', () => {
  assert.deepEqual(upscaleFrameWork(cleanGB10Sample), {
    sourceFrames: 121,
    processedFrames: 150,
    batches: 30,
    batch: 5,
    overlap: 1
  })
})

test('models every SeedVR2 phase near the clean GB10 measurement', () => {
  const estimate = videoUpscaleEstimate(cleanGB10Sample)
  assert.equal(estimate.spatialTiles, 4)
  assert.ok(estimate.encodeSeconds > 210 && estimate.encodeSeconds < 225)
  assert.ok(estimate.ditSeconds > 305 && estimate.ditSeconds < 320)
  assert.ok(estimate.decodeSeconds > 490 && estimate.decodeSeconds < 510)
  assert.ok(estimate.totalSeconds > 1060 && estimate.totalSeconds < 1100)
})

test('retry and historical timing metadata cannot contaminate the estimate', () => {
  const contaminated = {
    ...cleanGB10Sample,
    retry_count: 7,
    started_at: '2026-08-26T23:29:26+09:00',
    queued_at: '2026-08-26T23:00:00+09:00',
    previous_duration_seconds: 99999
  }
  assert.equal(videoUpscaleEstimateSeconds(contaminated), videoUpscaleEstimateSeconds(cleanGB10Sample))
})
