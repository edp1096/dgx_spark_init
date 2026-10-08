import assert from 'node:assert/strict'
import test from 'node:test'

import {
  generationQueuePosition,
  imageGenerationEstimateSeconds,
  imageGenerationProgress,
  modelPreparationProgress,
  recognitionProgressPercent,
  recognitionProgressText,
  recognitionQueuePosition,
  speechGenerationProgress,
  videoGenerationProgress
} from './jobProgress.js'

const baseTime = Date.parse('2026-08-27T12:00:00Z')

test('generation and recognition queues use durable queue order', () => {
  const jobs = [
    { id: 'image-2', kind: 'image', status: 'queued', created_at: '2026-08-27T11:00:02Z', params: { queued_at: '2026-08-27T11:00:02Z' } },
    { id: 'image-1', kind: 'image', status: 'queued', created_at: '2026-08-27T11:00:01Z', params: { queued_at: '2026-08-27T11:00:01Z' } },
    { id: 'recognition-1', kind: 'recognition', status: 'queued', created_at: '2026-08-27T11:00:03Z', params: {} }
  ]
  assert.equal(generationQueuePosition(jobs[0], jobs), 2)
  assert.equal(generationQueuePosition(jobs[1], jobs), 1)
  assert.equal(recognitionQueuePosition(jobs[2], jobs), 1)
  assert.match(recognitionProgressText(jobs[2], jobs), /대기 1번째/)
})

test('image progress learns from matching completed jobs', () => {
  const running = { id: 'running', kind: 'image', status: 'running', created_at: '2026-08-27T11:59:50Z', params: { started_at: '2026-08-27T11:59:50Z', mode: 'create', checkpoint: 'official', width: 1024, height: 1024, steps: 8 } }
  const completed = { id: 'completed', kind: 'image', status: 'completed', created_at: '2026-08-27T11:58:00Z', updated_at: '2026-08-27T11:58:40Z', params: { started_at: '2026-08-27T11:58:00Z', mode: 'create', checkpoint: 'official', width: 1024, height: 1024, steps: 8 } }
  const jobs = [running, completed]
  assert.equal(imageGenerationEstimateSeconds(running, jobs), 40)
  const progress = imageGenerationProgress(running, jobs, baseTime)
  assert.equal(progress.elapsed, '10/40초')
  assert.equal(Math.round(progress.percent), 25)
})

test('video, speech and recognition progress produce bounded UI values', () => {
  const video = { id: 'video', kind: 'video', status: 'running', created_at: '2026-08-27T11:59:50Z', params: { started_at: '2026-08-27T11:59:50Z', width: 768, height: 512, num_frames: 121, fps: 24 } }
  const speech = { id: 'speech', kind: 'speech', status: 'running', prompt: '안녕하세요. 테스트 음성입니다.', created_at: '2026-08-27T11:59:55Z', params: { started_at: '2026-08-27T11:59:55Z', language: 'Korean', speaker: 'Sohee' } }
  assert.ok(videoGenerationProgress(video, [video], baseTime).percent >= 5)
  assert.ok(speechGenerationProgress(speech, [speech], baseTime).percent >= 5)
  assert.equal(recognitionProgressPercent({ status: 'running', params: { stage: 'recognition', progress: 2, segments: 4 } }), 50)
})

test('model loading is a separate timed preparation stage', () => {
  const preparing = {
    id: 'preparing', kind: 'image', status: 'running', created_at: '2026-08-27T11:59:40Z',
    params: {
      stage: 'model-preparing', model_prepare_started_at: '2026-08-27T11:59:50Z',
      model_prepare_profile: 'krea-create', model_prepare_label: 'Krea 생성 모델 탑재',
      model_prepare_estimate_seconds: 40
    }
  }
  const progress = modelPreparationProgress(preparing, [preparing], baseTime)
  assert.match(progress.label, /Krea 생성 모델 탑재/)
  assert.equal(progress.elapsed, '10/40초')
  assert.equal(Math.round(progress.percent), 25)
})
