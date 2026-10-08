import assert from 'node:assert/strict'
import test from 'node:test'

import { runtimePhasePresentation } from './runtimePhasePresentation.js'

test('runtime phase presentation requires the current job operation', () => {
  const job = { id: 'job', params: { runtime_phase: {
    operation_id: 'job', phase: 'model_unloading', component: 'LTX DiT',
    detail: 'Transformer GPU 가중치 해제', progress: .78, memory_action: 'unload', resident_after: false
  } } }
  assert.deepEqual(runtimePhasePresentation(job), {
    phase: 'model_unloading', label: 'LTX DiT · 모델 해제',
    detail: 'Transformer GPU 가중치 해제 · 메모리 해제', progress: 78, residentAfter: false
  })
  job.params.runtime_phase.operation_id = 'other'
  assert.equal(runtimePhasePresentation(job), null)
})

test('runtime phase presentation distinguishes retained and loaded memory', () => {
  const job = { id: 'seed', params: { runtime_phase: {
    operation_id: 'seed', phase: 'cache_retaining', component: 'SeedVR2 DiT·VAE',
    detail: 'GPU 해제·CPU 캐시 유지', progress: .98, memory_action: 'retain', resident_after: true
  } } }
  const result = runtimePhasePresentation(job)
  assert.equal(result.label, 'SeedVR2 DiT·VAE · 캐시 유지')
  assert.match(result.detail, /캐시 유지/)
  assert.equal(result.residentAfter, true)
})
