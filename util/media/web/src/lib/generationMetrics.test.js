import assert from 'node:assert/strict'
import test from 'node:test'

import {
  imageGenerationDistance,
  imageGenerationKey,
  imageGenerationWork,
  orderedJobs,
  percentile,
  speechGenerationKey,
  videoGenerationPipeline,
  videoGenerationWork
} from './generationMetrics.js'

test('image metrics distinguish model and module pipelines', () => {
  const base = { params: { mode: 'create', checkpoint: 'official', width: 1024, height: 1024, steps: 8 } }
  const identity = { params: { ...base.params, identity: true, references: 2 } }
  assert.notEqual(imageGenerationKey(base), imageGenerationKey(identity))
  assert.ok(imageGenerationWork(identity) > imageGenerationWork(base))
  assert.ok(imageGenerationDistance(base, identity) > 0)
  const major = { params: { ...base.params, sequence_strategy: 'major', sequence_previous_job_id: 'first' } }
  assert.ok(imageGenerationWork(major) > imageGenerationWork(base) * 2)
  assert.notEqual(imageGenerationKey(major), imageGenerationKey(base))
})

test('video, speech, percentile and ordering helpers remain deterministic', () => {
  assert.equal(videoGenerationPipeline({ params: { mode: 'a2v' } }), 'a2v')
  assert.ok(videoGenerationWork({ params: { width: 768, height: 512, num_frames: 121 } }) > 0)
  assert.match(speechGenerationKey({ prompt: '안녕하세요', params: { language: 'Korean', speaker: 'Sohee' } }), /Korean/)
  assert.equal(percentile([1, 2, 3, 4], 0.5), 2)
  const jobs = [{ id: 'new' }, { id: 'old' }]
  assert.deepEqual(orderedJobs(jobs, 'asc').map((job) => job.id), ['old', 'new'])
  assert.deepEqual(jobs.map((job) => job.id), ['new', 'old'])
})
