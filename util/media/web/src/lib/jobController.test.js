import test from 'node:test'
import assert from 'node:assert/strict'
import { get } from 'svelte/store'
import { JobController } from './jobController.js'

test('job controller sorts jobs and maps engine health', async () => {
  let error = 'failed to fetch'
  const controller = new JobController({
    api: {
      jobs: async () => [{ id: 'older', created_at: '2026-01-01T00:00:00Z' }, { id: 'newer', created_at: '2026-01-02T00:00:00Z' }],
      engines: async () => [{ kind: 'image', status: 'online' }]
    },
    getError: () => error,
    setError: (value) => error = value
  })
  assert.equal(await controller.refresh(), true)
  assert.deepEqual(get(controller.state).jobs.map((job) => job.id), ['newer', 'older'])
  assert.equal(get(controller.state).engineStates.image, 'online')
  assert.equal(error, '')
})

test('job controller exposes action state and refreshes after retry', async () => {
  let retryCount = 0
  const controller = new JobController({
    api: {
      jobs: async () => [], engines: async () => [],
      retryJob: async (id) => { assert.equal(id, 'job-1'); retryCount += 1 }
    },
    getError: () => '', setError: () => {}
  })
  assert.equal(await controller.retryJob({ id: 'job-1' }), true)
  assert.equal(retryCount, 1)
  assert.equal(get(controller.state).retryingJob, '')
})
