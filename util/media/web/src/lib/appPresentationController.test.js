import test from 'node:test'
import assert from 'node:assert/strict'
import { AppPresentationController } from './appPresentationController.js'

test('app presentation derives aggregate engine and active job state', () => {
  const state = { jobs: [{ status: 'queued' }, { status: 'completed' }], engineStates: { image: 'online', video: 'offline' } }
  const controller = new AppPresentationController({ engineCatalog: [['image', 'Image'], ['video', 'Video']], actions: { getState: () => state } })
  assert.equal(controller.activeJobs().length, 1)
  assert.equal(controller.enginePresentation().aggregate, 'degraded')
  state.engineStates.video = 'online'
  assert.equal(controller.enginePresentation().aggregate, 'healthy')
})
