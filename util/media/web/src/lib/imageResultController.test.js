import test from 'node:test'
import assert from 'node:assert/strict'
import { get } from 'svelte/store'
import { ImageResultController } from './imageResultController.js'

function fixture() {
  const calls = []
  const controller = new ImageResultController({
    api: {
      upscaleImage: async (...args) => calls.push(['upscale', ...args]),
      detailEnhanceImage: async (...args) => calls.push(['detail', ...args]),
      garmentExtract: async (...args) => calls.push(['garment', ...args])
    },
    actions: {
      setError: (value) => calls.push(['error', value]), showNewest: () => calls.push(['newest']),
      refresh: async () => calls.push(['refresh']), showResults: () => calls.push(['results']),
      clearParentJob: () => calls.push(['parent']), clonePrompt: () => calls.push(['prompt']),
      cloneSettings: () => calls.push(['settings']), cloneReferences: async () => 2,
      showCreate: () => calls.push(['create']), scrollTop: () => calls.push(['scroll'])
    }
  })
  return { controller, calls }
}

test('image result post-processing owns busy state and refresh', async () => {
  const { controller, calls } = fixture()
  await controller.upscale({ id: 'job', output_url: '/job.png' })
  await controller.detailEnhance({ id: 'job', output_url: '/job.png' })
  assert.equal(get(controller.state).upscalingJob, '')
  assert.equal(get(controller.state).detailEnhancingJob, '')
  assert.ok(calls.some(([name]) => name === 'upscale'))
  assert.ok(calls.some(([name]) => name === 'detail'))
  controller.destroy()
})

test('image clone coordinates selected parts without changing the old result', async () => {
  const { controller, calls } = fixture()
  await controller.clone({ id: 'job' }, 'all')
  const state = get(controller.state)
  assert.match(state.cloneMessage, /이미지 2장/)
  assert.equal(state.cloningJob, '')
  assert.ok(calls.some(([name]) => name === 'parent'))
  assert.ok(calls.some(([name]) => name === 'prompt'))
  assert.ok(calls.some(([name]) => name === 'settings'))
  controller.destroy()
})
