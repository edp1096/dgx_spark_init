import test from 'node:test'
import assert from 'node:assert/strict'
import { get } from 'svelte/store'
import { SettingsController } from './settingsController.js'

test('settings controller refreshes status and clears failed usage safely', async () => {
  const controller = new SettingsController({
    api: {
      system: async () => { throw new Error('offline') },
      videoModels: async () => ({ ready: true }),
      imageCheckpoints: async () => ({ preparing: false })
    },
    setError: () => {},
    setMessage: () => {}
  })
  await Promise.all([controller.refreshSystemUsage(), controller.refreshVideoModels(), controller.refreshImageCheckpoints()])
  const state = get(controller.state)
  assert.equal(state.systemUsage.mem_used_gb, null)
  assert.equal(state.videoModelStatus.ready, true)
  assert.equal(state.imageCheckpointStatus.preparing, false)
})

test('model preparation owns busy state, messages and validation', async () => {
  const messages = []
  const errors = []
  const api = {
    prepareImageCheckpoints: async (_civitai, _hf, variants) => ({ started: variants.includes('official') }),
    imageCheckpoints: async () => ({ ready: true })
  }
  const controller = new SettingsController({ api, setError: (value) => errors.push(value), setMessage: (value) => messages.push(value) })
  const invalid = await controller.prepareImageCheckpoints({ civitaiToken: '', hfToken: '', variants: [] })
  assert.equal(invalid.clearTokens, false)
  assert.match(errors.at(-1), /하나 이상/)
  const result = await controller.prepareImageCheckpoints({ civitaiToken: ' c ', hfToken: ' h ', variants: ['official'] })
  assert.equal(result.clearTokens, true)
  assert.match(messages.at(-1), /시작/)
  assert.equal(get(controller.state).preparingImageCheckpoints, false)
})

test('settings save normalizes dimensions and frame count before persistence', async () => {
  let saved = null
  const busy = []
  const controller = new SettingsController({
    api: { saveConfig: async (value) => (saved = value, { config: value, restart_required: false }) },
    setError: () => {}, setMessage: () => {}, setBusy: (value) => busy.push(value)
  })
  const result = await controller.saveConfig({
    image: { default_width: 1001, default_height: 777 },
    video: { default_width: 1001, default_height: 777, default_fps: 24 }
  }, 5)
  assert.equal(saved.image.default_width % 8, 0)
  assert.equal(saved.video.default_width % 64, 0)
  assert.equal(saved.video.default_frames, 121)
  assert.equal(result.restart_required, false)
  assert.deepEqual(busy, [true, false])
})

test('temporary storage cleanup refreshes capacity and reports reclaimed bytes', async () => {
  const messages = []
  let storageCalls = 0
  const controller = new SettingsController({
    api: {
      storage: async () => ({ reclaimable_bytes: storageCalls++ ? 0 : 1024 }),
      cleanupTemporaryStorage: async () => ({ removed_directories: 2, removed_bytes: 1024 })
    },
    setError: () => {}, setMessage: (value) => messages.push(value)
  })
  await controller.loadStorage()
  const cleaned = await controller.cleanupStorage(() => true, (value) => `${value}B`)
  assert.equal(cleaned, true)
  assert.equal(get(controller.state).storage.reclaimable_bytes, 0)
  assert.match(messages.at(-1), /2개/)
})
