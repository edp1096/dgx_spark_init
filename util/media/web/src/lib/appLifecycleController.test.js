import test from 'node:test'
import assert from 'node:assert/strict'
import { AppLifecycleController, loadMediaPreferences, runtimeDefaults } from './appLifecycleController.js'

const storage = (values) => ({ getItem: (key) => values[key] ?? null })

test('media preferences validate view, page size and sort values', () => {
  const result = loadMediaPreferences(storage({
    'media-image-view': 'list',
    'media-video-view': 'bad',
    'media-image-page-size': '16',
    'media-image-sort-order': 'asc',
    'media-video-page-size': '999'
  }), () => [8, 12, 16], { pageSizes: { image: 8, video: 8 }, sortOrders: { image: 'desc', video: 'desc' } })
  assert.equal(result.views.image, 'list')
  assert.equal(result.views.video, 'gallery')
  assert.equal(result.pageSizes.image, 16)
  assert.equal(result.pageSizes.video, 8)
  assert.equal(result.sortOrders.image, 'asc')
})

test('runtime defaults apply configured forms and checkpoint safety', () => {
  const result = runtimeDefaults({
    image: { default_width: 768, default_height: 1024, default_mode: 'edit', default_checkpoint: 'moody-v7', default_prompt_enhancer: true },
    speech: { default_language: 'Korean', default_speaker: 'Sohee' },
    recognition: { default_language: 'Auto', default_output_formats: ['srt'], default_translation_mode: 'none', default_translation_language: 'Korean' },
    video: { default_width: 1280, default_height: 704, default_fps: 24, default_frames: 121 },
    prompt_enhancement: { default_enabled: false }
  }, { sampling_preset: 'default', filter_mode: 'balanced', filter_strength: 1 }, ['create', 'edit'])
  assert.equal(result.imageForm.mode, 'edit')
  assert.equal(result.videoDuration, 5)
  assert.equal(result.options.sampling_preset, 'moody')
  assert.equal(result.options.filter_mode, 'off')
})

test('lifecycle starts and clears every poller', async () => {
  const intervals = []
  const cleared = []
  const calls = []
  const controller = new AppLifecycleController({
    api: { config: async () => ({ ok: true }) }, storage: storage({}),
    timers: { setInterval: (fn, ms) => (intervals.push({ fn, ms }), intervals.length), clearInterval: (id) => cleared.push(id) },
    actions: {
      applyPreferences: () => calls.push('preferences'), pageSizeOptionsFor: () => [8],
      preferenceDefaults: () => ({ pageSizes: { image: 8 }, sortOrders: { image: 'desc' } }),
      applyConfig: () => calls.push('config'), setError: () => {}, refreshUserLoras: () => calls.push('loras'),
      refreshJobs: () => calls.push('jobs'), refreshSystemUsage: () => calls.push('system'),
      refreshVideoModels: () => calls.push('video'), refreshImageModels: () => calls.push('image'),
      shouldRefreshModels: () => false, setProgressClock: () => {}
    }
  })
  const stop = controller.start()
  await Promise.resolve()
  assert.deepEqual(intervals.map((item) => item.ms), [1500, 5000, 3000, 1000])
  assert.ok(calls.includes('config'))
  stop()
  assert.deepEqual(cleared, [1, 2, 3, 4])
})
