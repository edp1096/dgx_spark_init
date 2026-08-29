import test from 'node:test'
import assert from 'node:assert/strict'
import { ImageCreationController } from './imageCreationController.js'

function fixture() {
  let state = {
    config: { image: { default_checkpoint: 'official' }, prompt_enhancement: { default_enabled: true } }, busy: false,
    form: { prompt: '', width: 1024, height: 1024, seed: -1, mode: 'create' }, enhanceEnabled: true, enhancedPrompt: '', enhancedSource: '',
    resolutionMode: 'smart', aspectRatio: '16:9', megapixels: 1, modules: { identity: true, depth: false, anypaint: false }, options: {},
    identityPreset: '', identityPreserveItems: ['identity', 'face', 'pose'], identityPreserveCustom: '',
    identityImage: {}, identityReferences: [], visionImages: [], styleReferenceImages: [], styleSelections: [], userLoraSelections: [], references: []
  }
  const calls = []
  const controller = new ImageCreationController({
    api: {},
    catalogs: {
      identityPreserveCatalog: ['identity', 'face', 'pose'].map((id) => ({ id })), defaultIdentityPreserveItems: ['identity', 'face', 'pose'],
      identityPresetUI: { '': { showSecondary: false }, restage: { showSecondary: false } }, imageAspectRatios: [['16:9', 16 / 9]], imageModeChoices: ['create']
    },
    actions: {
      getState: () => state, patch: (patch) => state = { ...state, ...patch }, setKreaImage: (...args) => calls.push(args),
      clearAllInputs() {}, resetSequence() {}, closeSequence() {}, clearCloneMessage() {}, scrollTop() {},
      filterModeDefault: () => 1
    }
  })
  return { controller, state: () => state, calls }
}

test('image creation controller applies identity intent and smart resolution', () => {
  const { controller, state, calls } = fixture()
  controller.applyIdentityPreset('restage')
  assert.match(state().form.prompt, /same person/)
  assert.deepEqual(calls[0], ['identityReference', null])
  controller.applySmartResolution()
  assert.equal(state().form.width % 8, 0)
  assert.equal(state().form.height % 8, 0)
  assert.ok(state().form.width > state().form.height)
})

test('head swap preset applies the published BFS Krea defaults', () => {
  const { controller, state } = fixture()
  controller.applyIdentityPreset('headSwap')
  assert.equal(state().options.identity_strength, 1)
  assert.equal(state().options.ref_boost, 1)
  assert.equal(state().options.source_ref_boost, 1)
  assert.equal(state().options.grounding_px, 512)
  assert.equal(state().options.steps, 8)
  assert.equal(state().options.identity_encoder, 'default')
  assert.equal(state().options.filter_mode, 'off')
})

test('image creation reset restores one coherent default state', () => {
  const { controller, state } = fixture()
  controller.reset()
  assert.equal(state().form.prompt, '')
  assert.equal(state().options.checkpoint, 'official')
  assert.equal(state().modules.identity, false)
  assert.equal(state().enhanceEnabled, true)
})
