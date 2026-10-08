import test from 'node:test'
import assert from 'node:assert/strict'
import { ImageConfigurationController } from './imageConfigurationController.js'

test('image configuration keeps checkpoint and style selection coherent', () => {
  let state = {
    modules: { identity: false }, options: { checkpoint: 'official-int8', identity_model: 'convrot', sampling_preset: 'default' },
    styleSelections: [], userLoraSelections: [], userLoraCatalog: [], settings: { image: { visible_checkpoints: ['official-int8', 'official', 'moody-v7'], default_checkpoint: 'moody-v7' } }, checkpointStatus: {}
  }
  const controller = new ImageConfigurationController({ api: {}, catalogs: { kreaStyleCatalog: [{ name: 'retroanime', label: 'Retro' }], checkpointDisplayChoices: [['moody-v7', 'Moody']] }, actions: {
    getState: () => state, patch: (patch) => state = { ...state, ...patch }
  } })
  controller.toggleStyle('retroanime')
  assert.equal(state.styleSelections[0].name, 'retroanime')
  controller.selectCheckpoint('moody-v7')
  assert.equal(state.options.sampling_preset, 'moody')
  controller.setCheckpointVisible('moody-v7', false)
  assert.equal(state.settings.image.default_checkpoint, 'official-int8')
  assert.equal(state.options.checkpoint, 'official-int8')
})

test('selecting a user LoRA disables filter bypass without re-enabling it on removal', () => {
  let state = {
    modules: { identity: false },
    options: { checkpoint: 'official-int8', filter_mode: 'balanced', filter_strength: 1 },
    styleSelections: [], userLoraSelections: [],
    userLoraCatalog: [{ filename: 'face.safetensors', recommended_strength: 0.8 }],
    settings: { image: { visible_checkpoints: ['official-int8', 'official'], default_checkpoint: 'official-int8' } },
    checkpointStatus: {}
  }
  const controller = new ImageConfigurationController({ api: {}, catalogs: { kreaStyleCatalog: [], checkpointDisplayChoices: [] }, actions: {
    getState: () => state, patch: (patch) => state = { ...state, ...patch }
  } })
  controller.toggleUserLora('face.safetensors')
  assert.equal(state.userLoraSelections.length, 1)
  assert.equal(state.options.filter_mode, 'off')
  assert.equal(state.options.filter_strength, 0)
  controller.toggleUserLora('face.safetensors')
  assert.equal(state.userLoraSelections.length, 0)
  assert.equal(state.options.filter_mode, 'off')
})
