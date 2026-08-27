import test from 'node:test'
import assert from 'node:assert/strict'
import { ImageConfigurationController } from './imageConfigurationController.js'

test('image configuration keeps checkpoint and style selection coherent', () => {
  let state = {
    modules: { identity: false }, options: { checkpoint: 'official', identity_model: 'convrot', sampling_preset: 'default' },
    styleSelections: [], userLoraSelections: [], userLoraCatalog: [], settings: { image: { visible_checkpoints: ['official', 'moody-v7'], default_checkpoint: 'moody-v7' } }, checkpointStatus: {}
  }
  const controller = new ImageConfigurationController({ api: {}, catalogs: { kreaStyleCatalog: [{ name: 'retroanime', label: 'Retro' }], checkpointDisplayChoices: [['moody-v7', 'Moody']] }, actions: {
    getState: () => state, patch: (patch) => state = { ...state, ...patch }
  } })
  controller.toggleStyle('retroanime')
  assert.equal(state.styleSelections[0].name, 'retroanime')
  controller.selectCheckpoint('moody-v7')
  assert.equal(state.options.sampling_preset, 'moody')
  controller.setCheckpointVisible('moody-v7', false)
  assert.equal(state.settings.image.default_checkpoint, 'official')
  assert.equal(state.options.checkpoint, 'official')
})
