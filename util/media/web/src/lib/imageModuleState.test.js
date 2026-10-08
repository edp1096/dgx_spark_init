import test from 'node:test'
import assert from 'node:assert/strict'
import {
  selectionLabel,
  toggleImageModuleState,
  toggleStyleSelection,
  toggleUserLoraSelection,
  updateSelectionStrength
} from './imageModuleState.js'

const modules = { identity: false, depth: false, vision: false }
const options = { steps: 8, filter_mode: 'balanced', filter_strength: 1 }

test('identity activation applies its reliable baseline without UI side effects', () => {
  const result = toggleImageModuleState({ module: 'identity', modules, options, preserveItems: ['identity', 'pose'], imagePixels: 4096 * 4096 })
  assert.equal(result.modules.identity, true)
  assert.equal(result.options.steps, 10)
  assert.equal(result.options.filter_mode, 'off')
  assert.equal(result.applySmartResolution, true)
  assert.equal(result.imageMegapixels, 2)
})

test('depth activation removes only pose and composition preservation', () => {
  const result = toggleImageModuleState({ module: 'depth', modules, options, preserveItems: ['identity', 'pose', 'composition', 'lighting'], imagePixels: 1 })
  assert.deepEqual(result.preserveItems, ['identity', 'lighting'])
  const disabled = toggleImageModuleState({ module: 'depth', modules: result.modules, options, preserveItems: result.preserveItems, imagePixels: 1 })
  assert.deepEqual(disabled.clearTargets, ['depth'])
})

test('LoRA and style selections keep recommendation, limits and numeric strength', () => {
  const catalog = [{ filename: 'detail.safetensors', name: 'Detail', recommended_strength: 0.65 }]
  let loras = toggleUserLoraSelection([], catalog, 'detail.safetensors')
  assert.deepEqual(loras, [{ filename: 'detail.safetensors', strength: 0.65 }])
  loras = updateSelectionStrength(loras, 'filename', 'detail.safetensors', '0.8')
  assert.equal(loras[0].strength, 0.8)
  assert.equal(selectionLabel(catalog, 'filename', 'detail.safetensors', 'name'), 'Detail')
  assert.deepEqual(toggleStyleSelection([{ name: 'retroanime', strength: 1 }], 'retroanime'), [])
})
