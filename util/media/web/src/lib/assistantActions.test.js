import test from 'node:test'
import assert from 'node:assert/strict'
import { applyAssistantActionList } from './assistantActions.js'

test('assistant actions update forms through explicit host setters', () => {
  let imageForm = { prompt: '', width: 1024, height: 1024, seed: -1 }
  let tab = ''
  const env = {
    switchTab: (value) => tab = value,
    getImageForm: () => imageForm,
    setImageForm: (value) => imageForm = value,
    setImageEnhanceEnabled: () => {},
    setImageResolutionMode: () => {},
    resetImageEnhancement: () => {},
  }
  applyAssistantActionList([{ type: 'set_image', prompt: 'night city', width: 1001, height: 777, seed: 4 }], env)
  assert.equal(tab, 'image')
  assert.deepEqual(imageForm, { prompt: 'night city', width: 1000, height: 776, seed: 4 })
})
