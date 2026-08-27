import test from 'node:test'
import assert from 'node:assert/strict'
import { get } from 'svelte/store'
import { ImageSequenceController, imageSequenceBlockedMessage } from './imageSequenceController.js'

test('image sequence owns scenes and mask preview lifetime', () => {
  const revoked = []
  let nextURL = 0
  const controller = new ImageSequenceController({
    createObjectURL: () => `blob:${++nextURL}`,
    revokeObjectURL: (value) => revoked.push(value)
  })
  controller.addScene()
  controller.updatePrompt(2, 'third scene')
  controller.useMask(2, { name: 'mask.png' })
  assert.deepEqual(get(controller.state).regions, ['all', 'all', 'custom'])
  controller.updateRegion(2, 'left-arm')
  assert.deepEqual(revoked, ['blob:1'])
  assert.equal(get(controller.state).masks[2], null)
  controller.removeScene(2)
  assert.equal(get(controller.state).prompts.length, 2)
  controller.destroy()
})

test('image sequence reset and example keep arrays aligned', () => {
  const controller = new ImageSequenceController({ createObjectURL: () => '', revokeObjectURL: () => {} })
  controller.applyRobotExample()
  let state = get(controller.state)
  assert.equal(state.prompts.length, 3)
  assert.deepEqual(state.regions, ['all', 'left-arm', 'left-arm'])
  assert.equal(state.strength, 0.65)
  controller.reset(['first', ''])
  state = get(controller.state)
  assert.deepEqual(state.prompts, ['first', ''])
  assert.deepEqual(state.regions, ['all', 'all'])
  assert.equal(state.base, null)
  controller.destroy()
})

test('sequence compatibility reports only real blockers', () => {
  const base = { mode: 'create', modules: {}, moduleReason: '', width: 1024, height: 1024 }
  assert.equal(imageSequenceBlockedMessage(base), '')
  assert.match(imageSequenceBlockedMessage({ ...base, modules: { depth: true } }), /자세·구도/)
  assert.match(imageSequenceBlockedMessage({ ...base, mode: 'edit' }), /새 이미지 생성/)
  assert.match(imageSequenceBlockedMessage({ ...base, width: 2048, height: 2048 }), /2MP 이하/)
})
