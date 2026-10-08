import test from 'node:test'
import assert from 'node:assert/strict'
import { get } from 'svelte/store'
import { ImageInputController } from './imageInputController.js'

test('image input controller keeps server references reusable by every picker route', () => {
  const controller = new ImageInputController()
  const saved = { server: true, ref: 'job:output:0', url: '/output.png', name: 'saved.png' }
  controller.addRefs([saved], 4)
  controller.addKreaRefs('vision', [saved])
  controller.addIdentityReferences([saved])
  controller.setImage('depth', { ...saved, poseID: 'pose-1' })
  const state = get(controller.state)
  assert.equal(state.refs[0].ref, 'job:output:0')
  assert.equal(state.visionImages[0].preview, '/output.png')
  assert.equal(state.identityReferences[0].name, 'saved.png')
  assert.equal(state.depthImage.poseID, 'pose-1')
  controller.destroy()
})

test('changing AnyPaint source clears its stale mask', () => {
  const controller = new ImageInputController()
  const source = { server: true, ref: 'one', url: '/one.png' }
  controller.setImage('anypaint', source)
  controller.setImage('anypaintMask', { server: true, ref: 'mask', url: '/mask.png' })
  controller.setImage('anypaint', { server: true, ref: 'two', url: '/two.png' })
  const state = get(controller.state)
  assert.equal(state.anypaintImage.ref, 'two')
  assert.equal(state.anypaintMask, null)
  controller.destroy()
})
