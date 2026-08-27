import test from 'node:test'
import assert from 'node:assert/strict'
import { identityHasExtraUserPrompt, identityPreserveDefaults, implicitModulePrompt, isPureOutpaint, rawImagePrompt } from './imageWorkflow.js'

test('identity workflow keeps module fallback separate from a real user edit', () => {
  const input = { modules: { identity: true, depth: true }, identityPreset: 'tryon', anypaintMask: null, anypaintImage: null, options: {} }
  const implicit = implicitModulePrompt(input)
  assert.match(implicit, /complete outfit/)
  assert.match(implicit, /pose reference/)
  assert.equal(identityHasExtraUserPrompt({ enteredPrompt: implicit, implicitPrompt: implicit }), false)
  assert.equal(identityHasExtraUserPrompt({ enteredPrompt: 'Change the dress to red', implicitPrompt: implicit }), true)
})

test('raw identity prompt strips legacy envelope and adds missing pose instruction', () => {
  const prompt = rawImagePrompt({ enteredPrompt: 'Change: wear a red coat\nPreserve: background', implicitPrompt: '', modules: { identity: true, depth: true }, identityPreset: '', identityPreserveCustom: 'the face' })
  assert.equal(prompt, 'wear a red coat\nThe person now holds the same pose shown in the pose reference.\nKeep the face unchanged.')
})

test('outpaint and identity preserve defaults remain deterministic', () => {
  assert.equal(isPureOutpaint({ modules: { anypaint: true }, anypaintImage: {}, anypaintMask: null, options: { outpaint_left: 64 } }), true)
  assert.deepEqual(identityPreserveDefaults('tryon', ['identity'], true), ['identity', 'face', 'hair', 'body', 'background', 'lighting', 'untouched'])
})
