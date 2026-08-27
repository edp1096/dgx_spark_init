import test from 'node:test'
import assert from 'node:assert/strict'
import {
  appendMediaInputs,
  clearMediaInputs,
  mediaInputPreview,
  normalizeMediaInput,
  normalizeImageFiles,
  removeMediaInput,
  replaceMediaInput
} from './mediaInputs.js'

function fakeURLs() {
  const revoked = []
  let next = 1
  return {
    createObjectURL: () => `blob:test-${next++}`,
    revokeObjectURL: (url) => revoked.push(url),
    revoked
  }
}

test('local and server media inputs share one reusable shape', () => {
  const urls = fakeURLs()
  const local = normalizeMediaInput({ name: 'local.png', type: 'image/png' }, urls.createObjectURL)
  const server = normalizeMediaInput({ server: true, ref: 'job:output', url: '/api/output.png', name: 'saved.png' }, urls.createObjectURL)
  assert.equal(local.file.name, 'local.png')
  assert.equal(local.preview, 'blob:test-1')
  assert.equal(server.preview, '/api/output.png')
  assert.equal(mediaInputPreview(server), '/api/output.png')
})

test('local preset metadata survives normalization', () => {
  const urls = fakeURLs()
  const file = { name: 'pose.webp', type: 'image/webp', poseID: 'pose-12', posePrompt: 'arms raised' }
  const input = normalizeMediaInput(file, urls.createObjectURL)
  assert.equal(input.poseID, 'pose-12')
  assert.equal(input.posePrompt, 'arms raised')
})

test('limited media lists revoke overflow, removal and clearing', () => {
  const urls = fakeURLs()
  const files = normalizeImageFiles([
    { name: 'one.png', type: 'image/png' },
    { name: 'skip.txt', type: 'text/plain' }
  ], urls.createObjectURL)
  let list = appendMediaInputs(files, [
    { name: 'two.png', type: 'image/png' },
    { name: 'three.png', type: 'image/png' }
  ], 2, urls)
  assert.deepEqual(list.map((item) => item.name), ['one.png', 'two.png'])
  assert.deepEqual(urls.revoked, ['blob:test-3'])
  list = removeMediaInput(list, 0, urls.revokeObjectURL)
  assert.deepEqual(urls.revoked, ['blob:test-3', 'blob:test-1'])
  list = clearMediaInputs(list, urls.revokeObjectURL)
  assert.equal(list.length, 0)
  assert.deepEqual(urls.revoked, ['blob:test-3', 'blob:test-1', 'blob:test-2'])
})

test('replacing a media input releases only the previous local preview', () => {
  const urls = fakeURLs()
  const first = normalizeMediaInput({ name: 'first.png', type: 'image/png' }, urls.createObjectURL)
  const second = replaceMediaInput(first, { server: true, name: 'saved.png', url: '/saved.png' }, urls)
  assert.equal(second.preview, '/saved.png')
  assert.deepEqual(urls.revoked, ['blob:test-1'])
})
