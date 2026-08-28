import test from 'node:test'
import assert from 'node:assert/strict'
import { api } from '../api.js'

test('character sheet API preserves its PNG response as a blob', async () => {
  const previousFetch = globalThis.fetch
  const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/api/images/character-sheet')
    assert.equal(options.method, 'POST')
    return new Response(png, { status: 200, headers: { 'Content-Type': 'image/png' } })
  }
  try {
    const result = await api.createSequenceCharacterSheet(new FormData())
    assert.equal(result.type, 'image/png')
    assert.deepEqual(new Uint8Array(await result.arrayBuffer()), png)
  } finally {
    globalThis.fetch = previousFetch
  }
})

test('character sheet progress is correlated by operation id', async () => {
  const previousFetch = globalThis.fetch
  globalThis.fetch = async (url) => {
    assert.equal(url, '/api/images/character-sheet/status?operation_id=character-sheet-a%2Fb')
    return Response.json({ operation: { operation_id: 'character-sheet-a/b', phase: 'sampling', progress: 0.7 } })
  }
  try {
    const result = await api.sequenceCharacterSheetStatus('character-sheet-a/b')
    assert.equal(result.operation.progress, 0.7)
  } finally {
    globalThis.fetch = previousFetch
  }
})
