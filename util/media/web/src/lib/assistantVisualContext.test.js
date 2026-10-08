import test from 'node:test'
import assert from 'node:assert/strict'
import { needsVideoVisualContext } from './assistantVisualContext.js'

test('visual context is requested only for scene-aware video questions', () => {
  assert.equal(needsVideoVisualContext('시작 이미지와 마지막 장면을 연결해줘'), true)
  assert.equal(needsVideoVisualContext('write a video prompt'), true)
  assert.equal(needsVideoVisualContext('안녕하세요'), false)
})
