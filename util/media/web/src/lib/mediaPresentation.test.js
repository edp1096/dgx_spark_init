import assert from 'node:assert/strict'
import test from 'node:test'

import {
  captionLanguage,
  compactElapsed,
  formatBytes,
  imageModuleSummary,
  imagePromptModalText,
  mediaSummary,
  subtitleTranslationWarningText
} from './mediaPresentation.js'

test('media and subtitle presentation is derived without mutating jobs', () => {
  const job = {
    prompt: 'original',
    params: {
      translation_mode: 'none',
      detected_language: 'Japanese',
      media: { width: 1920, height: 1080, duration: 65, size: 1048576 },
      translation_warnings: [{ segment: 3, source: '原文', reason: '번역 실패' }]
    }
  }
  const before = structuredClone(job)
  assert.equal(captionLanguage(job), 'ja')
  assert.equal(mediaSummary(job), '1920×1080 · 1:05 · 1.0 MB')
  assert.equal(subtitleTranslationWarningText(job), '3번 자막\n원문: 原文\n번역 실패')
  assert.deepEqual(job, before)
})

test('image and elapsed labels preserve concise gallery text', () => {
  assert.equal(formatBytes(1536), '1.5 KB')
  assert.equal(compactElapsed(3661), '1시간 1분')
  assert.equal(imageModuleSummary({ params: { mode: 'create', identity: true, depth: true } }), ' · Identity + Depth')
  assert.equal(imagePromptModalText({ prompt: 'short', params: { enhanced_prompt: 'detailed' } }), '원문\nshort\n\n실제 생성 프롬프트\ndetailed')
})
