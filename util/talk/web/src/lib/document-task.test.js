import test from 'node:test';
import assert from 'node:assert/strict';
import { documentTaskFinished, documentTaskCompletionLabel } from './document-task.js';

test('name consistency completion is distinct from full semantic review', () => {
  const run = { status: 'completed', document: { pipeline: 'terms_v1', terminal: true, terms: { unresolved: 0 }, quality_status: '' } };
  assert.equal(documentTaskFinished(run), true);
  assert.equal(documentTaskCompletionLabel(run), '완료');
  run.document.terms.unresolved = 1;
  assert.equal(documentTaskCompletionLabel(run), '완료 · 확인 필요');
});
