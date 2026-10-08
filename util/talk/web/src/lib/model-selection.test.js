import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizeAvailableModels, resolveAvailableModel } from './model-selection.js';

test('uses the current runtime model when a conversation model is stale', () => {
  const current = 'qwen-current';
  assert.equal(resolveAvailableModel([current], 'gemma-previous', current), current);
});

test('keeps a conversation model while it remains available', () => {
  assert.equal(resolveAvailableModel(['model-a', 'model-b'], 'model-b', 'model-a'), 'model-b');
});

test('retains the remembered model while the model API is unavailable', () => {
  assert.equal(resolveAvailableModel([], 'remembered-model', 'configured-model'), 'remembered-model');
});

test('normalizes duplicate and blank model entries', () => {
  assert.deepEqual(normalizeAvailableModels([' model-a ', '', 'model-a', null, 'model-b']), ['model-a', 'model-b']);
});
