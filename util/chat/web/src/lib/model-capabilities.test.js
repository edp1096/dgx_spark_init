import assert from 'node:assert/strict';
import test from 'node:test';
import { modelCapabilities, normalizeReasoningEffort, reasoningEffortLabel, thinkingToggleValue } from './model-capabilities.js';

test('Gemma 4 exposes a boolean thinking control', () => {
  assert.deepEqual(modelCapabilities('gemma4'), {
    family: 'gemma4', reasoning: 'toggle', reasoningLevels: ['on', 'none'],
  });
});

test('Qwen 3.8 keeps typed effort levels without unsupported high', () => {
  assert.deepEqual(modelCapabilities('qwen3.8'), {
    family: 'qwen3.8', reasoning: 'effort', reasoningLevels: ['none', 'low', 'medium', 'xhigh'],
  });
});

test('GLM-5.3 exposes Entrpi thinking off and native effort levels', () => {
  assert.deepEqual(modelCapabilities('glm5.3'), {
    family: 'glm5.3', reasoning: 'effort', reasoningLevels: ['off', 'low', 'high', 'max'],
  });
  assert.equal(normalizeReasoningEffort('glm5.3', 'none'), 'off');
  assert.equal(normalizeReasoningEffort('glm5.3', 'low'), 'low');
  assert.equal(normalizeReasoningEffort('glm5.3', 'high'), 'high');
  assert.equal(normalizeReasoningEffort('glm5.3', 'max'), 'max');
  assert.equal(normalizeReasoningEffort('glm5.3', 'xhigh'), 'max');
  assert.equal(reasoningEffortLabel('off'), '꺼짐');
  assert.equal(reasoningEffortLabel('high'), 'High');
  assert.equal(reasoningEffortLabel('max'), 'Max');
});

test('model names are not guessed', () => {
  assert.equal(modelCapabilities('nvidia/Gemma-4-31B-IT-NVFP4').family, 'generic');
});

test('legacy effort values map cleanly to the Gemma toggle', () => {
  assert.equal(thinkingToggleValue('low'), 'on');
  assert.equal(thinkingToggleValue('none'), 'none');
});

test('Qwen 3.8 accepts only its four supported effort values', () => {
  assert.equal(normalizeReasoningEffort('qwen3.8', 'xhigh'), 'xhigh');
  assert.equal(normalizeReasoningEffort('qwen3.8', 'high'), 'medium');
  assert.equal(normalizeReasoningEffort('qwen3.8', 'on'), 'medium');
  assert.equal(reasoningEffortLabel('none'), '꺼짐');
  assert.equal(reasoningEffortLabel('medium'), 'Medium');
});
