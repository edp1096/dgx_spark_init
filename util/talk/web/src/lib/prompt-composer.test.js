import { test } from 'node:test';
import assert from 'node:assert/strict';
import fixture from '../../../internal/config/testdata/prompt-composer.json' with { type: 'json' };
import { renderPrompt, toggleCondition, removeBlock, switchPromptMode, selectionError } from './prompt-composer.js';

test('preview matches the server golden text independent of click order', () => {
  const c = structuredClone(fixture.composer);
  assert.equal(renderPrompt(c), fixture.expected);
  c.condition_ids.reverse(); assert.equal(renderPrompt(c), fixture.expected);
});
test('exclusive choices replace one another while independent conditions remain', () => {
  const c = structuredClone(fixture.composer);
  c.blocks.push({ id: 'long', kind: 'condition', category: 'length', group: 'length', name: 'long', prompt: 'long' });
  toggleCondition(c, 'long', true);
  assert.deepEqual(c.condition_ids, ['last', 'long']);
  assert.equal(selectionError(c), '');
  c.condition_ids.push('first'); assert.notEqual(selectionError(c), '');
});
test('deleting a block removes references from current and saved selections', () => {
  const c = structuredClone(fixture.composer);
  c.combinations = [{ id: 'saved', persona_id: 'p', condition_ids: ['first'], extra: '' }];
  removeBlock(c, 'first'); removeBlock(c, 'p');
  assert.deepEqual(c.condition_ids, ['last']);
  assert.equal(c.persona_id, '');
  assert.deepEqual(c.combinations[0].condition_ids, []);
  assert.equal(c.combinations[0].persona_id, '');
});
test('switching from edited direct text preserves that text', () => {
  const model = { system_prompt: 'new direct text', prompt_composer: { ...structuredClone(fixture.composer), enabled: false } };
  switchPromptMode(model, true);
  assert.equal(model.prompt_composer.extra, 'new direct text');
  assert.equal(model.prompt_composer.persona_id, '');
  switchPromptMode(model, false);
  assert.equal(model.system_prompt, '[직접 추가]\nnew direct text');
});
