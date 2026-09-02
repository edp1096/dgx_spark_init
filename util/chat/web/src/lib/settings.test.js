import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizePublicSettings } from './settings.js';

test('normalizes partial public settings without duplicating backend defaults', () => {
  const settings = normalizePublicSettings({ model: {}, appearance: { theme: 'unknown' } });
  assert.deepEqual(settings.model.system_prompt_presets, []);
  assert.equal(settings.image.endpoint, 'http://127.0.0.1:8691');
  assert.equal(settings.image.mode, 'basic');
  assert.equal(settings.model.system_prompt_preset, '');
  assert.equal(settings.appearance.theme, 'system');
  assert.deepEqual(settings.tts, { hanja_reading: 'korean', omit_parentheticals: true });
  for (const section of ['server', 'context', 'memory', 'asr', 'tools', 'extra']) {
    assert.deepEqual(settings[section], {});
  }
});

test('preserves Japanese automatic Hanja reading', () => {
  const settings = normalizePublicSettings({ tts: { hanja_reading: 'japanese', omit_parentheticals: false } });
  assert.equal(settings.tts.hanja_reading, 'japanese');
  assert.equal(settings.tts.omit_parentheticals, false);
});
