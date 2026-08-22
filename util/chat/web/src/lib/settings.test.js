import assert from 'node:assert/strict';
import test from 'node:test';
import { normalizePublicSettings } from './settings.js';

test('normalizes partial public settings without duplicating backend defaults', () => {
  const settings = normalizePublicSettings({ model: {}, appearance: { theme: 'unknown' } });
  assert.deepEqual(settings.model.system_prompt_presets, []);
  assert.equal(settings.model.system_prompt_preset, '');
  assert.equal(settings.appearance.theme, 'system');
  for (const section of ['server', 'context', 'asr', 'tts', 'tools', 'extra']) {
    assert.deepEqual(settings[section], {});
  }
});
