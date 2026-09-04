import assert from 'node:assert/strict';
import test from 'node:test';
import { applyExternalModelType, normalizePublicSettings } from './settings.js';

test('normalizes partial public settings without duplicating backend defaults', () => {
  const settings = normalizePublicSettings({ model: {}, appearance: { theme: 'unknown' } });
  assert.deepEqual(settings.model.system_prompt_presets, []);
  assert.equal(settings.image.endpoint, 'http://127.0.0.1:8691');
  assert.equal(settings.image.mode, 'basic');
  assert.equal(settings.model.system_prompt_preset, '');
  assert.equal(settings.appearance.theme, 'system');
  assert.deepEqual(settings.tts, { hanja_reading: 'korean', omit_parentheticals: true });
  assert.deepEqual(settings.memory, { always_max_results: 6, always_token_budget: 1024, max_results: 5, token_budget: 2048 });
  for (const section of ['server', 'context', 'asr', 'tools']) {
    assert.deepEqual(settings[section], {});
  }
  assert.deepEqual(settings.extra, { collector_enabled: true });
});

test('preserves Japanese automatic Hanja reading', () => {
  const settings = normalizePublicSettings({ tts: { hanja_reading: 'japanese', omit_parentheticals: false } });
  assert.equal(settings.tts.hanja_reading, 'japanese');
  assert.equal(settings.tts.omit_parentheticals, false);
});

test('selecting the Entrpi GLM-5.3 type applies its single-server defaults', () => {
  const settings = normalizePublicSettings({
    model: { endpoint: 'http://192.168.100.61:8000', default_model: 'old', reasoning_effort: 'low' },
    context: { window_tokens: 32768 },
    asr: { enabled: true },
    tts: { enabled: true },
    tools: { media_import_enabled: true },
    image: { enabled: true },
    extra: { ssh_enabled: true, collector_enabled: true },
  });
  applyExternalModelType(settings, 'glm5.3');
  assert.equal(settings.model.endpoint, 'http://192.168.100.61:8000');
  assert.equal(settings.model.default_model, 'glm-5.3-flash');
  assert.equal(settings.model.model_type, 'glm5.3');
  assert.equal(settings.model.reasoning_effort, 'max');
  assert.equal(settings.context.window_tokens, 524288);
  assert.equal(settings.asr.enabled, false);
  assert.equal(settings.tts.enabled, false);
  assert.equal(settings.image.enabled, false);
  assert.equal(settings.extra.ssh_enabled, false);
  assert.equal(settings.extra.collector_enabled, false);
  assert.equal(settings.tools.media_import_enabled, false);
});

test('selecting another external model type changes only the request profile', () => {
  const settings = normalizePublicSettings({ model: { default_model: 'custom' } });
  applyExternalModelType(settings, 'generic');
  assert.equal(settings.model.model_type, 'generic');
  assert.equal(settings.model.default_model, 'custom');
});
