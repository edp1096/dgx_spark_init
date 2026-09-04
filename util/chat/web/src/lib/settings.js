// The Go configuration normalizer is the source of truth for defaults. This
// function only makes older or partial API responses safe for the UI to edit.
export function normalizePublicSettings(settings) {
  if (!settings || typeof settings !== 'object') return settings;
  settings.server ||= {};
  settings.version ||= 2;
  settings.runtime ||= {};
  settings.runtime.mode ||= 'managed';
  settings.runtime.bundle ||= 'flash-next';
  if (!Number.isFinite(Number(settings.runtime.memory_reserve_gib))) settings.runtime.memory_reserve_gib = 8;
  settings.model ||= {};
  settings.context ||= {};
  settings.memory ||= {};
  if (!Number.isFinite(Number(settings.memory.always_max_results)) || Number(settings.memory.always_max_results) < 1) settings.memory.always_max_results = 6;
  if (!Number.isFinite(Number(settings.memory.always_token_budget)) || Number(settings.memory.always_token_budget) < 256) settings.memory.always_token_budget = 1024;
  if (!Number.isFinite(Number(settings.memory.max_results)) || Number(settings.memory.max_results) < 1) settings.memory.max_results = 5;
  if (!Number.isFinite(Number(settings.memory.token_budget)) || Number(settings.memory.token_budget) < 256) settings.memory.token_budget = 2048;
  settings.asr ||= {};
  settings.tts ||= {};
  settings.tools ||= {};
  settings.image ||= { enabled: false, endpoint: 'http://127.0.0.1:8691', model: '', mode: 'basic', default_size: '1024x1024', timeout: '30m' };
  if (!['basic', 'extended'].includes(settings.image.mode)) settings.image.mode = 'basic';
  settings.extra ||= {};
  if (typeof settings.extra.collector_enabled !== 'boolean') settings.extra.collector_enabled = true;
  settings.appearance ||= {};
  if (!Array.isArray(settings.model.system_prompt_presets)) settings.model.system_prompt_presets = [];
  settings.model.system_prompt_preset ||= '';
  if (!Number.isFinite(Number(settings.model.thinking_budget))) settings.model.thinking_budget = 0;
  if (!['korean', 'chinese', 'japanese'].includes(settings.tts.hanja_reading)) settings.tts.hanja_reading = 'korean';
  if (typeof settings.tts.omit_parentheticals !== 'boolean') settings.tts.omit_parentheticals = true;
  if (!['dark', 'light', 'system'].includes(settings.appearance.theme)) settings.appearance.theme = 'system';
  return settings;
}

export function applyExternalModelType(settings, modelType) {
  if (!settings?.model) return settings;
  settings.model.model_type = modelType;
  if (modelType !== 'glm5.3') return settings;

  settings.model.default_model = 'glm-5.3-flash';
  settings.model.reasoning_effort = 'max';
  settings.context.window_tokens = 524288;
  settings.asr.enabled = false;
  settings.tts.enabled = false;
  settings.image.enabled = false;
  settings.extra.ssh_enabled = false;
  settings.extra.collector_enabled = false;
  settings.tools.media_import_enabled = false;
  return settings;
}
