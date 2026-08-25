// The Go configuration normalizer is the source of truth for defaults. This
// function only makes older or partial API responses safe for the UI to edit.
export function normalizePublicSettings(settings) {
  if (!settings || typeof settings !== 'object') return settings;
  settings.server ||= {};
  settings.model ||= {};
  settings.context ||= {};
  settings.asr ||= {};
  settings.tts ||= {};
  settings.tools ||= {};
  settings.image ||= { enabled: true, endpoint: 'http://127.0.0.1:8691', model: 'krea2-turbo-nvfp4', default_size: '1024x1024', timeout: '30m' };
  settings.extra ||= {};
  settings.appearance ||= {};
  if (!Array.isArray(settings.model.system_prompt_presets)) settings.model.system_prompt_presets = [];
  settings.model.system_prompt_preset ||= '';
  if (!['dark', 'light', 'system'].includes(settings.appearance.theme)) settings.appearance.theme = 'system';
  return settings;
}
