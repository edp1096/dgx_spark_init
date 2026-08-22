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
  settings.extra ||= {};
  settings.appearance ||= {};
  if (!Array.isArray(settings.model.system_prompt_presets)) settings.model.system_prompt_presets = [];
  settings.model.system_prompt_preset ||= '';
  if (!['dark', 'light', 'system'].includes(settings.appearance.theme)) settings.appearance.theme = 'system';
  return settings;
}
