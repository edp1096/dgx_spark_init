export function normalizeAvailableModels(models) {
  if (!Array.isArray(models)) return [];
  return [...new Set(models
    .filter((model) => typeof model === 'string')
    .map((model) => model.trim())
    .filter(Boolean))];
}

// A session remembers the model that produced its previous replies, but that
// historical value must not become a selectable model after the runtime has
// switched to a different engine.
export function resolveAvailableModel(models, preferred = '', fallback = '') {
  const available = normalizeAvailableModels(models);
  if (!available.length) return preferred || fallback || '';
  if (preferred && available.includes(preferred)) return preferred;
  if (fallback && available.includes(fallback)) return fallback;
  return available[0];
}
