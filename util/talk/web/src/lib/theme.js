export const themeChoices = ['dark', 'light', 'system'];

export function normalizeTheme(value) {
  return themeChoices.includes(value) ? value : 'system';
}

export function resolveTheme(value, prefersDark = globalThis.matchMedia?.('(prefers-color-scheme: dark)').matches ?? true) {
  const preference = normalizeTheme(value);
  return preference === 'system' ? (prefersDark ? 'dark' : 'light') : preference;
}

export function storedTheme(storage = globalThis.localStorage) {
  try { return normalizeTheme(storage?.getItem('sparktalk.theme')); }
  catch { return 'system'; }
}

export function applyTheme(value, persist = true, root = globalThis.document?.documentElement) {
  const preference = normalizeTheme(value);
  const resolved = resolveTheme(preference);
  if (root) {
    root.dataset.theme = resolved;
    root.dataset.themePreference = preference;
    root.style.colorScheme = resolved;
    const themeColor = globalThis.document?.querySelector('meta[name="theme-color"]');
    themeColor?.setAttribute('content', resolved === 'dark' ? '#10131a' : '#f5f7fb');
  }
  if (persist) {
    try { globalThis.localStorage?.setItem('sparktalk.theme', preference); } catch { /* storage unavailable */ }
  }
  return resolved;
}
