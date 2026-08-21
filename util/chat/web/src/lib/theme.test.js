import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeTheme, resolveTheme, storedTheme } from './theme.js';

test('theme preference resolves explicit and system modes', () => {
  assert.equal(resolveTheme('dark', false), 'dark');
  assert.equal(resolveTheme('light', true), 'light');
  assert.equal(resolveTheme('system', true), 'dark');
  assert.equal(resolveTheme('system', false), 'light');
  assert.equal(normalizeTheme('unknown'), 'system');
});

test('stored theme falls back safely', () => {
  assert.equal(storedTheme({ getItem: () => 'light' }), 'light');
  assert.equal(storedTheme({ getItem: () => 'invalid' }), 'system');
  assert.equal(storedTheme({ getItem: () => { throw new Error('blocked'); } }), 'system');
});
