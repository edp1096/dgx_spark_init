import test from 'node:test';
import assert from 'node:assert/strict';
import { sessionPage } from './session-pages.js';

test('bounds rendered sessions and clamps the page after deletion or moving', () => {
  const items = Array.from({ length: 1000 }, (_, id) => ({ id }));
  assert.equal(sessionPage(items).items.length, 15);
  assert.equal(sessionPage(items, 1).items[0].id, 15);
  assert.equal(sessionPage(items, 99).items.length, 10);
  assert.equal(sessionPage(items.slice(0, 16), 66).page, 1);
  assert.equal(sessionPage([], 10).page, 0);
  assert.deepEqual(sessionPage([], 10).items, []);
});
