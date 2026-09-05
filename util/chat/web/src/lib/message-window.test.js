import assert from 'node:assert/strict';
import test from 'node:test';
import { initialMessageStart, messageWindowAround, shiftedMessageWindow } from './message-window.js';

test('message window starts with only the latest messages', () => {
  assert.equal(initialMessageStart(120), 88);
  assert.equal(initialMessageStart(20), 0);
});

test('message window moves backward and forward without retaining distant DOM', () => {
  assert.deepEqual(shiftedMessageWindow(120, 88, 120, 'previous'), { start: 64, end: 112 });
  assert.deepEqual(shiftedMessageWindow(120, 64, 112, 'next'), { start: 72, end: 120 });
});

test('message window can jump directly around a hidden search result', () => {
  assert.deepEqual(messageWindowAround(120, 40), { start: 28, end: 76 });
  assert.deepEqual(messageWindowAround(20, 8), { start: 0, end: 20 });
});
