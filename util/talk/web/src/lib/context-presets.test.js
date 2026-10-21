import test from 'node:test';
import assert from 'node:assert/strict';
import { effectiveOutputReserve } from './context-presets.js';

test('a manual output reserve carried from a larger model leaves input space', () => {
  const context = { output_auto: false, output_reserve: 262144, safety_margin: 2048 };
  assert.equal(effectiveOutputReserve({ ...context, window_tokens: 1048576 }), 262144);
  assert.equal(effectiveOutputReserve({ ...context, window_tokens: 262144 }), 65536);
  assert.equal(effectiveOutputReserve({ ...context, window_tokens: 0 }), 262144);
  assert.equal(effectiveOutputReserve({ output_reserve: 16384, window_tokens: 262144 }), 16384);
});
