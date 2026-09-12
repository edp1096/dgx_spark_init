import test from 'node:test';
import assert from 'node:assert/strict';
import { formatMetric, performanceTitle } from './performance.js';
import { applyVariant } from './message-variants.js';
import { createStreamHandlers } from './chat-stream.js';
import { consumeSSE } from '../api/chat.js';

test('metrics distinguish unavailable, valid zero and estimated values', () => {
  for (const value of [undefined, null, NaN, Infinity, -1, '20']) assert.equal(formatMetric(value, true, 'tok/s'), '—');
  assert.equal(formatMetric(0, false, 's'), '0 s');
  assert.equal(formatMetric(18.44, true, 'tok/s'), '≈ 18.4 tok/s');
  assert.equal(formatMetric(2.125, false, 's'), '2.13 s');
  assert.match(performanceTitle({ calls: 2, cached_tokens: 512 }, 'pp'), /512/);
});

test('final streamed measurements replace live estimates and survive variant selection', async () => {
  const message = { content: '', reasoning_content: '' };
  const measured = { pp: 1000, tg: 20, ttft: .5, calls: 1 };
  const sse = 'event: performance\ndata: {"tg":12,"tg_estimated":true,"live":true}\n\n'
    + 'event: reasoning\ndata: {"delta":"생각"}\n\n'
    + 'event: delta\ndata: {"delta":"답변"}\n\n'
    + `event: performance\ndata: ${JSON.stringify(measured)}\n\n`
    + 'event: done\ndata: {}\n\n';
  await consumeSSE(new Response(sse), createStreamHandlers(message, () => {}));
  assert.deepEqual(message.performance, measured);
  message.variants = [{ content: 'legacy' }, { content: 'measured', performance: measured }];
  applyVariant(message, 0);
  assert.equal(message.performance, null);
  applyVariant(message, 1);
  assert.deepEqual(message.performance, measured);
});
