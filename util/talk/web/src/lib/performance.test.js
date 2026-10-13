import test from 'node:test';
import assert from 'node:assert/strict';
import { formatMetric, performanceTitle, formatElapsed, responseTiming } from './performance.js';
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
  const measured = { pp: 1000, tg: 20, ttft: .5, calls: 1, elapsed_seconds: 83.2, finished_at: '2026-10-11T13:14:05Z', finish_status: 'completed' };
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

test('whole-response elapsed time and local completion clock validate legacy and stopped data', () => {
  assert.equal(formatElapsed(0), '0초');
  assert.equal(formatElapsed(83.2), '1분 23초');
  assert.equal(formatElapsed(3601), '1시간 0분 1초');
  for (const value of [null, -1, NaN, Infinity, '10']) assert.equal(formatElapsed(value), '');
  for (const value of [null, {}, { elapsed_seconds: 1, finished_at: 'bad' }]) assert.equal(responseTiming(value), null);
  const measured = { elapsed_seconds: 83, finished_at: '2026-10-11T13:14:05Z', finish_status: 'completed' };
  const result = responseTiming(measured);
  assert.equal(result.label, '완료');
  assert.equal(result.elapsed, '1분 23초');
  assert.equal(result.clock, new Date(measured.finished_at).toLocaleTimeString('ko-KR', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }));
  assert.equal(responseTiming({ ...measured, finish_status: 'cancelled' }).label, '중지');
  assert.equal(responseTiming({ ...measured, finish_status: 'failed' }).label, '실패');
  assert.equal(responseTiming({ ...measured, live: true }), null);
});
