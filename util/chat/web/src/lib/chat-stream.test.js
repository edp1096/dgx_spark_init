import test from 'node:test';
import assert from 'node:assert/strict';
import { createStreamHandlers } from './chat-stream.js';

test('stream handlers update one assistant message and publish every event', () => {
  const message = { content: '', reasoning_content: '', tool_trace: [], activity: '' };
  let publishes = 0;
  const handlers = createStreamHandlers(message, () => { publishes += 1; });

  handlers.reasoning('생각');
  handlers.toolStart({ id: 'one', name: 'web_search' });
  handlers.toolResult({ id: 'one', result: '검색 결과' });
  handlers.delta('답변');

  assert.equal(message.reasoning_content, '생각');
  assert.equal(message.content, '답변');
  assert.equal(message.tool_trace[0].result, '검색 결과');
  assert.equal(message.activity, 'answer');
  assert.equal(publishes, 4);
});
