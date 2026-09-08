import test from 'node:test';
import assert from 'node:assert/strict';
import { applyVariant, hydrateMessages, variantIndices } from './message-variants.js';

test('hydrateMessages selects the latest answer for the active user branch', () => {
  const messages = hydrateMessages([
    {
      role: 'user', content: '두 번째 질문',
      variants: [{ content: '첫 질문' }, { content: '두 번째 질문' }],
    },
    {
      role: 'assistant', content: '이전 답변',
      variants: [
        { content: '첫 질문 답변', parent_variant: 0 },
        { content: '두 번째 질문의 첫 답변', parent_variant: 1 },
        { content: '두 번째 질문의 재시도 답변', parent_variant: 1 },
      ],
    },
  ]);

  assert.equal(messages[0].variant_index, 1);
  assert.equal(messages[1].variant_index, 2);
  assert.equal(messages[1].content, '두 번째 질문의 재시도 답변');
  assert.deepEqual(variantIndices(messages[1], 1, messages), [1, 2]);
});

test('applyVariant updates all visible response fields', () => {
  const message = {
    variants: [{
      content: '답변', reasoning_content: '생각', tool_trace: [{ name: 'web_search' }], attachments: [{ id: 'image' }],
    }],
  };

  assert.equal(applyVariant(message, 0), true);
  assert.equal(message.content, '답변');
  assert.equal(message.reasoning_content, '생각');
  assert.equal(message.tool_trace[0].name, 'web_search');
  assert.equal(message.attachments[0].id, 'image');
  assert.equal(message.variant_index, 0);
});
