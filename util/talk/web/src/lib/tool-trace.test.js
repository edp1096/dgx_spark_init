import test from 'node:test';
import assert from 'node:assert/strict';
import { finishTool, requestToolApproval, resolveToolApproval, startTool } from './tool-trace.js';

test('tool trace transitions from running to completed', () => {
  const message = { activity: '', tool_trace: [] };
  startTool(message, { id: 'tool-1', name: 'web_search', arguments: '{}' });
  assert.equal(message.activity, 'tool');
  assert.equal(message.tool_trace[0].running, true);

  finishTool(message, { id: 'tool-1', result: 'done' });
  assert.equal(message.activity, 'reasoning');
  assert.equal(message.tool_trace[0].running, false);
  assert.equal(message.tool_trace[0].result, 'done');
});

test('memory approval retains the user-visible proposal until resolved', () => {
  const message = { activity: 'tool', tool_trace: [] };
  startTool(message, { id: 'memory-1', name: 'memory_propose', arguments: '{}' });
  requestToolApproval(message, {
    id: 'memory-1', approval_id: 'approval-1', approval_kind: 'memory',
    kind: 'user', title: '응답 길이', content: '답변은 두 문장 이하로 작성한다.',
  });
  assert.equal(message.tool_trace[0].approval_required, true);
  assert.equal(message.tool_trace[0].title, '응답 길이');
  assert.equal(message.tool_trace[0].content, '답변은 두 문장 이하로 작성한다.');

  resolveToolApproval(message, { id: 'memory-1', approved: true, decision: 'once' });
  assert.equal(message.tool_trace[0].approval_required, false);
  assert.equal(message.tool_trace[0].approved, true);
});

test('memory management approval retains the exact target and proposed change', () => {
  const message = { activity: 'tool', tool_trace: [] };
  startTool(message, { id: 'memory-update', name: 'memory_manage', arguments: '{"action":"update","memory_id":7}' });
  requestToolApproval(message, {
    id: 'memory-update', approval_id: 'approval-update', approval_kind: 'memory_manage', action: 'update', memory_id: 7,
    before_kind: 'memory', before_title: '장비', before_content: 'DGX Spark', before_enabled: true,
    kind: 'user', title: '주력 장비', content: 'DGX Spark', enabled: true,
  });
  assert.equal(message.tool_trace[0].memory_id, 7);
  assert.equal(message.tool_trace[0].before_title, '장비');
  assert.equal(message.tool_trace[0].title, '주력 장비');
});

test('knowledge import approval retains the target collection and source URLs', () => {
  const message = { activity: 'tool', tool_trace: [] };
  startTool(message, { id: 'knowledge-import', name: 'knowledge_import', arguments: '{"action":"import_urls"}' });
  requestToolApproval(message, {
    id: 'knowledge-import', approval_id: 'approval-knowledge', approval_kind: 'knowledge_import',
    collection_id: 3, collection_name: '통풍 보관함',
    urls: ['https://example.com/guideline.pdf', 'https://example.org/review.pdf'], mode: 'auto',
  });
  assert.equal(message.tool_trace[0].collection_name, '통풍 보관함');
  assert.deepEqual(message.tool_trace[0].urls, ['https://example.com/guideline.pdf', 'https://example.org/review.pdf']);
});
