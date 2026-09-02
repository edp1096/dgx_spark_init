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
