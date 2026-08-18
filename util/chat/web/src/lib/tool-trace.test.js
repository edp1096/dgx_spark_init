import test from 'node:test';
import assert from 'node:assert/strict';
import { finishTool, startTool } from './tool-trace.js';

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
