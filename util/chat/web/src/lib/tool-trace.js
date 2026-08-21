export function startTool(message, data) {
  message.activity = 'tool';
  message.tool_trace = [...(message.tool_trace || []), { ...data, result: '', error: '', running: true }];
}

export function finishTool(message, data) {
  const trace = [...(message.tool_trace || [])];
  const index = trace.findIndex((item) => item.id === data.id && item.running);
  if (index >= 0) trace[index] = { ...trace[index], ...data, running: false, approval_required: false };
  else trace.push({ ...data, running: false });
  message.tool_trace = trace;
  message.activity = 'reasoning';
}

function updateTool(message, data, update) {
  const trace = [...(message.tool_trace || [])];
  const index = trace.findIndex((item) => item.id === data.id);
  if (index < 0) return;
  trace[index] = update({ ...trace[index] });
  message.tool_trace = trace;
}

export function requestToolApproval(message, data) {
  updateTool(message, data, (tool) => ({ ...tool, ...data, approval_required: true, approval_answered: false }));
}

export function resolveToolApproval(message, data) {
  updateTool(message, data, (tool) => ({ ...tool, approval_required: false, approval_answered: true, approved: data.approved, approval_decision: data.decision || '' }));
}

export function markToolExecution(message, data) {
  updateTool(message, data, (tool) => ({ ...tool, execution_status: data.status || 'running' }));
}

export function appendToolOutput(message, data) {
  updateTool(message, data, (tool) => ({
    ...tool,
    output: `${tool.output || ''}${data.delta || ''}`,
    streams: { ...(tool.streams || {}), [data.stream || 'stdout']: `${tool.streams?.[data.stream || 'stdout'] || ''}${data.delta || ''}` },
  }));
}
