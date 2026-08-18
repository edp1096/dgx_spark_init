export function startTool(message, data) {
  message.activity = 'tool';
  message.tool_trace = [...(message.tool_trace || []), { ...data, result: '', error: '', running: true }];
}

export function finishTool(message, data) {
  const trace = [...(message.tool_trace || [])];
  const index = trace.findIndex((item) => item.id === data.id && item.running);
  if (index >= 0) trace[index] = { ...trace[index], ...data, running: false };
  else trace.push({ ...data, running: false });
  message.tool_trace = trace;
  message.activity = 'reasoning';
}
