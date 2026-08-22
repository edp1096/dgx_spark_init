export async function streamChat(sessionId, content, attachments, model, reasoningEffort, toolsEnabled, signal, handlers) {
  const response = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session_id: sessionId, content, attachments, model, reasoning_effort: reasoningEffort, tools_enabled: toolsEnabled }), signal });
  return consumeSSE(response, handlers);
}
export async function retryMessage(messageId, model, reasoningEffort, toolsEnabled, userVariant, signal, handlers) {
  const response = await fetch(`/api/messages/${messageId}/retry`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model, reasoning_effort: reasoningEffort, tools_enabled: toolsEnabled, user_variant: userVariant }), signal });
  return consumeSSE(response, handlers);
}
export async function editMessage(messageId, content, attachments, model, reasoningEffort, toolsEnabled, signal, handlers) {
  const response = await fetch(`/api/messages/${messageId}/edit`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content, attachments, model, reasoning_effort: reasoningEffort, tools_enabled: toolsEnabled }), signal });
  return consumeSSE(response, handlers);
}
export async function consumeSSE(response, handlers) {
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split('\n\n');
    buffer = events.pop() || '';
    for (const block of events) {
      const event = block.match(/^event:\s*(.+)$/m)?.[1] || 'message';
      const raw = block.match(/^data:\s*(.+)$/m)?.[1];
      if (!raw) continue;
      const data = JSON.parse(raw);
      if (event === 'delta') handlers.delta?.(data.delta || '');
      if (event === 'reasoning') handlers.reasoning?.(data.delta || '');
      if (event === 'tool_start') handlers.toolStart?.(data);
      if (event === 'tool_approval') handlers.toolApproval?.(data);
      if (event === 'tool_approval_resolved') handlers.toolApprovalResolved?.(data);
      if (event === 'tool_execution') handlers.toolExecution?.(data);
      if (event === 'tool_output') handlers.toolOutput?.(data);
      if (event === 'tool_result') handlers.toolResult?.(data);
      if (event === 'media_attached') handlers.mediaAttached?.(data);
      if (event === 'ssh_grant_changed') handlers.sshGrantChanged?.(data);
      if (event === 'context') handlers.context?.(data);
      if (event === 'error') throw new Error(data.error || '응답 오류');
      if (event === 'done') handlers.done?.();
    }
  }
}
