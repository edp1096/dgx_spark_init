async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
  if (response.status === 204) return null;
  return response.json();
}

export const listSessions = () => request('/api/sessions');
export const createSession = (title = '새 대화', model = '', reasoningEffort = '') => request('/api/sessions', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ title, model, reasoning_effort: reasoningEffort }),
});
export const deleteSession = (id) => request(`/api/sessions/${id}`, { method: 'DELETE' });
export const renameSession = (id, title) => request(`/api/sessions/${id}`, {
  method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }),
});
export const setSessionGroup = (id, groupId = '') => request(`/api/sessions/${id}/group`, {
  method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ group_id: groupId }),
});
export const listGroups = () => request('/api/groups');
export const createGroup = (name) => request('/api/groups', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }),
});
export const renameGroup = (id, name) => request(`/api/groups/${id}`, {
  method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }),
});
export const moveGroup = (id, direction) => request(`/api/groups/${id}/move`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ direction }),
});
export const deleteGroup = (id) => request(`/api/groups/${id}`, { method: 'DELETE' });
export const listMessages = (id) => request(`/api/sessions/${id}/messages`);
export const getHealth = () => request('/api/health');
export const getModels = () => request('/api/models');
export const getConfig = () => request('/api/config');
export const saveConfig = (config) => request('/api/config', {
  method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config),
});
export const getMediaUsage = () => request('/api/media');
export const cleanupMedia = (keepIds = []) => request('/api/media', {
  method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ keep_ids: keepIds }),
});

export const uploadImage = (file, signal) => {
  const body = new FormData();
  body.append('image', file);
  return request('/api/images', { method: 'POST', body, signal });
};

export const uploadAttachment = (file, signal) => {
  const body = new FormData();
  body.append('file', file);
  return request('/api/files', { method: 'POST', body, signal });
};

export async function streamChat(sessionId, content, attachments, model, reasoningEffort, toolsEnabled, signal, handlers) {
  const response = await fetch('/api/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, content, attachments, model, reasoning_effort: reasoningEffort, tools_enabled: toolsEnabled }), signal,
  });
  return consumeSSE(response, handlers);
}

export async function retryMessage(messageId, model, reasoningEffort, toolsEnabled, userVariant, signal, handlers) {
  const response = await fetch(`/api/messages/${messageId}/retry`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, reasoning_effort: reasoningEffort, tools_enabled: toolsEnabled, user_variant: userVariant }), signal,
  });
  return consumeSSE(response, handlers);
}

export async function editMessage(messageId, content, attachments, model, reasoningEffort, toolsEnabled, signal, handlers) {
  const response = await fetch(`/api/messages/${messageId}/edit`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content, attachments, model, reasoning_effort: reasoningEffort, tools_enabled: toolsEnabled }), signal,
  });
  return consumeSSE(response, handlers);
}

async function consumeSSE(response, handlers) {
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
      if (event === 'tool_result') handlers.toolResult?.(data);
      if (event === 'error') throw new Error(data.error || '응답 오류');
      if (event === 'done') handlers.done?.();
    }
  }
}
