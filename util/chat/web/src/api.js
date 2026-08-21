async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    let message = text;
    try { message = JSON.parse(text)?.error || text; } catch { /* plain text error */ }
    throw new Error(message || `HTTP ${response.status}`);
  }
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
export const getContextState = (id) => request(`/api/sessions/${id}/context`);
export const compactContext = (id) => request(`/api/sessions/${id}/context/compact`, { method: 'POST' });
export const clearContext = (id) => request(`/api/sessions/${id}/context`, { method: 'DELETE' });
export const listSSHConversationGrants = (id) => request(`/api/sessions/${id}/ssh-grants`);
export const revokeSSHConversationGrant = (sessionId, hostId) => request(`/api/sessions/${sessionId}/ssh-grants/${encodeURIComponent(hostId)}`, { method: 'DELETE' });
export const clearSSHConversationGrants = (id) => request(`/api/sessions/${id}/ssh-grants`, { method: 'DELETE' });
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

export const listSSHHosts = () => request('/api/ssh/hosts');
export const createSSHHost = (host) => request('/api/ssh/hosts', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(host),
});
export const updateSSHHost = (id, host) => request(`/api/ssh/hosts/${id}`, {
  method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(host),
});
export const deleteSSHHost = (id) => request(`/api/ssh/hosts/${id}`, { method: 'DELETE' });
export const listSSHKeys = () => request('/api/ssh/keys');
export const generateSSHKey = (keyId) => request('/api/ssh/keys/generate', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key_id: keyId }),
});
export const importSSHKey = (keyId, file) => {
  const body = new FormData();
  body.append('key_id', keyId);
  body.append('key', file);
  return request('/api/ssh/keys', { method: 'POST', body });
};
export const deleteSSHKey = (keyId) => request(`/api/ssh/keys/${encodeURIComponent(keyId)}`, { method: 'DELETE' });
export const trustSSHHost = (id, publicKey) => request(`/api/ssh/hosts/${id}/trust`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ public_key: publicKey }),
});
export const answerToolApproval = (id, decision) => request(`/api/tool-approvals/${id}`, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision }),
});

export async function testSSHHost(id) {
  const response = await fetch(`/api/ssh/hosts/${id}/test`, { method: 'POST' });
  const text = await response.text();
  let details = null;
  try { details = JSON.parse(text); } catch { details = null; }
  if (!response.ok) {
    const error = new Error(details?.error || text || `HTTP ${response.status}`);
    error.status = response.status;
    error.details = details;
    throw error;
  }
  return details || {};
}

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

export const transcribeVoice = (blob, filename, signal) => {
  const body = new FormData();
  body.append('audio', blob, filename);
  return request('/api/asr/transcribe', { method: 'POST', body, signal });
};

export async function streamSpeech(text, seed, signal, onChunk) {
  const response = await fetch('/api/tts/speech', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, seed }), signal,
  });
  if (!response.ok) throw new Error((await response.text()) || `TTS HTTP ${response.status}`);
  if (!response.body) throw new Error('TTS 스트림을 읽을 수 없습니다.');
  const sampleRate = Number(response.headers.get('X-Audio-Sample-Rate')) || 24000;
  const reader = response.body.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value?.length) await onChunk(value, sampleRate);
  }
  return { sampleRate };
}

export const uploadMediaURL = (url, signal) => request('/api/media/source', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ url }), signal,
});

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
