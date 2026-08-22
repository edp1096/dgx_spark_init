import { request } from './request.js';

export const listSessions = () => request('/api/sessions');
export const createSession = (title = '새 대화', model = '', reasoningEffort = '') => request('/api/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title, model, reasoning_effort: reasoningEffort }) });
export const deleteSession = (id) => request(`/api/sessions/${id}`, { method: 'DELETE' });
export const renameSession = (id, title) => request(`/api/sessions/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) });
export const setSessionGroup = (id, groupId = '') => request(`/api/sessions/${id}/group`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ group_id: groupId }) });
export const listGroups = () => request('/api/groups');
export const createGroup = (name) => request('/api/groups', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
export const renameGroup = (id, name) => request(`/api/groups/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) });
export const moveGroup = (id, direction) => request(`/api/groups/${id}/move`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ direction }) });
export const deleteGroup = (id) => request(`/api/groups/${id}`, { method: 'DELETE' });
export const listMessages = (id) => request(`/api/sessions/${id}/messages`);
export const getContextState = (id) => request(`/api/sessions/${id}/context`);
export const compactContext = (id) => request(`/api/sessions/${id}/context/compact`, { method: 'POST' });
export const clearContext = (id) => request(`/api/sessions/${id}/context`, { method: 'DELETE' });
export const listSSHConversationGrants = (id) => request(`/api/sessions/${id}/ssh-grants`);
export const revokeSSHConversationGrant = (sessionId, hostId) => request(`/api/sessions/${sessionId}/ssh-grants/${encodeURIComponent(hostId)}`, { method: 'DELETE' });
export const clearSSHConversationGrants = (id) => request(`/api/sessions/${id}/ssh-grants`, { method: 'DELETE' });
