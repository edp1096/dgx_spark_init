import { request } from './request.js';

export const listSSHHosts = () => request('/api/ssh/hosts');
export const createSSHHost = (host) => request('/api/ssh/hosts', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(host) });
export const updateSSHHost = (id, host) => request(`/api/ssh/hosts/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(host) });
export const deleteSSHHost = (id) => request(`/api/ssh/hosts/${id}`, { method: 'DELETE' });
export const listSSHKeys = () => request('/api/ssh/keys');
export const generateSSHKey = (keyId) => request('/api/ssh/keys/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key_id: keyId }) });
export const importSSHKey = (keyId, file, replace = false) => {
  const body = new FormData();
  body.append('key_id', keyId);
  body.append('key', file);
  if (replace) body.append('replace', 'true');
  return request('/api/ssh/keys', { method: 'POST', body });
};
export const deleteSSHKey = (keyId) => request(`/api/ssh/keys/${encodeURIComponent(keyId)}`, { method: 'DELETE' });
export const trustSSHHost = (id, publicKey) => request(`/api/ssh/hosts/${id}/trust`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ public_key: publicKey }) });
export const answerToolApproval = (id, decision) => request(`/api/tool-approvals/${id}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ decision }) });

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

export const getKeyStore = () => request('/api/ssh/key-store');
export const keyStoreAction = (body) => request('/api/ssh/key-store', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
