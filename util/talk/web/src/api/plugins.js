import { request } from './request.js';
const path = id => `/api/plugins/${encodeURIComponent(id)}`;
export const listPlugins = () => request('/api/plugins');
export const pluginRuns = id => request(`${path(id)}/runs`);
export const pluginAction = (id, action, body = {}) => request(`${path(id)}/${action}`, {
  method: action === 'configure' ? 'PUT' : 'POST',
  headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

export const installPlugin = file => request('/api/plugins/install', {method:'POST',headers:{'Content-Type':'application/zip'},body:file});
