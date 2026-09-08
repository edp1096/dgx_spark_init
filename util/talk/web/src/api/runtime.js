import { request } from './request.js';

export const getRuntime = () => request('/api/runtime');
export const startRuntimeBundle = (id) => request(`/api/runtime/bundles/${encodeURIComponent(id)}/start`, { method: 'POST' });
export const stopRuntimeBundle = (id) => request(`/api/runtime/bundles/${encodeURIComponent(id)}/stop`, { method: 'POST' });
export const controlRuntimeComponent = (id, action) => request(`/api/runtime/components/${encodeURIComponent(id)}/${encodeURIComponent(action)}`, { method: 'POST' });
