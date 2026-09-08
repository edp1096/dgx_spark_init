import { request } from './request.js';

export const getHealth = () => request('/api/health');
export const getModels = () => request('/api/models');
export const getConfig = () => request('/api/config');
export const saveConfig = (config) => request('/api/config', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(config) });
export const getMediaUsage = () => request('/api/media');
export const cleanupMedia = (keepIds = []) => request('/api/media', { method: 'DELETE', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ keep_ids: keepIds }) });
