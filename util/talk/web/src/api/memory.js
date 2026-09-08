import { request } from './request.js';

export const listMemories = () => request('/api/memories');
export const createMemory = (memory) => request('/api/memories', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(memory),
});
export const updateMemory = (id, memory) => request(`/api/memories/${id}`, {
  method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(memory),
});
export const deleteMemory = (id) => request(`/api/memories/${id}`, { method: 'DELETE' });
