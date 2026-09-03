import { request } from './request.js';

const knowledgeCollectionPayload = (collection) => {
  const payload = {
    name: String(collection?.name || ''),
    description: String(collection?.description || ''),
  };
  if (typeof collection?.enabled === 'boolean') payload.enabled = collection.enabled;
  return payload;
};

export const listKnowledgeCollections = () => request('/api/knowledge/collections');
export const createKnowledgeCollection = (collection) => request('/api/knowledge/collections', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(knowledgeCollectionPayload(collection)),
});
export const updateKnowledgeCollection = (id, collection) => request(`/api/knowledge/collections/${id}`, {
  method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(knowledgeCollectionPayload(collection)),
});
export const deleteKnowledgeCollection = (id) => request(`/api/knowledge/collections/${id}`, { method: 'DELETE' });
export const listKnowledgeDocuments = (collectionID) => request(`/api/knowledge/documents?collection_id=${encodeURIComponent(collectionID)}`);
export const uploadKnowledgeDocument = (collectionID, file, title = '') => {
  const body = new FormData();
  body.append('collection_id', String(collectionID));
  body.append('title', title);
  body.append('file', file);
  return request('/api/knowledge/documents', { method: 'POST', body });
};
export const collectKnowledgeSource = (collectionID, url, mode = 'auto') => request('/api/knowledge/sources', {
  method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ collection_id: Number(collectionID), url, mode }),
});
export const listKnowledgeJobs = () => request('/api/knowledge/jobs');
export const updateKnowledgeJob = (id, action) => request(`/api/knowledge/jobs/${encodeURIComponent(id)}/${encodeURIComponent(action)}`, { method: 'POST' });
export const deleteKnowledgeDocument = (id) => request(`/api/knowledge/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
export const ocrKnowledgeDocument = (id) => request(`/api/knowledge/documents/${encodeURIComponent(id)}/ocr`, { method: 'POST' });
export const searchKnowledge = (query, collectionID = 0, limit = 20) => request(`/api/knowledge/search?q=${encodeURIComponent(query)}&collection_id=${encodeURIComponent(collectionID)}&limit=${encodeURIComponent(limit)}`);
export const knowledgeSourceURL = (id) => `/api/knowledge/documents/${encodeURIComponent(id)}/source`;
