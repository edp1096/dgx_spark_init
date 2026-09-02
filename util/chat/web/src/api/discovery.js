import { request } from './request.js';

export const searchConversations = (query, limit = 5) => request(`/api/search?q=${encodeURIComponent(query)}&limit=${limit}`);
export const searchConversationPage = (query, options = {}) => {
  const params = new URLSearchParams({ q: query, limit: String(options.limit || 20), sort: options.sort || 'relevance', scope: options.scope || 'all' });
  if (options.from) params.set('from', options.from);
  if (options.to) params.set('to', options.to);
  if (options.cursor) params.set('cursor', options.cursor);
  return request(`/api/search/page?${params}`);
};
export const listSkills = () => request('/api/skills');
export const listToolAudits = (limit = 30) => request(`/api/tool-audit?limit=${limit}`);
