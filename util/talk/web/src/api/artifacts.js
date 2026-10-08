import { request } from './request.js';
const url = (session, suffix = '') => `/api/artifacts${suffix}?session_id=${encodeURIComponent(session)}`;
export const loadCodeProjects = session => request(url(session), { method: 'POST' });
export const readCodeProject = (session, id, version) => request(url(session, `/${encodeURIComponent(id)}`) + (version ? `&version=${version}` : ''));
export const codeProjectVersions = (session, id) => request(url(session, `/${encodeURIComponent(id)}/versions`));
export const restoreCodeProject = (session, id, base, version) => request(url(session, `/${encodeURIComponent(id)}/restore`), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ base_version: base, version }) });
