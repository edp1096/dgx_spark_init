import { request } from './request.js';

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
export async function streamSpeech(text, signal, onChunk) {
  const response = await fetch('/api/tts/speech', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal });
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
export const uploadMediaURL = (url, signal) => request('/api/media/source', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ url }), signal });
