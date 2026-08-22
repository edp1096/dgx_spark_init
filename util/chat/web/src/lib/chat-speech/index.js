import { normalizeEnglishChatSpeech } from './english.js';
import { normalizeKoreanChatSpeech } from './korean.js';

const NORMALIZERS = [
  normalizeKoreanChatSpeech,
  normalizeEnglishChatSpeech,
];

export function normalizeChatSpeech(text) {
  return NORMALIZERS.reduce((value, normalize) => normalize(value), String(text || ''));
}

export { normalizeEnglishChatSpeech, normalizeKoreanChatSpeech };
