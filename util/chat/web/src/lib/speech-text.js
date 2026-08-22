import { cleanMarkdownLine, sourceOnlyLine, speechTextFromMarkdown } from './speech-normalizer.js';

export { normalizeSpeechNotation, speechTextFromMarkdown } from './speech-normalizer.js';

function safeBoundary(source) {
  const newline = source.indexOf('\n');
  let sentence = -1;
  for (let index = 0; index < source.length - 1; index += 1) {
    const character = source[index];
    if (!/[.!?。！？…]/u.test(character) || !/\s/u.test(source[index + 1])) continue;
    if (character === '.' && /\d/u.test(source[index - 1] || '') && /\d/u.test(source[index + 1] || '')) continue;
    sentence = index + 1;
    break;
  }
  if (newline >= 0 && (sentence < 0 || newline < sentence)) return newline + 1;
  return sentence;
}

export function createSpeechChunker() {
  let buffer = '';
  let insideToolCall = false;

  function cleanFragment(fragment) {
    let value = fragment.trim();
    if (!value) return '';
    if (insideToolCall) {
      const end = value.search(/<\/tool_call>/iu);
      if (end < 0) return '';
      value = value.slice(end + '</tool_call>'.length).trim();
      insideToolCall = false;
    }
    const start = value.search(/<tool_call\b/iu);
    if (start >= 0) {
      const before = value.slice(0, start);
      const after = value.slice(start).match(/<\/tool_call>([\s\S]*)$/iu);
      insideToolCall = !after;
      value = `${before} ${after?.[1] || ''}`.trim();
    }
    if (!value || sourceOnlyLine(value)) return '';
    return cleanMarkdownLine(value);
  }

  return {
    push(delta) {
      buffer += String(delta || '');
      const chunks = [];
      let boundary = safeBoundary(buffer);
      while (boundary > 0) {
        const cleaned = cleanFragment(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary);
        if (cleaned) chunks.push(cleaned);
        boundary = safeBoundary(buffer);
      }
      return chunks;
    },
    finish() {
      const cleaned = speechTextFromMarkdown(buffer);
      buffer = '';
      return cleaned ? cleaned.split('\n').filter(Boolean) : [];
    },
  };
}

export function createSpeechBatcher({ maxChunks = 3, targetCharacters = 140 } = {}) {
  let pending = [];
  let characters = 0;

  function flush() {
    if (!pending.length) return [];
    const batch = pending.join('\n');
    pending = [];
    characters = 0;
    return [batch];
  }

  return {
    push(chunks) {
      const batches = [];
      for (const chunk of Array.isArray(chunks) ? chunks : [chunks]) {
        const value = String(chunk || '').trim();
        if (!value) continue;
        pending.push(value);
        characters += value.length;
        if (pending.length >= maxChunks || characters >= targetCharacters) batches.push(...flush());
      }
      return batches;
    },
    finish: flush,
  };
}
