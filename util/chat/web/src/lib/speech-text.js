function sourceOnlyLine(line) {
  const value = line.trim();
  if (!value) return false;
  const withoutMarkdownLinks = value.replace(/\[[^\]]+\]\(https?:\/\/[^)]+\)/giu, '');
  const withoutURLs = withoutMarkdownLinks.replace(/https?:\/\/\S+/giu, '');
  const remainder = withoutURLs.replace(/[\s|·,;/\\—–-]+/gu, '');
  const hasLink = withoutMarkdownLinks !== value || withoutURLs !== withoutMarkdownLinks;
  return hasLink && remainder === '';
}

function compactDecimal(number) {
  return String(number).replace(/\.0+$/u, '');
}

const KOREAN_CHAT_SHORTHAND = new Map([
  ['ㅍㅎㅎ', '푸하하'],
  // ㄳ is also commonly entered as the single compound-final character.
  // Include its jongseong and halfwidth Unicode variants, plus leading jamo.
  ['ㄳ', '감사'],
  ['ᆪ', '감사'],
  ['ﾣ', '감사'],
  ['ᄀᄉ', '감사'],
  ['ㅇㅋ', '오키'],
  ['ㅇㅇ', '응응'],
  ['ㄴㄴ', '노노'],
  ['ㄱㄱ', '고고'],
  ['ㅂㅂ', '바이바이'],
  ['ㅂㅇ', '바이'],
  ['ㅎㅇ', '하이'],
  ['ㄱㅅ', '감사'],
  ['ㅈㅅ', '죄송'],
  ['ㅊㅋ', '축하'],
  ['ㅅㄱ', '수고'],
  ['ㄷㄷ', '덜덜'],
  ['ㅁㄹ', '몰라'],
  ['ㄱㅊ', '괜찮아'],
  ['ㄹㅇ', '리얼'],
  ['ㅇㅈ', '인정'],
  ['ㅇㄷ', '어디'],
  ['ㅉㅉ', '쯧쯧'],
]);

const KOREAN_CHAT_SHORTHAND_PATTERN = /ㅍㅎㅎ|ᄀᄉ|ㄳ|ᆪ|ﾣ|ㅇㅋ|ㅇㅇ|ㄴㄴ|ㄱㄱ|ㅂㅂ|ㅂㅇ|ㅎㅇ|ㄱㅅ|ㅈㅅ|ㅊㅋ|ㅅㄱ|ㄷㄷ|ㅁㄹ|ㄱㅊ|ㄹㅇ|ㅇㅈ|ㅇㄷ|ㅉㅉ/gu;

function expandKoreanChatShorthand(text) {
  return text.replace(KOREAN_CHAT_SHORTHAND_PATTERN, (value) => KOREAN_CHAT_SHORTHAND.get(value));
}

export function normalizeSpeechNotation(text) {
  const visualEmojiRemoved = String(text || '')
    // Visual emoji add no useful information to speech and some TTS models
    // pronounce their Unicode names or pause unpredictably.
    .replace(/[\u{1F1E6}-\u{1F1FF}]{2}/gu, '')
    .replace(/\p{Extended_Pictographic}(?:[\uFE0E\uFE0F])?(?:\u200D\p{Extended_Pictographic}(?:[\uFE0E\uFE0F])?)*/gu, '')
    // These are visual faces rather than pronounceable abbreviations.
    .replace(/[ㅠㅜ](?:[\s._-]*[ㅠㅜ])+/gu, '')
    .replace(/ㅡ(?:[\s._-]*ㅡ)+/gu, '');

  return expandKoreanChatShorthand(visualEmojiRemoved)
    // Korean chat commonly omits the vowel in laughter. Restore it before
    // sending the text to TTS so isolated consonants are pronounced naturally.
    .replace(/ㅎ{2,}/gu, (laughter) => laughter.replace(/ㅎ/gu, '흐'))
    .replace(/ㅋ{2,}/gu, (laughter) => laughter.replace(/ㅋ/gu, '크'))
    .replace(/(-?\d+(?:\.\d+)?)\s*m\s*\/\s*s\b/giu, (_match, number) => `초속 ${compactDecimal(number)}미터`)
    .replace(/(-?\d+(?:\.\d+)?)\s*km\s*\/\s*h\b/giu, (_match, number) => `시속 ${compactDecimal(number)}킬로미터`)
    .replace(/(-?\d+(?:\.\d+)?)\s*mm\s*\/\s*h\b/giu, (_match, number) => `시간당 ${compactDecimal(number)}밀리미터`)
    .replace(/(-?\d+(?:[.,]\d+)?)\s*[~～]\s*(-?\d+(?:[.,]\d+)?)\s*(?:°\s*F|℉)/giu, '화씨 $1도에서 $2도')
    .replace(/(-?\d+(?:[.,]\d+)?)\s*(?:°\s*F|℉)/giu, '화씨 $1도')
    .replace(/(-?\d+(?:[.,]\d+)?)\s*[~～]\s*(-?\d+(?:[.,]\d+)?)\s*(?:°\s*C|℃)/giu, '$1도에서 $2도')
    .replace(/(-?\d+(?:[.,]\d+)?)\s*(?:°\s*C|℃)/giu, '$1도')
    .replace(/(-?\d+(?:[.,]\d+)?)\s*[~～]\s*(-?\d+(?:[.,]\d+)?)\s*%/gu, '$1퍼센트에서 $2퍼센트')
    .replace(/(-?\d+(?:[.,]\d+)?)\s*%/gu, '$1퍼센트')
    .replace(/(-?\d+(?:[.,]\d+)?)\s*[~～]\s*(-?\d+(?:[.,]\d+)?)/gu, '$1에서 $2')
    .replace(/(?:°\s*C|℃)/giu, '섭씨')
    .replace(/(?:°\s*F|℉)/giu, '화씨')
    .replace(/[~～]+/gu, ', ')
    .replace(/\t+/gu, ', ')
    .replace(/[ \t]{2,}/gu, ' ')
    .trim();
}

function stripListMarker(line) {
  return line
    .replace(/^(?:[-*+]|[•◦▪▫‣⁃●○◆◇■□▶▷])\s*(?:\[[ xX✓✔]\]\s*)?/u, '')
    .replace(/^(?:\(\d{1,3}\)|\d{1,3}[.)]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*/u, '');
}

function cleanMarkdownLine(line) {
  let value = line.trim();
  if (!value || /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/u.test(value)) return '';
  value = value
    .replace(/^#{1,6}\s+/u, '')
    .replace(/^>\s?/u, '')
    .trim();
  value = stripListMarker(value)
    .replace(/^[-*+]\s+/u, '')
    .replace(/!\[[^\]]*\]\([^)]+\)/gu, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/gu, '$1')
    .replace(/https?:\/\/\S+/giu, '')
    .replace(/<[^>]+>/gu, '')
    .replace(/~~([^~]+)~~/gu, '$1')
    .replace(/[*_`]+/gu, '')
    .trim();
  if (!value) return '';

  if (value.startsWith('|') && value.endsWith('|')) {
    value = value.slice(1, -1).split('|').map((cell) => cell.trim()).filter(Boolean).join(', ');
  }
  value = normalizeSpeechNotation(value);
  if (!value) return '';
  // Markdown blocks are visually separated, but TTS receives one string. Add
  // an explicit sentence boundary so adjacent blocks are not spoken together.
  if (!/[.!?。！？…:;]$/u.test(value)) value += '.';
  return value;
}

export function speechTextFromMarkdown(markdown) {
  const source = String(markdown || '')
    .replace(/<tool_call\b[^>]*>[\s\S]*?<\/tool_call>/giu, '')
    .replace(/<tool_call\b[^>]*>[\s\S]*$/giu, '')
    .trim();
  if (!source) return '';

  const lines = source.split(/\r?\n/u);
  while (lines.length && !lines.at(-1).trim()) lines.pop();
  while (lines.length && sourceOnlyLine(lines.at(-1))) {
    lines.pop();
    while (lines.length && !lines.at(-1).trim()) lines.pop();
  }
  return lines.map(cleanMarkdownLine).filter(Boolean).join('\n');
}

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
