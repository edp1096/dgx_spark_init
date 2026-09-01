import { normalizeChatSpeech } from './chat-speech/index.js';

export function sourceOnlyLine(line) {
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

function normalizeNumericSlashLists(text) {
  const number = '-?\\d+(?:[.,]\\d+)?';
  const series = new RegExp(`(${number}(?:\\s+\\/\\s+${number}){2,})`, 'gu');
  return String(text || '').replace(series, (match, _values, offset, source) => {
    // An equals sign makes this an arithmetic expression (for example,
    // "12 / 3 / 2 = 2"). Without one, three or more spaced numeric values
    // are normally a status sequence such as a system load average.
    if (source.slice(offset + match.length).trimStart().startsWith('=')) return match;
    return match.replace(/\s+\/\s+/gu, ', ');
  });
}

function normalizeSimpleExpressions(text) {
  const number = '-?\\d+(?:[.,]\\d+)?';
  const expression = new RegExp(
    `(${number}(?:\\s+(?:더하기|빼기|곱하기|나누기)\\s+(${number}))+)\\s*=\\s*`,
    'gu',
  );
  return String(text || '')
    .replace(new RegExp(`(${number})\\s*\\+\\s*(?=${number})`, 'gu'), '$1 더하기 ')
    .replace(new RegExp(`(${number})\\s*[-−]\\s*(?=${number})`, 'gu'), '$1 빼기 ')
    .replace(new RegExp(`(${number})\\s*[×✕*]\\s*(?=${number})`, 'gu'), '$1 곱하기 ')
    .replace(new RegExp(`(${number})\\s*[÷/]\\s*(?=${number})`, 'gu'), '$1 나누기 ')
    .replace(expression, (_match, value, lastNumber) => {
      const finalDigit = String(lastNumber).match(/\d(?=\D*$)/u)?.[0] || '';
      const topicParticle = /[013678]/u.test(finalDigit) ? '은' : '는';
      return `${value}${topicParticle} `;
    });
}

function removeVisualSymbols(text) {
  const protectedValues = [];
  const protectedText = String(text || '').replace(
    /https?:\/\/\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b/giu,
    (value) => {
      const token = `ZZZSPEECHPROTECTED${protectedValues.length}ZZZ`;
      protectedValues.push(value);
      return token;
    },
  );
  const cleaned = protectedText
    // 화살표·수학·통화·장식 기호는 화면용 표식이다. 이름을 읽지 않고
    // 문장끼리 붙지 않도록 짧은 쉼만 남긴다.
    .replace(/[\p{Sm}\p{Sc}\p{Sk}\p{So}](?:[\uFE0E\uFE0F])?/gu, ', ')
    .replace(/[#%‰‱&@|/\\]+/gu, ', ')
    .replace(/(?:,\s*){2,}/gu, ', ')
    .replace(/\s+,/gu, ',')
    .replace(/^\s*,\s*|\s*,\s*$/gu, '')
    .replace(/[ \t]{2,}/gu, ' ');
  return protectedValues.reduce(
    (value, protectedValue, index) => value.replace(`ZZZSPEECHPROTECTED${index}ZZZ`, protectedValue),
    cleaned,
  );
}

export function normalizeSpeechNotation(text) {
  const visualEmojiRemoved = String(text || '')
    .replace(/[\u{1F1E6}-\u{1F1FF}]{2}/gu, '')
    .replace(/\p{Extended_Pictographic}(?:[\uFE0E\uFE0F])?(?:\u200D\p{Extended_Pictographic}(?:[\uFE0E\uFE0F])?)*/gu, '')
    .replace(/[ㅠㅜ](?:[\s._-]*[ㅠㅜ])+/gu, '')
    .replace(/ㅡ(?:[\s._-]*ㅡ)+/gu, '');

  const semanticNotation = normalizeSimpleExpressions(normalizeNumericSlashLists(normalizeChatSpeech(visualEmojiRemoved)))
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
    .replace(/\t+/gu, ', ');

  return removeVisualSymbols(semanticNotation)
    .replace(/[ \t]{2,}/gu, ' ')
    .trim();
}

function stripListMarker(line) {
  return line
    .replace(/^(?:[-*+]|[•◦▪▫‣⁃●○◆◇■□▶▷])\s*(?:\[[ xX✓✔]\]\s*)?/u, '')
    .replace(/^(?:\(\d{1,3}\)|\d{1,3}[.)]|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳])\s*/u, '');
}

function removeParentheticalAsides(value) {
  let previous = '';
  while (value !== previous) {
    previous = value;
    value = value.replace(/\([^()]*\)|（[^（）]*）/gu, '');
  }
  return value;
}

export function cleanMarkdownLine(line, { omitParentheticals = false } = {}) {
  let value = line.trim();
  if (!value || /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/u.test(value)) return '';
  value = value.replace(/^#{1,6}\s+/u, '').replace(/^>\s?/u, '').trim();
  value = stripListMarker(value)
    .replace(/^[-*+]\s+/u, '')
    .replace(/!\[[^\]]*\]\([^)]+\)/gu, '')
    .replace(/\[([^\]]+)\]\([^)]+\)/gu, '$1')
    .replace(/https?:\/\/\S+/giu, '')
    .replace(/<[^>]+>/gu, '')
    .replace(/~~([^~]+)~~/gu, '$1')
    .replace(/[*_`]+/gu, '')
    .trim();
  if (omitParentheticals) value = removeParentheticalAsides(value).trim();
  if (!value) return '';
  if (value.startsWith('|') && value.endsWith('|')) {
    value = value.slice(1, -1).split('|').map((cell) => cell.trim()).filter(Boolean).join(', ');
  }
  value = normalizeSpeechNotation(value);
  if (!value) return '';
  if (!/[.!?。！？…:;]$/u.test(value)) value += '.';
  return value;
}

export function speechTextFromMarkdown(markdown, options = {}) {
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
  return lines.map((line) => cleanMarkdownLine(line, options)).filter(Boolean).join('\n');
}
