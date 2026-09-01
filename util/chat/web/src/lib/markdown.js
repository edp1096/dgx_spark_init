import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';

marked.use(markedKatex({ throwOnError: false, nonStandard: true }));

function repairStrongBeforeKoreanSuffix(value) {
  const repair = (text) => text.replace(/(\*\*[^*\n]+\*\*)(?=\p{Script=Hangul})/gu, '$1<!-- -->');
  const protectedCode = /```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*(?:`|$)/gu;
  let result = '';
  let cursor = 0;
  for (const match of value.matchAll(protectedCode)) {
    result += repair(value.slice(cursor, match.index));
    result += match[0];
    cursor = match.index + match[0].length;
  }
  return result + repair(value.slice(cursor));
}

export function normalizeMarkdown(value) {
  const source = String(value || '')
    .replace(/<\|(?:turn|start|end|channel)[^>]*>/gi, '')
    .replace(/<channel\|>/gi, '');
  const trimmed = source.trim();
  const envelope = trimmed.match(/^```(?:markdown|md)\s*\n([\s\S]*?)\n```$/i);
  return repairStrongBeforeKoreanSuffix(envelope ? envelope[1] : source);
}

export function parseMarkdown(value) {
  return marked.parse(normalizeMarkdown(value), { gfm: true, breaks: true });
}
