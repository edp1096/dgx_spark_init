import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';

marked.use(markedKatex({ throwOnError: false, nonStandard: true }));

function escapeHTML(value) {
  return String(value || '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function codeLanguage(value) {
  const language = String(value || '').trim().split(/\s+/)[0].toLowerCase();
  return /^[a-z0-9_+#.-]{1,32}$/.test(language) ? language : 'code';
}

marked.use({
  renderer: {
    code(token) {
      const source = String(token?.text || '');
      const language = codeLanguage(token?.lang);
      const lineCount = source ? source.split('\n').length : 0;
      const isLong = lineCount > 12 || source.length > 900;
      return `<div class="code-card${isLong ? ' code-card-long' : ''}" data-code-card>`
        + `<div class="code-card-header"><span>${escapeHTML(language)}</span><div>`
        + '<button type="button" data-code-copy>복사</button>'
        + (isLong ? '<button type="button" data-code-toggle aria-expanded="false">전체 보기</button>' : '')
        + '</div></div>'
        + `<pre><code class="language-${escapeHTML(language)}">${escapeHTML(source)}</code></pre>`
        + '</div>';
    },
  },
});

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
