const FENCE_RE = /```([^\n`]*)\n([\s\S]*?)```/g;

function languageOf(raw) {
  return (raw || '').trim().split(/\s+/)[0].replace(/[{}\.]/g, '').toLowerCase();
}

function escapeScriptEnd(source) {
  return source.replace(/<\/script/gi, '<\\/script');
}

function titleFromHTML(html, fallback) {
  return html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1]?.replace(/<[^>]+>/g, '').trim() || fallback;
}

function documentFor(html, css, javascript) {
  const policy = "default-src 'none'; img-src data: blob: https:; media-src data: blob: https:; font-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; frame-src 'none'; base-uri 'none'; form-action 'none'";
  let source = html.trim();
  if (!/^\s*<!doctype|^\s*<html[\s>]/i.test(source)) {
    source = `<!doctype html><html><head><meta charset="utf-8"></head><body>${source || '<main id="app"></main>'}</body></html>`;
  }
  const head = `<meta http-equiv="Content-Security-Policy" content="${policy}"><meta name="viewport" content="width=device-width,initial-scale=1">${css ? `<style>${css}</style>` : ''}`;
  const script = javascript ? `<script>${escapeScriptEnd(javascript)}<\/script>` : '';
  source = /<head[\s>]/i.test(source)
    ? source.replace(/<head([^>]*)>/i, `<head$1>${head}`)
    : source.replace(/<html([^>]*)>/i, `<html$1><head>${head}</head>`);
  source = /<\/body>/i.test(source) ? source.replace(/<\/body>/i, `${script}</body>`) : `${source}${script}`;
  return source;
}

export function artifactsFromMessage(message, messageIndex = 0) {
  if (message?.role !== 'assistant' || !message.content) return [];
  const blocks = [];
  for (const match of message.content.matchAll(FENCE_RE)) {
    const language = languageOf(match[1]);
    if (['html', 'htm', 'css', 'js', 'javascript', 'svg'].includes(language)) {
      blocks.push({ language, source: match[2].trim() });
    }
  }
  if (!blocks.length) return [];
  const styles = blocks.filter((item) => item.language === 'css').map((item) => item.source).join('\n\n');
  const scripts = blocks.filter((item) => ['js', 'javascript'].includes(item.language)).map((item) => item.source).join('\n\n');
  let documents = blocks.filter((item) => ['html', 'htm'].includes(item.language));
  if (!documents.length) {
    const svg = blocks.find((item) => item.language === 'svg');
    documents = [{ language: svg ? 'svg' : 'html', source: svg?.source || '<main id="app"></main>' }];
  }
  const messageKey = message.id || `pending-${messageIndex}`;
  const variant = message.variant_index ?? 0;
  return documents.map((item, index) => {
    const fallback = documents.length > 1 ? `웹 생성물 ${index + 1}` : '웹 생성물';
    return {
      id: `${messageKey}:${variant}:${index}`,
      messageId: message.id,
      title: titleFromHTML(item.source, fallback),
      html: item.source,
      css: styles,
      javascript: scripts,
      document: documentFor(item.source, styles, scripts),
    };
  });
}

export function artifactsFromMessages(messages = []) {
  return messages.flatMap((message, index) => artifactsFromMessage(message, index));
}
