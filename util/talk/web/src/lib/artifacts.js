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

function projectDocumentFor(html, css, javascript) {
  let source = html.trim();
  if (!/^\s*<!doctype|^\s*<html[\s>]/i.test(source)) {
    source = `<!doctype html><html><head><meta charset="utf-8"></head><body>${source || '<main id="app"></main>'}</body></html>`;
  }
  let head = '<meta name="viewport" content="width=device-width,initial-scale=1">';
  if (css && !/<link\b[^>]*\bhref=["'](?:\.\/)?style\.css["']/i.test(source)) {
    head += '<link rel="stylesheet" href="./style.css">';
  }
  if (javascript && !/<script\b[^>]*\bsrc=["'](?:\.\/)?script\.js["']/i.test(source)) {
    source = /<\/body>/i.test(source)
      ? source.replace(/<\/body>/i, '<script src="./script.js" defer></script></body>')
      : `${source}<script src="./script.js" defer></script>`;
  }
  source = /<head[\s>]/i.test(source)
    ? source.replace(/<head([^>]*)>/i, `<head$1>${head}`)
    : source.replace(/<html([^>]*)>/i, `<html$1><head>${head}</head>`);
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
    const projectHTML = projectDocumentFor(item.source, styles, scripts);
    const files = [{ name: 'index.html', language: 'html', source: projectHTML }];
    if (styles) files.push({ name: 'style.css', language: 'css', source: styles });
    if (scripts) files.push({ name: 'script.js', language: 'javascript', source: scripts });
    return {
      id: `${messageKey}:${variant}:${index}`,
      messageId: message.id,
      title: titleFromHTML(item.source, fallback),
      html: item.source,
      css: styles,
      javascript: scripts,
      document: documentFor(item.source, styles, scripts),
      files,
    };
  });
}

export function artifactsFromMessages(messages = []) {
  return messages.flatMap((message, index) => artifactsFromMessage(message, index));
}

// Persistent projects keep their identity across messages and revisions.
export function artifactFromProject(project) {
  const files = (project.files || []).map(file => ({ ...file, language: file.name.split('.').pop() }));
  const entry = files.find(file => file.name === 'index.html') || files.find(file => /\.(html?|svg)$/i.test(file.name));
  let html = entry ? (entry.source.trim() ? entry.source : '<main>저장된 웹 파일이 비어 있습니다. 버전 기록에서 정상 버전을 확인하거나 코드를 다시 저장해 주세요.</main>') : '<main>실행할 HTML 파일이 없습니다. 소스 보기에서 파일을 확인해 주세요.</main>';
  const used = new Set();
  const deferred = [];
  html = html.replace(/<script\b([^>]*?)\bsrc=["'](?:\.\/)?([^"']+)["']([^>]*)>\s*<\/script>/gi, (tag, before, name, after) => {
    const file = files.find(file => file.name === name);
    if (!file) return tag;
    used.add(name);
    const script = `<script${before}${after}>${escapeScriptEnd(file.source)}</script>`;
    // Inline classic scripts ignore defer. Run these after the body exists,
    // preserving the ordering of the referenced deferred files.
    if (/\bdefer\b/i.test(before + after) && !/\btype\s*=\s*["']module["']/i.test(before + after)) {
      deferred.push(script); return '';
    }
    return script;
  });
  html = html.replace(/<link\b[^>]*\bhref=["'](?:\.\/)?([^"']+)["'][^>]*>/gi, (tag, name) => {
    const file = files.find(file => file.name === name && /\.css$/i.test(name));
    if (!file) return tag;
    used.add(name);
    return `<style>${file.source}</style>`;
  });
  if (deferred.length) html = /<\/body>/i.test(html) ? html.replace(/<\/body>/i, () => `${deferred.join('\n')}</body>`) : html + deferred.join('\n');
  const css = files.filter(file => /\.css$/i.test(file.name) && !used.has(file.name)).map(file => file.source).join('\n');
  const js = files.filter(file => /\.js$/i.test(file.name) && !used.has(file.name)).map(file => file.source).join('\n');
  return { ...project, persistent: true, files, document: documentFor(html, css, js) };
}
