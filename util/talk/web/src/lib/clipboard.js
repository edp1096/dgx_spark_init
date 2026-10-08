// Copy on HTTPS and on local/LAN HTTP, where Clipboard API may be unavailable.
export async function copyText(text) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return;
    }
  } catch { /* Fall back to a selected textarea while handling the user click. */ }
  const focused = document.activeElement;
  const selection = window.getSelection();
  const ranges = Array.from({ length: selection?.rangeCount || 0 }, (_, index) => selection.getRangeAt(index).cloneRange());
  const field = document.createElement('textarea');
  field.value = text;
  field.readOnly = true;
  field.style.cssText = 'position:fixed;left:0;top:0;opacity:0;pointer-events:none';
  document.body.append(field);
  try {
    field.focus({ preventScroll: true });
    field.select();
    if (!document.execCommand('copy')) throw new Error('Clipboard copy was blocked');
  } finally {
    field.remove();
    focused?.focus?.({ preventScroll: true });
    if (selection) {
      selection.removeAllRanges();
      for (const range of ranges) selection.addRange(range);
    }
  }
}

// Accept the same sanitized HTML used by the reply renderer. Read the whole
// document, including folded code, without copying code-card controls or KaTeX's
// duplicate visual/accessibility representations.
export function replyPlainText(html) {
  const template = document.createElement('template');
  template.innerHTML = html;
  const code = [];
  const children = node => [...node.childNodes].map(walk).join('');
  function walk(node) {
    if (node.nodeType === Node.TEXT_NODE) return node.textContent.replace(/\s+/g, ' ');
    if (node.nodeType !== Node.ELEMENT_NODE) return children(node);
    if (node.matches('button, .code-card-header, .code-card-footer, script, style')) return '';
    if (node.matches('.katex')) {
      // DOMPurify may unwrap MathML's annotation, leaving the TeX as a direct
      // text child after the presentation tree.
      const math = node.querySelector('.katex-mathml math');
      const source = node.querySelector('annotation')?.textContent
        || [...(math?.childNodes || [])].filter(child => child.nodeType === Node.TEXT_NODE).map(child => child.textContent).join('').trim();
      return source || node.querySelector('.katex-html')?.textContent || node.textContent;
    }
    if (node.tagName === 'PRE') {
      const index = code.push(node.textContent) - 1;
      return `\n\n\u0000code${index}\u0000\n\n`;
    }
    if (node.tagName === 'BR') return '\n';
    if (node.tagName === 'IMG') return node.getAttribute('alt') || '';
    if (node.matches('input[type="checkbox"]')) return node.checked ? '☑ ' : '☐ ';
    if (node.matches('ol, ul')) {
      let number = Number(node.getAttribute('start') || 1);
      const items = [...node.children].filter(child => child.tagName === 'LI').map(item => {
        if (item.hasAttribute('value')) number = Number(item.getAttribute('value'));
        const marker = node.tagName === 'OL' ? `${number++}. ` : '• ';
        return marker + children(item).trim().replace(/\n{2,}/g, '\n').replaceAll('\n', '\n  ');
      });
      return '\n\n' + items.join('\n') + '\n\n';
    }
    if (node.tagName === 'TABLE') {
      return '\n\n' + [...node.rows].map(row => [...row.cells].map(cell => children(cell).trim().replace(/\n+/g, ' ')).join('\t')).join('\n') + '\n\n';
    }
    const text = children(node);
    return node.matches('p, div, section, blockquote, h1, h2, h3, h4, h5, h6, hr') ? `\n\n${text}\n\n` : text;
  }
  return children(template.content).replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim()
    .replace(/\u0000code(\d+)\u0000/g, (_, index) => code[Number(index)]);
}
