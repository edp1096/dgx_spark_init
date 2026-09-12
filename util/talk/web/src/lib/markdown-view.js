// Preserve interactive code cards while the surrounding Markdown streams.
// `html` must already be sanitized by the caller.
export function markdownView(node, html) {
  let previousHTML;
  function update(nextHTML) {
    if (nextHTML === previousHTML) return;
    previousHTML = nextHTML;
    const pane = node.closest('.messages');
    const panePosition = pane && { top: pane.scrollTop, left: pane.scrollLeft };
    const oldCards = [...node.querySelectorAll('[data-code-card]')].map(card => {
      const pre = card.querySelector('pre');
      return { card, pre, top: pre?.scrollTop || 0, left: pre?.scrollLeft || 0 };
    });
    const focused = node.contains(document.activeElement) ? document.activeElement : null;
    const template = document.createElement('template');
    template.innerHTML = nextHTML;
    const positions = [];
    [...template.content.querySelectorAll('[data-code-card]')].forEach((fresh, index) => {
      const saved = oldCards[index];
      const old = saved?.card;
      const oldCode = old?.querySelector('code');
      const nextCode = fresh.querySelector('code');
      const pre = saved?.pre;
      const buttons = old?.querySelector('.code-card-header > div');
      // Append-only updates belong to the same streamed block. Replacements
      // such as another answer variant get a fresh card instead.
      if (!pre || !buttons || !oldCode || !nextCode || oldCode.className !== nextCode.className
          || !nextCode.textContent.startsWith(oldCode.textContent)) return;
      positions.push({ pre, top: saved.top, left: saved.left });
      const suffix = nextCode.textContent.slice(oldCode.textContent.length);
      if (suffix) {
        if (oldCode.lastChild?.nodeType === Node.TEXT_NODE) oldCode.lastChild.appendData(suffix);
        else oldCode.append(document.createTextNode(suffix));
      }
      old.classList.toggle('code-card-long', fresh.classList.contains('code-card-long'));
      const toggle = fresh.querySelector('[data-code-toggle]');
      if (toggle && !old.querySelector('[data-code-toggle]')) {
        buttons.append(toggle);
      }
      fresh.replaceWith(old);
    });
    node.replaceChildren(template.content);
    for (const { pre, top, left } of positions) {
      pre.scrollTop = top;
      pre.scrollLeft = left;
    }
    // Moving a tall expanded card can temporarily shrink the message pane
    // and clamp its scroll offset. Restore it before the next browser frame.
    if (panePosition) {
      const behavior = pane.style.scrollBehavior;
      pane.style.scrollBehavior = 'auto';
      pane.scrollTop = panePosition.top;
      pane.scrollLeft = panePosition.left;
      pane.style.scrollBehavior = behavior;
    }
    if (focused && node.contains(focused) && document.activeElement !== focused) {
      focused.focus({ preventScroll: true });
    }
  }
  update(html);
  return { update };
}
