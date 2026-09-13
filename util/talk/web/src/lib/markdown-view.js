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
    const retained = new Map();
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
      const footer = fresh.querySelector('.code-card-footer');
      if (footer && !old.querySelector('.code-card-footer')) old.append(footer);
      retained.set(fresh, old);
    });
    // Keep retained cards connected throughout the update. Detaching them even
    // briefly cancels a pointer click when a chunk arrives between down and up.
    function reconcile(parent, freshParent) {
      let cursor = parent.firstChild;
      for (const fresh of [...freshParent.childNodes]) {
        const card = retained.get(fresh);
        if (card) {
          if (cursor !== card) parent.insertBefore(card, cursor);
          cursor = card.nextSibling;
          continue;
        }
        const sameKind = cursor && cursor.nodeType === fresh.nodeType
          && cursor.nodeName === fresh.nodeName && cursor.namespaceURI === fresh.namespaceURI
          && !cursor.matches?.('[data-code-card]');
        if (sameKind) {
          if (fresh.nodeType === Node.ELEMENT_NODE) {
            for (const attr of [...cursor.attributes]) {
              if (!fresh.hasAttribute(attr.name)) cursor.removeAttribute(attr.name);
            }
            for (const attr of fresh.attributes) {
              if (cursor.getAttribute(attr.name) !== attr.value) cursor.setAttribute(attr.name, attr.value);
            }
            reconcile(cursor, fresh);
          } else if (cursor.nodeValue !== fresh.nodeValue) {
            cursor.nodeValue = fresh.nodeValue;
          }
          cursor = cursor.nextSibling;
        } else {
          const added = fresh.cloneNode(false);
          parent.insertBefore(added, cursor);
          if (fresh.nodeType === Node.ELEMENT_NODE) reconcile(added, fresh);
        }
      }
      while (cursor) {
        const next = cursor.nextSibling;
        cursor.remove();
        cursor = next;
      }
    }
    reconcile(node, template.content);
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
