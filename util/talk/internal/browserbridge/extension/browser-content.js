(() => {
  if (globalThis.__talkBrowserCore) return;
  globalThis.__talkBrowserCore = true;
  const documentID = crypto.randomUUID(),
    observations = new Map();
  const visible = (el) => {
    const r = el.getBoundingClientRect(),
      s = getComputedStyle(el);
    return (
      r.width > 0 &&
      r.height > 0 &&
      s.visibility !== "hidden" &&
      s.display !== "none"
    );
  };
  const name = (el) =>
    String(
      el.getAttribute("aria-label") ||
        el.labels?.[0]?.innerText ||
        el.innerText ||
        el.getAttribute("placeholder") ||
        el.getAttribute("title") ||
        "",
    )
      .trim()
      .slice(0, 240);
  const fingerprint = (el) =>
    JSON.stringify([
      el.tagName,
      el.getAttribute("role"),
      el.getAttribute("type"),
      el.getAttribute("href"),
      name(el),
    ]);
  const editable = (el) =>
    el.matches(
      "textarea,input:not([type=button]):not([type=submit]):not([type=checkbox]):not([type=radio]):not([type=file]):not([type=hidden])",
    ) || el.isContentEditable;
  function resolve(m) {
    if (m.document_id !== documentID)
      throw Error("문서가 변경됐습니다. observe로 다시 관찰하세요.");
    const entry = observations.get(m.session_id)?.get(m.ref);
    if (
      !entry ||
      !entry.el.isConnected ||
      entry.signature !== fingerprint(entry.el)
    )
      throw Error(
        "요소가 변경되었거나 만료됐습니다. observe로 다시 관찰하세요.",
      );
    const el = entry.el;
    if (
      !visible(el) ||
      el.matches(":disabled") ||
      el.getAttribute("aria-disabled") === "true"
    )
      throw Error("요소가 숨겨졌거나 비활성화됐습니다.");
    return el;
  }
  const point = (el) => globalThis.__talkDOM.point(el);
  function dirty() {
    return [
      ...document.querySelectorAll(
        "input,textarea,select,[contenteditable=true]",
      ),
    ].some((el) => {
      if (el.isContentEditable) return Boolean(el.textContent.trim());
      if (el.tagName === "SELECT")
        return el.selectedIndex !== el.cloneNode(true).selectedIndex;
      if (el.type === "checkbox" || el.type === "radio")
        return el.checked !== el.defaultChecked;
      return el.value !== el.defaultValue;
    });
  }
  chrome.runtime.onMessage.addListener((m, s, reply) => {
    if (s.id !== chrome.runtime.id || m.type !== "TALK_BROWSER_V14") return;
    try {
      if (m.action === "observe") {
        const entries = new Map(),
          elements = [];
        for (const el of document.querySelectorAll(
          "a[href],button,input,textarea,select,[role=button],[role=link],[role=checkbox],[role=radio],[contenteditable=true],[tabindex]",
        )) {
          if (!visible(el) || elements.length >= 300) continue;
          const ref = crypto.randomUUID();
          entries.set(ref, { el, signature: fingerprint(el) });
          elements.push({
            ref,
            tag: el.tagName.toLowerCase(),
            role: el.getAttribute("role"),
            name: name(el),
            input_type: el.getAttribute("type"),
            disabled:
              el.matches(":disabled") ||
              el.getAttribute("aria-disabled") === "true",
            ...(el.tagName === "SELECT"
              ? {
                  options: [...el.options]
                    .slice(0, 100)
                    .map((o) => ({
                      value: o.value,
                      label: o.label,
                      selected: o.selected,
                    })),
                }
              : {}),
          });
        }
        observations.set(m.session_id, entries);
        if (observations.size > 50)
          observations.delete(observations.keys().next().value);
        reply({
          ok: true,
          document_id: documentID,
          url: location.href,
          title: document.title,
          text: document.body?.innerText.slice(0, 16000) || "",
          elements,
          truncated: elements.length === 300,
        });
        return;
      }
      if (m.action === "state") {
        reply({
          ok: true,
          dirty: dirty(),
          document_id: documentID,
          url: location.href,
        });
        return;
      }
      if (m.action === "text") {
        reply({
          ok: true,
          text: document.body?.innerText.slice(0, 64000) || "",
        });
        return;
      }
      if (m.action === "scroll") {
        globalThis.__talkDOM.scroll(window, m.dy);
        reply({ ok: true, x: scrollX, y: scrollY });
        return;
      }
      const el = resolve(m);
      if (m.action === "prepare") {
        reply({ ok: true, point: point(el), label: name(el) });
        return;
      }
      if (m.action === "verify") {
        point(el);
        reply({ ok: true });
        return;
      }
      if (m.action === "focus") {
        globalThis.__talkDOM.focus(el);
        reply({ ok: true });
        return;
      }
      if (m.action === "prepare_text") {
        if (!editable(el) || el.readOnly)
          throw Error("편집 가능한 입력칸이 아닙니다.");
        globalThis.__talkDOM.focus(el, true);
        reply({ ok: true });
        return;
      }
      if (m.action === "verify_text") {
        reply({
          ok: (el.isContentEditable ? el.textContent : el.value) === m.text,
        });
        return;
      }
      if (m.action === "select") {
        if (el.tagName !== "SELECT" || el.multiple)
          throw Error("단일 선택 상자가 아닙니다.");
        const option = [...el.options].find(
          (o) => o.value === m.value && !o.disabled,
        );
        if (!option) throw Error("선택 가능한 값이 아닙니다.");
        reply({ ok: globalThis.__talkDOM.select(el, m.value) });
        return;
      }
      throw Error("지원하지 않는 페이지 작업");
    } catch (e) {
      reply({ ok: false, error: e.message });
    }
    return false;
  });
})();
