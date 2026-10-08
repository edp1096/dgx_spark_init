// Shared isolated-world DOM primitives, injected into each permitted frame.
(() => {
  if (globalThis.__talkDOM) return;
  globalThis.__talkDOM = {
    point(el) {
      if (
        !el.isConnected ||
        el.matches(":disabled") ||
        el.getAttribute("aria-disabled") === "true"
      )
        throw Error("요소가 변경됐거나 비활성화됐습니다.");
      el.scrollIntoView({
        block: "center",
        inline: "center",
        behavior: "instant",
      });
      const r = el.getBoundingClientRect(),
        left = Math.max(0, r.left),
        right = Math.min(innerWidth, r.right),
        top = Math.max(0, r.top),
        bottom = Math.min(innerHeight, r.bottom);
      if (right <= left || bottom <= top)
        throw Error("요소가 화면 안에 없습니다.");
      const x = (left + right) / 2,
        y = (top + bottom) / 2,
        hit = document.elementFromPoint(x, y);
      if (hit !== el && !el.contains(hit))
        throw Error("다른 요소가 대상을 가리고 있습니다.");
      return { x, y };
    },
    focus(el, selectText = false) {
      this.point(el);
      if (el.readOnly) throw Error("읽기 전용 입력칸입니다.");
      el.focus();
      if (document.activeElement !== el && !el.contains(document.activeElement))
        throw Error("입력칸 포커스를 확인하지 못했습니다.");
      if (selectText) {
        if (el.isContentEditable) {
          const r = document.createRange();
          r.selectNodeContents(el);
          const s = getSelection();
          s.removeAllRanges();
          s.addRange(r);
        } else el.select();
      }
    },
    select(el, value) {
      this.point(el);
      const option = [...el.options].find(
        (o) => o.value === value && !o.disabled,
      );
      if (!option) throw Error("선택할 수 없는 값입니다.");
      el.value = value;
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return el.value === value;
    },
    scroll(el, dy) {
      el.scrollBy({ top: dy, left: 0, behavior: "instant" });
    },
  };
})();
