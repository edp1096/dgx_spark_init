// Shared Chrome execution primitives. No site-specific parsing or policy here.
export const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
export async function permitted(url) {
  try {
    const u = new URL(url);
    return (
      ["http:", "https:"].includes(u.protocol) &&
      !u.username &&
      !u.password &&
      (await chrome.permissions.contains({ origins: [u.origin + "/*"] }))
    );
  } catch {
    return false;
  }
}
export class BrowserDriver {
  constructor(session, check) {
    this.session = session;
    this.check = check;
    this.effect = "none";
  }
  mark() {
    this.check();
    this.effect = "unknown";
  }
  async tab(id) {
    this.check();
    if (!Number.isInteger(id) || id <= 0)
      throw Error("유효한 tab_id가 필요합니다.");
    const tab = await chrome.tabs.get(id);
    if (!(await permitted(tab.url)))
      throw Error(
        "이 사이트 접근 권한이 없습니다. 확장 연결 화면에서 사이트를 허용하세요.",
      );
    await this.session.claim(id);
    this.check();
    return tab;
  }
  async debug(tabId, fn, alreadyAttached = false) {
    await this.tab(tabId);
    const target = { tabId };
    let attached = false;
    try {
      if (!alreadyAttached) {
        await chrome.debugger.attach(target, "1.3");
        attached = true;
      }
      this.check();
      return await fn(target);
    } finally {
      if (attached) await chrome.debugger.detach(target).catch(() => {});
    }
  }
  async point(tabId, frameId, point) {
    const all = await chrome.webNavigation.getAllFrames({ tabId });
    let frame = all.find((f) => f.frameId === frameId);
    if (!frame) throw Error("대상 프레임이 변경됐습니다. 다시 관찰하세요.");
    while (frame.parentFrameId >= 0) {
      const results = await chrome.scripting.executeScript({
        target: { tabId, frameIds: [frame.parentFrameId] },
        func: (url, p) => {
          const els = [...document.querySelectorAll("iframe,frame")].filter(
            (el) => el.src === url,
          );
          if (els.length !== 1)
            throw Error("프레임 위치를 하나로 확인할 수 없습니다.");
          const el = els[0],
            r = el.getBoundingClientRect(),
            sx = r.width / el.offsetWidth,
            sy = r.height / el.offsetHeight;
          // Rotated/skewed frames need a quad transform; never guess their coordinates.
          const tr = getComputedStyle(el).transform;
          if (tr !== "none") {
            const m = new DOMMatrix(tr);
            if (m.b || m.c) throw Error("회전된 프레임은 지원하지 않습니다.");
          }
          const result = {
            x: r.left + (el.clientLeft + p.x) * sx,
            y: r.top + (el.clientTop + p.y) * sy,
          };
          if (document.elementFromPoint(result.x, result.y) !== el)
            throw Error("프레임이 다른 요소에 가려졌습니다.");
          return result;
        },
        args: [frame.url, point],
      });
      point = results[0]?.result;
      if (!point) throw Error("프레임 좌표 확인 실패");
      frame = all.find((f) => f.frameId === frame.parentFrameId);
      if (!frame) throw Error("상위 프레임 확인 실패");
    }
    return point;
  }
  async clickPrepared(
    t,
    prepare,
    verify = async () => {},
    report = async () => ({}),
  ) {
    let pressed = false,
      held = false;
    try {
      return await this.debug(
        t.tab_id,
        async (target) => {
          await chrome.debugger.sendCommand(target, "Page.enable");
          const prepared = await prepare();
          if (!prepared?.ok)
            throw Error(prepared?.error || "클릭 대상을 확인하지 못했습니다.");
          const point = await this.point(t.tab_id, t.frame_id, prepared.point);
          await verify(prepared);
          this.check();
          try {
            for (const type of [
              "mouseMoved",
              "mousePressed",
              "mouseReleased",
            ]) {
              this.check();
              if (type === "mousePressed") {
                this.mark();
                pressed = true;
                held = true;
              }
              await chrome.debugger.sendCommand(
                target,
                "Input.dispatchMouseEvent",
                {
                  type,
                  x: point.x,
                  y: point.y,
                  button: type === "mouseMoved" ? "none" : "left",
                  buttons: type === "mousePressed" ? 1 : 0,
                  clickCount: type === "mouseMoved" ? 0 : 1,
                },
              );
              if (type === "mouseReleased") held = false;
            }
          } finally {
            // Release a held pointer even after cancellation; a dispatched press cannot be undone.
            if (held)
              await chrome.debugger
                .sendCommand(target, "Input.dispatchMouseEvent", {
                  type: "mouseReleased",
                  x: point.x,
                  y: point.y,
                  button: "left",
                  buttons: 0,
                  clickCount: 1,
                })
                .catch(() => {});
          }
          const observed = await report(prepared);
          return {
            ok: true,
            click_dispatched: true,
            method: "chrome_mouse_input",
            label: prepared.label,
            trusted_event: observed.trusted_event ?? null,
            user_activation: observed.user_activation ?? null,
            receipt: prepared.receipt,
          };
        },
        t.debuggerAttached,
      );
    } catch (error) {
      error.pointerPressed = pressed;
      throw error;
    }
  }
  async insertPrepared(tabId, text, prepare) {
    if (typeof text !== "string" || text.length > 100000)
      throw Error("입력 문자열은 100000자 이하여야 합니다.");
    return this.debug(tabId, async (target) => {
      await prepare();
      this.mark();
      await chrome.debugger.sendCommand(target, "Input.insertText", { text });
    });
  }
  async reload(id) {
    const before = await this.tab(id);
    let complete = false;
    const listener = (tabId, change) => {
      if (tabId === id && change.status === "complete") complete = true;
    };
    chrome.tabs.onUpdated.addListener(listener);
    try {
      this.mark();
      await chrome.tabs.reload(id);
      for (let i = 0; i < 75 && !complete; i++) {
        await pause(200);
        this.check();
      }
      if (!complete)
        return {
          ok: false,
          status: "refresh_unconfirmed",
          effect_state: "unknown",
          tab_id: id,
          error: "새로고침 완료를 확인하지 못했습니다. 다시 관찰하세요.",
        };
      const tab = await this.tab(id);
      return {
        ok: true,
        status: "refreshed",
        tab_id: id,
        url: tab.url,
        title: tab.title,
      };
    } finally {
      chrome.tabs.onUpdated.removeListener(listener);
    }
  }
  async close(id) {
    await this.tab(id);
    this.mark();
    await chrome.tabs.remove(id);
    await this.session.release(id);
  }
}
