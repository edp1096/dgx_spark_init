import { pause, permitted } from "./browser-driver.js";
async function frames(driver, id) {
  await driver.tab(id);
  const result = [];
  for (const f of await chrome.webNavigation.getAllFrames({ tabId: id })) {
    if (!(await permitted(f.url))) continue;
    try {
      await chrome.scripting.executeScript({
        target: { tabId: id, frameIds: [f.frameId] },
        files: ["dom-core.js", "browser-content.js"],
      });
      result.push(f);
    } catch (e) {
      if (f.frameId === 0) throw e;
    }
  }
  return result;
}
async function send(driver, tabId, frameId, message) {
  await driver.tab(tabId);
  driver.check();
  const result = await chrome.tabs.sendMessage(
    tabId,
    { type: "TALK_BROWSER_V14", session_id: driver.session.id, ...message },
    { frameId },
  );
  if (!result?.ok)
    throw Error(result?.error || "페이지 응답을 확인하지 못했습니다.");
  return result;
}
async function clean(driver, id) {
  const fs = await frames(driver, id);
  if (!fs.length) throw Error("페이지 상태를 확인하지 못했습니다.");
  // Unknown/inaccessible child frames can hold unsaved edits too.
  if (
    fs.length !==
    (await chrome.webNavigation.getAllFrames({ tabId: id })).length
  )
    throw Error(
      "접근할 수 없는 프레임이 있어 변경 여부를 확인하지 못했습니다.",
    );
  for (const f of fs)
    if ((await send(driver, id, f.frameId, { action: "state" })).dirty)
      throw Error("입력된 변경 사항이 있어 탭을 닫거나 이동하지 않았습니다.");
}
export async function browserTool(cmd, driver) {
  const a = cmd.args || {},
    session = driver.session;
  if (a.action === "tabs") {
    const tabs = [];
    for (const t of await chrome.tabs.query({}))
      if (await permitted(t.url))
        tabs.push({
          tab_id: t.id,
          window_id: t.windowId,
          title: t.title,
          url: t.url,
          attached: session.owns(t.id),
          opener_tab_id: t.openerTabId,
        });
    return { ok: true, tabs };
  }
  if (a.action === "stop") {
    // Release borrowed tabs, leave all windows and unsaved content intact.
    return { ok: true, status: "stopped", released_tabs: await session.stop() };
  }
  if (a.action === "release") {
    await session.release(a.tab_id);
    return { ok: true, status: "released", tab_id: a.tab_id };
  }
  if (a.action === "open") {
    if (!(await permitted(a.url)))
      throw Error("허용된 http/https 사이트 주소를 지정하세요.");
    driver.mark();
    const tab = await chrome.tabs.create({ url: a.url, active: false });
    await session.claim(tab.id, true);
    return { ok: true, status: "opened", tab_id: tab.id };
  }
  if (a.action === "attach") {
    const tab = await driver.tab(a.tab_id);
    return { ok: true, status: "attached", tab_id: tab.id, url: tab.url };
  }
  if (!session.owns(a.tab_id))
    throw Error("먼저 attach로 이 작업에 사용할 탭을 지정하세요.");
  await driver.tab(a.tab_id);
  if (a.action === "observe") {
    const result = [],
      refs = new Map();
    for (const f of await frames(driver, a.tab_id)) {
      const view = await send(driver, a.tab_id, f.frameId, {
        action: "observe",
      });
      for (const el of view.elements)
        refs.set(el.ref, {
          frame_id: f.frameId,
          document_id: view.document_id,
        });
      result.push({ ...view, frame_id: f.frameId });
    }
    session.refs.set(a.tab_id, refs);
    return { ok: true, tab_id: a.tab_id, frames: result };
  }
  if (["reload", "close", "navigate"].includes(a.action)) {
    await clean(driver, a.tab_id);
    session.refs.delete(a.tab_id);
    if (a.action === "reload") return driver.reload(a.tab_id);
    if (a.action === "close") {
      await driver.close(a.tab_id);
      return { ok: true, status: "closed", tab_id: a.tab_id };
    }
    if (!(await permitted(a.url)))
      throw Error("허용된 사이트 주소를 지정하세요.");
    driver.mark();
    await chrome.tabs.update(a.tab_id, { url: a.url });
    return { ok: true, status: "navigation_requested", tab_id: a.tab_id };
  }
  if (a.action === "screenshot")
    return driver.debug(a.tab_id, async (target) => {
      const shot = await chrome.debugger.sendCommand(
        target,
        "Page.captureScreenshot",
        { format: "jpeg", quality: 65, captureBeyondViewport: false },
      );
      return {
        ok: true,
        tab_id: a.tab_id,
        image_url: "data:image/jpeg;base64," + shot.data,
      };
    });
  if (a.action === "scroll") {
    if (!Number.isFinite(a.dy) || Math.abs(a.dy) > 5000)
      throw Error("dy는 -5000..5000 범위 숫자여야 합니다.");
    await frames(driver, a.tab_id);
    driver.mark();
    return send(driver, a.tab_id, 0, { action: "scroll", dy: a.dy });
  }
  if (a.action === "wait") {
    if (typeof a.text !== "string" || !a.text || a.text.length > 1000)
      throw Error("기다릴 텍스트가 필요합니다.");
    const ms = a.timeout_ms ?? 5000;
    if (!Number.isInteger(ms) || ms < 1 || ms > 15000)
      throw Error("대기 시간은 1..15000ms입니다.");
    const end = Date.now() + ms;
    do {
      driver.check();
      for (const f of await frames(driver, a.tab_id))
        if (
          (
            await send(driver, a.tab_id, f.frameId, { action: "text" })
          ).text.includes(a.text)
        )
          return { ok: true, status: "matched" };
      await pause(150);
    } while (Date.now() < end);
    return {
      ok: false,
      status: "timeout",
      error: "지정한 텍스트를 관찰하지 못했습니다.",
    };
  }
  const ref = session.refs.get(a.tab_id)?.get(a.ref);
  if (!ref) throw Error("요소 참조가 없습니다. observe로 다시 관찰하세요.");
  const message = { ...ref, ref: a.ref };
  const target = { tab_id: a.tab_id, frame_id: ref.frame_id };
  const call = (action, extra = {}) =>
    send(driver, a.tab_id, ref.frame_id, { ...message, action, ...extra });
  if (a.action === "click")
    return driver.clickPrepared(
      target,
      () => call("prepare"),
      () => call("verify"),
    );
  if (a.action === "fill") {
    await driver.insertPrepared(a.tab_id, a.text, () => call("prepare_text"));
    await call("verify_text", { text: a.text });
    return { ok: true, status: "filled" };
  }
  if (a.action === "select") {
    driver.mark();
    await call("select", { value: a.value });
    return { ok: true, status: "selected" };
  }
  if (a.action === "press") {
    const keys = {
      Enter: 13,
      Tab: 9,
      Escape: 27,
      ArrowDown: 40,
      ArrowUp: 38,
      ArrowLeft: 37,
      ArrowRight: 39,
      Backspace: 8,
      Delete: 46,
      Space: 32,
    };
    if (!Object.hasOwn(keys, a.key)) throw Error("지원하지 않는 키");
    await call("focus");
    return driver.debug(a.tab_id, async (t) => {
      await call("verify");
      driver.mark();
      try {
        await chrome.debugger.sendCommand(t, "Input.dispatchKeyEvent", {
          type: "keyDown",
          key: a.key === "Space" ? " " : a.key,
          code: a.key,
          windowsVirtualKeyCode: keys[a.key],
        });
      } finally {
        await chrome.debugger
          .sendCommand(t, "Input.dispatchKeyEvent", {
            type: "keyUp",
            key: a.key === "Space" ? " " : a.key,
            code: a.key,
            windowsVirtualKeyCode: keys[a.key],
          })
          .catch(() => {});
      }
      return { ok: true, status: "key_dispatched" };
    });
  }
  throw Error("지원하지 않는 브라우저 작업");
}
