// Naver product/form semantics. Browser input is provided by the common driver.
const allowed = (url) => {
  try {
    const u = new URL(url);
    return (
      u.protocol === "https:" &&
      ["naver.com", "naverpay.com"].some(
        (h) => u.hostname === h || u.hostname.endsWith("." + h),
      )
    );
  } catch {
    return false;
  }
};
const pause = (ms) => new Promise((r) => setTimeout(r, ms));
const popupEvents = new Map();
async function notePopupTab(tab) {
  try {
    const w = await chrome.windows.get(tab.windowId);
    if (w.type === "popup")
      popupEvents.set(tab.id, {
        tabId: tab.id,
        windowId: tab.windowId,
        at: Date.now(),
        url: tab.url || tab.pendingUrl || "",
        openerTabId: tab.openerTabId,
      });
  } catch {}
}
chrome.tabs.onCreated.addListener((tab) => {
  notePopupTab(tab);
});
chrome.tabs.onUpdated.addListener((tabId, change, tab) => {
  if (change.url || change.status === "complete") notePopupTab(tab);
});
chrome.windows.onRemoved.addListener((windowId) => {
  for (const [id, v] of popupEvents)
    if (v.windowId === windowId) popupEvents.delete(id);
});

export function createNaverState() {
  return {
    openedForms: new Map(),
    targets: {},
    releaseTab(tabId) {
      for (const [key, target] of Object.entries(this.targets))
        if (target.tab_id === tabId) delete this.targets[key];
      for (const [key, form] of this.openedForms)
        if (form.tabId === tabId) this.openedForms.delete(key);
    },
  };
}
export function naverAdapter(state, driver) {
  async function frames(tabId) {
    const tab = await driver.tab(tabId);
    if (!allowed(tab.url)) throw Error("네이버 탭만 제어할 수 있습니다.");
    await chrome.scripting
      .executeScript({
        target: { tabId, allFrames: true },
        files: ["dom-core.js", "naver-content.js"],
      })
      .catch(() => {});
    return (await chrome.webNavigation.getAllFrames({ tabId })).filter((f) =>
      allowed(f.url),
    );
  }
  async function send(tabId, frameId, msg) {
    await driver.tab(tabId);
    driver.check();
    return chrome.tabs.sendMessage(
      tabId,
      { ...msg, type: "TALK_REVIEW_V14" },
      { frameId },
    );
  }
  async function inspect(tabId, product = "") {
    const out = [];
    for (const f of await frames(tabId)) {
      try {
        const r = await send(tabId, f.frameId, { action: "inspect", product });
        if (r?.ok) out.push({ frame_id: f.frameId, ...r });
      } catch (e) {
        out.push({
          frame_id: f.frameId,
          url: f.url,
          ok: false,
          error: e.message,
        });
      }
    }
    return out;
  }
  async function remember(items) {
    const targets = state.targets;
    for (const v of items) targets[v.id] = v;
    const values = Object.values(targets)
      .sort((a, b) => b.at - a.at)
      .slice(0, 500);
    state.targets = Object.fromEntries(values.map((v) => [v.id, v]));
  }
  async function getTarget(id) {
    const targets = state.targets;
    const t = targets[id];
    if (!t || Date.now() - t.at > 30 * 60 * 1000)
      throw Error("상품 대상이 만료됐습니다. 구매상품 목록을 다시 조회하세요.");
    return t;
  }
  async function realReviewClick(
    t,
    check,
    request = { action: "prepare_open", id: t.id, url: t.url },
  ) {
    return driver.clickPrepared(
      t,
      () => send(t.tab_id, t.frame_id, request),
      async (prepared) => {
        if (prepared.receipt) {
          const verified = await send(t.tab_id, t.frame_id, {
            ...request,
            action: "verify_snapshot",
          });
          if (!verified?.ok)
            throw Error(verified?.error || "등록 전 내용 확인 실패");
        }
      },
      async (prepared) => {
        try {
          return await send(t.tab_id, t.frame_id, {
            action: "click_report",
            id: prepared.click_id || t.id,
          });
        } catch {
          return {};
        }
      },
    );
  }
  async function fillPopup(form, draft, check) {
    delete form.completedSnapshot;
    const target = { tab_id: form.tabId, frame_id: form.frameId };
    const rawAnswers = Array.isArray(draft.answers) ? draft.answers : [];
    const normalizedAnswers = rawAnswers.flatMap((a) =>
      Array.isArray(a?.option_ids)
        ? a.option_ids.map((option_id) => ({ ...a, option_id }))
        : a?.option_id
          ? [a]
          : [],
    );
    const fields = {
      form_id: form.id,
      product: draft.product,
      text: draft.text,
      rating: draft.rating,
      answers: normalizedAnswers,
    };
    const required = (form.questions || []).filter((q) => q.required);
    for (const q of required)
      if (!normalizedAnswers.some((a) => a.question_id === q.id))
        throw Error("필수 추가 질문 응답이 없습니다: " + q.prompt);
    for (const answer of fields.answers) {
      const q = (form.questions || []).find((q) => q.id === answer.question_id);
      if (!q || !q.options.some((o) => o.id === answer.option_id))
        throw Error("현재 상품의 추가 질문 선택지가 아닙니다.");
      if (
        q.type === "single" &&
        new Set(
          fields.answers
            .filter((a) => a.question_id === q.id)
            .map((a) => a.option_id),
        ).size > 1
      )
        throw Error("단일 평가에는 하나의 선택지만 지정할 수 있습니다.");
    }
    const selected = await realReviewClick(target, check, {
      ...fields,
      action: "prepare_rating",
    });
    for (const answer of fields.answers) {
      if (!answer?.question_id || !answer?.option_id) continue;
      const prepared = await send(form.tabId, form.frameId, {
        ...fields,
        ...answer,
        action: "prepare_answer",
      });
      if (!prepared?.ok) throw Error(prepared?.error || "추가 질문 선택 실패");
      if (prepared.already_selected) continue;
      if (prepared.select) {
        check();
        driver.mark();
        const r = await send(form.tabId, form.frameId, {
          ...fields,
          ...answer,
          action: "set_select_answer",
        });
        if (!r?.ok) throw Error(r?.error || "추가 질문 선택 실패");
      } else
        await realReviewClick(target, check, {
          ...fields,
          ...answer,
          action: "prepare_answer",
        });
      await pause(120);
      check();
    }
    await driver.insertPrepared(form.tabId, draft.text, async () => {
      const prepared = await send(form.tabId, form.frameId, {
        ...fields,
        action: "prepare_text",
      });
      if (!prepared?.ok) throw Error(prepared?.error || "입력칸 선택 실패");
    });
    await pause(250);
    check();
    const actual = await send(form.tabId, form.frameId, {
      ...fields,
      action: "verify_snapshot",
    });
    if (!actual?.ok) throw Error(actual?.error || "입력 결과 불일치");
    return {
      ...actual,
      status: "filled",
      submitted: false,
      rating_input: selected.method,
      questions: form.questions || [],
    };
  }

  async function submitPopup(form, snapshot, check) {
    const target = { tab_id: form.tabId, frame_id: form.frameId };
    let pressed = false;
    try {
      const clicked = await realReviewClick(target, check, {
        action: "prepare_submit",
        form_id: form.id,
        product: snapshot.product,
        text: snapshot.text,
        rating: snapshot.rating,
        answers: snapshot.answers,
      });
      pressed = true;
      const result = await send(form.tabId, form.frameId, {
        action: "observe_submit",
        receipt: clicked.receipt,
      });
      if (result?.status === "submitted") form.completedSnapshot = snapshot;
      return result;
    } catch (error) {
      const attempted = pressed || !!error.pointerPressed;
      return {
        ok: false,
        status: attempted ? "uncertain" : "blocked",
        attempted_submit: attempted,
        error: error.message,
      };
    }
  }

  // Handle only the resume-draft confirm during an explicit open operation.
  // Register before input/navigation; dialogs block content-script messaging.
  const isResumeDraftDialog = (message) =>
    /리뷰|작성\s*중|임시\s*저장/.test(message) &&
    /이어서\s*(?:작성|쓰)|계속\s*작성|작성.*계속/.test(message);
  async function openForm(t, check) {
    const records = [],
      pending = new Set();
    let failure = "";
    const debuggee = { tabId: t.tab_id };
    const checked = () => {
      check();
      if (failure) throw Error(failure);
    };
    const runJob = (promise) => {
      pending.add(promise);
      promise.finally(() => pending.delete(promise));
    };
    const onEvent = (source, method, params) => {
      if (source.tabId !== t.tab_id) return;
      const target = {
        tabId: source.tabId,
        ...(source.sessionId ? { sessionId: source.sessionId } : {}),
      };
      if (method !== "Page.javascriptDialogOpening") return;
      if (
        params.type !== "confirm" ||
        !isResumeDraftDialog(params.message || "")
      ) {
        failure =
          "리뷰 이어쓰기 확인창과 다른 브라우저 대화상자가 열렸습니다. 자동 승인하지 않았습니다.";
        return;
      }
      runJob(
        (async () => {
          try {
            checked();
            await chrome.debugger.sendCommand(
              target,
              "Page.handleJavaScriptDialog",
              { accept: true },
            );
            records.push({
              tab_id: source.tabId,
              type: "resume_draft",
              accepted: true,
            });
          } catch (e) {
            failure = e.message;
          }
        })(),
      );
    };
    chrome.debugger.onEvent.addListener(onEvent);
    try {
      return await driver.debug(t.tab_id, async () => {
        await chrome.debugger.sendCommand(debuggee, "Page.enable");
        checked();
        const form = await openFormInner(
          { ...t, debuggerAttached: true },
          checked,
        );
        await Promise.all([...pending]);
        checked();
        if (form.draft_resumed)
          records.push({
            tab_id: form.tabId,
            type: "resume_draft",
            accepted: true,
          });
        form.observation = { ...form.observation, draft_dialogs: records };
        return form;
      });
    } finally {
      await Promise.allSettled([...pending]);
      chrome.debugger.onEvent.removeListener(onEvent);
    }
  }
  const openedForms = state.openedForms;
  async function openFormInner(t, check) {
    const readPopup = async (tabId) => {
      for (const frame of await frames(tabId)) {
        check();
        const prepared = await send(tabId, frame.frameId, {
          action: "prepare_resume",
        });
        if (!prepared?.ok)
          throw Error(prepared?.error || "이어쓰기 확인창 인식 실패");
        if (!prepared.none) {
          await realReviewClick(
            {
              tab_id: tabId,
              frame_id: frame.frameId,
              debuggerAttached: tabId === t.tab_id,
            },
            check,
            { action: "prepare_resume" },
          );
          await pause(200);
        }
      }
      return inspect(tabId, t.product);
    };

    const cached = openedForms.get(t.id);
    if (cached) {
      try {
        const detail = await readPopup(cached.tabId);
        if (
          detail.some(
            (f) =>
              f.frame_id === cached.frameId &&
              (f.forms || []).some(
                (x) => x.id === cached.id && x.product_matched,
              ),
          )
        )
          return cached;
      } catch {}
      openedForms.delete(t.id);
    }
    const popupSince = Date.now();
    const windowsBefore = await chrome.windows.getAll({
      windowTypes: ["normal", "popup"],
    });
    const windowTypes = new Map(windowsBefore.map((w) => [w.id, w.type]));
    const tabView = (tab) => ({
      id: tab.id,
      window_id: tab.windowId,
      window_type: windowTypes.get(tab.windowId),
      url: tab.url || tab.pendingUrl,
      pending_url: tab.pendingUrl,
      status: tab.status,
      title: tab.title,
      opener_tab_id: tab.openerTabId,
    });
    const before = (await chrome.tabs.query({}))
      .filter((tab) => allowed(tab.url))
      .map(tabView);
    const beforeByID = new Map(before.map((tab) => [tab.id, tab]));
    const beforePopupForms = new Map();
    const existingMatches = [];
    for (const tab of before.filter((tab) => tab.window_type === "popup")) {
      try {
        const detail = await readPopup(tab.id);
        beforePopupForms.set(
          tab.id,
          JSON.stringify(detail.map((f) => f.forms || [])),
        );
        for (const f of detail)
          for (const x of f.forms || [])
            if (x.product_matched)
              existingMatches.push({
                windowId: tab.window_id,
                windowType: tab.window_type,
                tabId: tab.id,
                frameId: f.frame_id,
                ...x,
              });
      } catch {}
    }
    // Recovery path: a previous attempt may already have opened the correct Naver review popup.
    // Reuse it instead of requiring a new tab/window event. Product matching disambiguates multiple popups.
    if (existingMatches.length === 1) {
      const recovered = existingMatches[0];
      recovered.observation = {
        recovered_existing_popup: true,
        before_tabs: before,
      };
      openedForms.set(t.id, recovered);
      return recovered;
    }
    if (existingMatches.length > 1) {
      const err = Error(
        "이미 열린 동일 상품 리뷰 입력창이 여러 개라 하나로 결정할 수 없습니다.",
      );
      err.observation = {
        existing_matches: existingMatches.map((x) => ({
          tab_id: x.tabId,
          frame_id: x.frameId,
          form_id: x.id,
          context: x.context,
        })),
      };
      throw err;
    }
    check();
    await frames(t.tab_id);
    let opened;
    try {
      opened = await realReviewClick(t, check);
    } catch (e) {
      opened = { ok: false, click_dispatched: null, error: e.message };
    }
    const observation = {
      before_windows: windowsBefore.map((w) => ({ id: w.id, type: w.type })),
      before_tabs: before,
      open_result: opened,
      new_tabs: [],
      changed_tabs: [],
      new_windows: [],
      reused_windows: [],
      observed_frames: [],
    };
    let form;
    for (let n = 0; n < 20 && !form; n++) {
      await pause(400);
      check();
      const windowsAfter = await chrome.windows.getAll({
        windowTypes: ["normal", "popup"],
      });
      for (const w of windowsAfter) windowTypes.set(w.id, w.type);
      observation.new_windows = windowsAfter
        .filter((w) => !windowsBefore.some((old) => old.id === w.id))
        .map((w) => ({ id: w.id, type: w.type }));
      const after = (await chrome.tabs.query({}))
        .filter((tab) => allowed(tab.url || tab.pendingUrl))
        .map(tabView);
      observation.new_tabs = after.filter((tab) => !beforeByID.has(tab.id));
      observation.changed_tabs = after.filter(
        (tab) =>
          beforeByID.has(tab.id) && beforeByID.get(tab.id).url !== tab.url,
      );
      // Pre-existing unrelated review-detail tabs are not evidence of this click.
      // Also inspect new noopener windows: they need not have openerTabId set.
      const eventPopupIds = new Set(
        [...popupEvents.values()]
          .filter(
            (v) =>
              v.at >= popupSince - 250 &&
              (!v.openerTabId || v.openerTabId === t.tab_id),
          )
          .map((v) => v.tabId),
      );
      const candidates = after
        .filter(
          (tab) =>
            tab.id === t.tab_id ||
            eventPopupIds.has(tab.id) ||
            !beforeByID.has(tab.id) ||
            observation.changed_tabs.some((v) => v.id === tab.id) ||
            tab.window_type === "popup",
        )
        .sort(
          (a, b) =>
            (eventPopupIds.has(b.id) ? 1 : 0) -
            (eventPopupIds.has(a.id) ? 1 : 0),
        );
      const matches = [],
        observed = [];
      for (const tab of candidates) {
        try {
          const detail = await readPopup(tab.id);
          const reused =
            beforePopupForms.has(tab.id) &&
            JSON.stringify(detail.map((f) => f.forms || [])) !==
              beforePopupForms.get(tab.id);
          if (
            reused &&
            !observation.reused_windows.some((w) => w.id === tab.window_id)
          )
            observation.reused_windows.push({
              id: tab.window_id,
              type: tab.window_type,
            });
          const isReviewPopup =
            tab.window_type === "popup" &&
            /\/popup\/reviews\/(?:monthly-)?form(?:[?#]|$)/.test(
              String(tab.url || ""),
            );
          const eligible =
            tab.id === t.tab_id ||
            !beforeByID.has(tab.id) ||
            observation.changed_tabs.some((v) => v.id === tab.id) ||
            reused ||
            isReviewPopup;
          for (const f of detail) {
            observed.push({
              window_id: tab.window_id,
              window_type: tab.window_type,
              tab_id: tab.id,
              frame_id: f.frame_id,
              url: f.url,
              forms: f.forms || [],
              editors: f.editors || [],
              error: f.error,
            });
            for (const x of f.forms || [])
              if (eligible && x.product_matched)
                matches.push({
                  windowId: tab.window_id,
                  windowType: tab.window_type,
                  tabId: tab.id,
                  frameId: f.frame_id,
                  ...x,
                });
          }
        } catch (e) {
          observed.push({ tab_id: tab.id, url: tab.url, error: e.message });
        }
      }
      observation.observed_frames = observed;
      if (matches.length > 1) {
        observation.failure = "ambiguous_editors";
        break;
      }
      if (matches.length === 1) form = matches[0];
    }
    if (!form) {
      const hasForms = observation.observed_frames.some((f) => f.forms?.length);
      const hasEditors = observation.observed_frames.some(
        (f) => f.editors?.length,
      );
      observation.failure ||= hasForms
        ? "editor_detected_but_unmatched"
        : hasEditors
          ? "editor_detected_but_form_unrecognized"
          : observation.new_tabs.length || observation.changed_tabs.length
            ? "navigation_observed_without_editor"
            : "no_editor_or_navigation_observed";
      const err = Error(
        hasForms
          ? "리뷰 폼은 찾았지만 상품 연결을 확인하지 못했습니다. 관찰된 form context를 확인하세요."
          : hasEditors
            ? "리뷰 입력칸은 찾았지만 폼 구조를 인식하지 못했습니다. DOM 인식 규칙을 확인하세요."
            : "리뷰 입력창을 확인하지 못했습니다. 클릭 전후 관찰 결과를 확인하세요.",
      );
      err.observation = observation;
      throw err;
    }
    form.observation = observation;
    openedForms.set(t.id, form);
    return form;
  }
  async function run(cmd) {
    const a = cmd.args || {};
    const check = driver.check;
    check();
    if (cmd.action === "refresh") {
      if (!Number.isInteger(a.tab_id) || a.tab_id <= 0)
        throw Error("새로고침할 tab_id가 필요합니다.");
      const tab = await chrome.tabs.get(a.tab_id),
        win = await chrome.windows.get(tab.windowId);
      if (!allowed(tab.url) || win.type === "popup")
        throw Error("네이버 메인 탭만 새로고침할 수 있습니다.");
      for (const frame of await frames(tab.id)) {
        const state = await send(tab.id, frame.frameId, {
          action: "close_state",
        });
        if (!state?.ok || state.dirty)
          throw Error("작성 중인 내용이 있어 새로고침하지 않았습니다.");
      }
      check();
      const result = await driver.reload(tab.id);
      const targets = state.targets;
      for (const [id, target] of Object.entries(targets))
        if (target.tab_id === tab.id) delete targets[id];
      if (!result.ok) return result;
      return {
        ...result,
        guidance:
          "inspect this tab again; previous product IDs are invalid after reload.",
      };
    }
    if (cmd.action === "close_popup") {
      if (!Number.isInteger(a.tab_id) || a.tab_id <= 0)
        throw Error("닫을 리뷰 팝업의 tab_id가 필요합니다.");
      let tab;
      try {
        tab = await chrome.tabs.get(a.tab_id);
      } catch {
        return { ok: true, status: "already_closed", tab_id: a.tab_id };
      }
      const win = await chrome.windows.get(tab.windowId);
      if (
        win.type !== "popup" ||
        !allowed(tab.url) ||
        !/^\/popup\/reviews\/(?:form|monthly-form|redirect)$/.test(
          new URL(tab.url).pathname,
        )
      )
        throw Error(
          "네이버 리뷰 팝업만 닫을 수 있습니다. 일반 탭은 닫지 않았습니다.",
        );
      const available = await frames(tab.id);
      if (!available.length) throw Error("팝업 상태를 확인하지 못했습니다.");
      for (const frame of available) {
        const state = await send(tab.id, frame.frameId, {
          action: "close_state",
        });
        if (!state?.ok) throw Error("작성 중인 내용 확인 실패");
        if (state.dirty) {
          const form = [...openedForms.values()].find(
            (f) =>
              f.tabId === tab.id &&
              f.frameId === frame.frameId &&
              f.completedSnapshot,
          );
          if (!form)
            throw Error(
              "작성 중이거나 등록 결과가 불확실한 내용이 있어 닫지 않았습니다.",
            );
          const current = await send(tab.id, frame.frameId, {
            ...form.completedSnapshot,
            form_id: form.id,
            action: "verify_snapshot",
          });
          if (!current?.ok)
            throw Error("등록 후 수정된 내용이 있어 닫지 않았습니다.");
        }
      }
      check();
      const latest = await chrome.tabs.get(tab.id);
      if (latest.url !== tab.url || latest.windowId !== tab.windowId)
        throw Error("팝업이 변경되어 닫지 않았습니다.");
      await driver.close(tab.id);
      for (const [id, form] of openedForms)
        if (form.tabId === tab.id) openedForms.delete(id);
      return {
        ok: true,
        status: "popup_closed",
        tab_id: tab.id,
        window_id: tab.windowId,
      };
    }
    if (cmd.action === "tabs")
      return {
        ok: true,
        tabs: (await chrome.tabs.query({}))
          .filter((t) => allowed(t.url))
          .map((t) => ({ tab_id: t.id, title: t.title, url: t.url })),
      };
    if (["inspect", "navigate", "scroll"].includes(cmd.action)) {
      let tabId = a.tab_id,
        details,
        actionResult;
      if (cmd.action === "navigate") {
        const target = await getTarget(a.target_id);
        tabId = target.tab_id;
        if (target.kind !== "navigation")
          throw Error("조회 결과의 navigation 대상만 이동할 수 있습니다.");
        check();
        await frames(tabId);
        try {
          actionResult = await realReviewClick(target, check, {
            action: "prepare_navigation",
            id: target.id,
          });
        } catch (e) {
          actionResult = {
            ok: false,
            status: "navigation_unconfirmed",
            error: e.message,
          };
        }
        if (
          actionResult?.ok === false &&
          actionResult.status !== "navigation_unconfirmed"
        )
          return actionResult;
        await pause(1800);
        check();
        details = await inspect(tabId);
      } else if (cmd.action === "scroll") {
        const available = await frames(tabId);
        const selected = available.find((f) => f.frameId === (a.frame_id || 0));
        if (!selected) throw Error("스크롤할 프레임을 찾지 못했습니다.");
        let result;
        for (let i = 0; i < (a.steps || 1); i++) {
          check();
          driver.mark();
          result = await send(tabId, selected.frameId, {
            action: "scroll",
            steps: 1,
          });
          if (!result?.ok) break;
        }
        if (!result?.ok)
          return result || { ok: false, error: "스크롤 응답 없음" };
        details = [{ frame_id: selected.frameId, ...result }];
      } else details = await inspect(tabId);
      const items = [],
        navigation = [];
      for (const f of details) {
        for (const item of f.items || [])
          items.push({
            ...item,
            kind: "item",
            tab_id: tabId,
            frame_id: f.frame_id,
            url: f.url,
            at: Date.now(),
          });
        for (const item of f.navigation || [])
          navigation.push({
            ...item,
            kind: "navigation",
            tab_id: tabId,
            frame_id: f.frame_id,
            url: f.url,
            at: Date.now(),
          });
      }
      await remember([...items, ...navigation]);
      const extractionFailed =
        items.length === 0 &&
        details.some((f) => f.status === "extraction_failed");
      return {
        ok: details.length > 0 && !extractionFailed,
        error: extractionFailed
          ? "리뷰 버튼을 상품 ID와 연결하지 못했습니다. 상품 목록으로 진단 텍스트를 대신 사용하거나 미로딩이라고 단정하지 마세요."
          : undefined,
        actionable_item_count: items.length,
        items,
        navigation,
        action_result: actionResult,
        frames: details.map(({ items, navigation, ...f }) => f),
        scope:
          "현재 Chrome의 렌더링된 DOM을 읽었습니다. 빈 items는 미로딩의 증거가 아닙니다. diagnostics를 확인하고 navigation 또는 scroll을 사용하세요. 전체 구매내역 수집 완료를 의미하지 않습니다.",
      };
    }
    if (cmd.action === "resolve")
      return { ok: true, items: await Promise.all(a.ids.map(getTarget)) };
    if (cmd.action === "open") {
      const t = await getTarget(a.target_id);
      if (t.kind !== "item") throw Error("상품 id를 지정하세요.");
      const form = await openForm(t, check);
      return {
        ok: true,
        status: "editor_open",
        target_id: t.id,
        product: t.product,
        window_id: form.windowId,
        window_type: form.windowType,
        tab_id: form.tabId,
        frame_id: form.frameId,
        form_id: form.id,
        context: form.context,
        questions: form.questions || [],
        rating_values: form.rating_values || [],
        fill_contract: {
          reviews: [
            {
              id: t.id,
              product: t.product,
              rating: "1..5 integer",
              text: "10..5000 chars",
              answers: (form.questions || []).map((q) => ({
                question_id: q.id,
                option_id: "choose one option id from questions.options",
              })),
            },
          ],
        },
        observation: form.observation,
      };
    }
    if (cmd.action === "fill") {
      if (a.reviews?.length !== 1)
        throw Error("입력은 상품 한 개씩 처리합니다.");
      const draft = a.reviews[0],
        t = await getTarget(draft.id);
      if (t.kind !== "item" || t.product !== draft.product)
        throw Error("상품 대상 불일치");
      const form = await openForm(t, check);
      check();
      return fillPopup(form, draft, check);
    }
    if (["read_current", "submit_current"].includes(cmd.action)) {
      const t = await getTarget(a.target_id),
        form = openedForms.get(t.id);
      if (t.kind !== "item" || !form)
        throw Error("먼저 이 상품의 리뷰창을 open 또는 fill로 열어야 합니다.");
      const message = { form_id: form.id, product: t.product };
      if (cmd.action === "read_current")
        return send(form.tabId, form.frameId, {
          ...message,
          action: "read_current",
        });
      if (
        !a.snapshot ||
        a.snapshot.product !== t.product ||
        a.snapshot.form_id !== form.id
      )
        throw Error("확인한 입력창과 현재 대상이 다릅니다.");
      const { receipts = {} } = await chrome.storage.local.get("receipts");
      const digest = await crypto.subtle.digest(
        "SHA-256",
        new TextEncoder().encode(t.url + "|" + t.summary),
      );
      const key =
        "item:" +
        Array.from(new Uint8Array(digest), (v) =>
          v.toString(16).padStart(2, "0"),
        ).join("");
      if (receipts[key])
        return {
          ok: false,
          status: "already_attempted",
          error:
            "이미 등록을 시도한 상품입니다. 구매내역에서 결과를 확인하세요.",
        };
      check();
      receipts[key] = { status: "uncertain", product: t.product };
      await chrome.storage.local.set({ receipts });
      const result = await submitPopup(form, a.snapshot, check);
      if (result?.attempted_submit === false) delete receipts[key];
      else receipts[key] = result || { status: "uncertain" };
      await chrome.storage.local.set({ receipts });
      return result;
    }
    if (cmd.action !== "submit") throw Error("지원하지 않는 작업");
    const { receipts = {} } = await chrome.storage.local.get("receipts");
    // Persist before any mutation. A retry after interruption must not submit twice.
    if (receipts[cmd.id]) return receipts[cmd.id];
    receipts[cmd.id] = {
      ok: false,
      status: "uncertain",
      error:
        "이 요청은 이미 시작됐습니다. 구매후기 등록 여부를 직접 확인하세요.",
    };
    await chrome.storage.local.set({ receipts });
    const results = [];
    for (const draft of a.reviews) {
      let result;
      try {
        const t = await getTarget(draft.id);
        if (t.product !== draft.product)
          throw Error("승인한 상품명과 현재 상품이 다릅니다.");
        const digest = await crypto.subtle.digest(
          "SHA-256",
          new TextEncoder().encode(t.url + "|" + t.summary),
        );
        const key =
          "item:" +
          Array.from(new Uint8Array(digest), (v) =>
            v.toString(16).padStart(2, "0"),
          ).join("");
        if (receipts[key])
          throw Error(
            "이미 등록을 시도한 상품입니다. 중복 등록 방지를 위해 구매내역에서 확인하세요.",
          );
        const form = await openForm(t, check),
          tabId = form.tabId;
        await fillPopup(form, draft, check);
        check();
        receipts[key] = {
          status: "uncertain",
          product: t.product,
          at: Date.now(),
        };
        await chrome.storage.local.set({ receipts });
        result = await submitPopup(form, draft, check);
        if (!result)
          result = {
            ok: false,
            status: "uncertain",
            error: "등록 응답을 확인하지 못했습니다.",
          };
        if (result.attempted_submit === false) delete receipts[key];
        else receipts[key] = result;
      } catch (e) {
        result = { ok: false, error: e.message, observation: e.observation };
      }
      results.push({ id: draft.id, ...result });
      // Stop on an error instead of continuing with potentially changed navigation.
      if (!result.ok) break;
    }
    const result = {
      ok: results.length === a.reviews.length && results.every((r) => r.ok),
      results,
      not_attempted: a.reviews.slice(results.length).map((v) => v.id),
    };
    receipts[cmd.id] = result;
    await chrome.storage.local.set({ receipts });
    return result;
  }

  return run;
}
