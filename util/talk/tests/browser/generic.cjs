// Real MV3 extension + Chrome + WebSocket. All sites are local synthetic fixtures.
const { chromium } = require("../../web/node_modules/playwright");
const {
  wsServer,
} = require("../../web/node_modules/playwright-core/lib/utilsBundle.js");
const http = require("node:http");
const fs = require("node:fs"),
  os = require("node:os"),
  path = require("node:path"),
  assert = require("node:assert/strict");
(async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "talk-browser-core-"));
  const wss = new wsServer({ port: 0, host: "127.0.0.1" });
  await new Promise((r) => wss.once("listening", r));
  let socket,
    seq = 0;
  const pending = new Map();
  wss.on("connection", (ws) => {
    socket = ws;
    ws.on("message", (data) => {
      const m = JSON.parse(data);
      if (m.token) {
        assert.equal(m.protocol, 14);
        ws.send(JSON.stringify({ ready: true }));
        return;
      }
      if (m.id) {
        const p = pending.get(m.id);
        if (p) {
          clearTimeout(p.timer);
          pending.delete(m.id);
          p.resolve(m.result);
        }
      }
    });
  });
  const begin = (
    action,
    args = {},
    session = "test-a",
    expires = Date.now() + 20000,
  ) => {
    const id = String(++seq);
    const promise = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(Error("RPC timeout " + JSON.stringify(args)));
      }, 25000);
      pending.set(id, { resolve, reject, timer });
    });
    socket.send(
      JSON.stringify({ id, action, args, session_id: session, expires }),
    );
    return { id, promise };
  };
  const call = (args, session) => begin("browser", args, session).promise;
  const ok = async (args, session) => {
    const r = await call(args, session);
    assert.equal(r.ok, true, JSON.stringify(r));
    return r;
  };
  const ext = path.join(dir, "extension");
  fs.cpSync(
    path.resolve(__dirname, "../../internal/browserbridge/extension"),
    ext,
    { recursive: true },
  );
  const mf = path.join(ext, "manifest.json"),
    manifest = JSON.parse(fs.readFileSync(mf));
  manifest.host_permissions.push(
    "http://127.0.0.1/*",
    "https://fixture.test/*",
  );
  fs.writeFileSync(mf, JSON.stringify(manifest));
  let fixture;
  const context = await chromium.launchPersistentContext(
    path.join(dir, "profile"),
    {
      executablePath:
        process.env.TALK_CHROME ||
        "/home/edp1096/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome",
      headless: true,
      args: [
        "--no-sandbox",
        `--disable-extensions-except=${ext}`,
        `--load-extension=${ext}`,
      ],
    },
  );
  try {
    const worker =
      context.serviceWorkers()[0] ||
      (await context.waitForEvent("serviceworker"));
    await worker.evaluate(
      async (server) =>
        chrome.storage.local.set({ server, token: "a".repeat(64) }),
      `http://127.0.0.1:${wss.address().port}`,
    );
    const settings = await context.newPage();
    await settings.goto(new URL("connect.html", worker.url()).href);
    await settings.evaluate(() =>
      chrome.runtime.sendMessage({ type: "CONNECT" }),
    );
    await settings.waitForFunction(async () =>
      Boolean((await chrome.runtime.sendMessage({ type: "STATUS" })).connected),
    );
    const html = `<!doctype html><title>Generic fixture</title><label>Name<input id="name"></label><button id="send">Send</button><select aria-label="Choice"><option value="a">A</option><option value="b">B</option></select><div id="result"></div><iframe src="/frame"></iframe><script>window.count=0;document.querySelector('#send').onclick=e=>{if(!e.isTrusted)throw Error('untrusted');window.count++;document.querySelector('#result').textContent='Done '+window.count;};</script>`;
    fixture = http.createServer((req, res) => {
      res.setHeader("Content-Type", "text/html");
      res.end(
        req.url === "/frame"
          ? "<button onclick=\"this.textContent='Frame done'\">Frame action</button>"
          : html,
      );
    });
    await new Promise((r) => fixture.listen(0, "127.0.0.1", r));
    const origin = `http://127.0.0.1:${fixture.address().port}`;
    const sitesPage = await context.newPage();
    await sitesPage.goto(new URL('sites.html',worker.url()).href);
    await sitesPage.locator('#allowedSites li').first().waitFor();
    assert.ok(await sitesPage.locator('main').evaluate(el=>el.getBoundingClientRect().width)>700);
    await sitesPage.locator('#filter').fill('no-such-allowed-site');
    assert.equal(await sitesPage.locator('#allowedSites li').count(),0);
    assert.equal(await sitesPage.locator('#empty').isVisible(),true);
    await sitesPage.locator('#filter').fill('fixture.test');
    assert.ok(await sitesPage.locator('#allowedSites li').count()>=1);
    assert.equal(await sitesPage.locator('#allowedSites button').count(),0);
    await sitesPage.locator('#filter').fill('');
    await sitesPage.screenshot({path:'/tmp/talk-browser-sites-desktop.png',fullPage:true});
    await sitesPage.setViewportSize({width:390,height:844});
    assert.equal(await sitesPage.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
    await sitesPage.screenshot({path:'/tmp/talk-browser-sites-mobile.png',fullPage:true});
    await sitesPage.close();
    const page = await context.newPage();
    await page.goto(origin + "/main");
    const tabs = await ok({ action: "tabs" }),
      tab = tabs.tabs.find((t) => t.url === origin + "/main").tab_id;
    assert.equal((await call({ action: "observe", tab_id: tab })).ok, false);
    await ok({ action: "attach", tab_id: tab });
    assert.equal(
      (await call({ action: "attach", tab_id: tab }, "test-b")).ok,
      false,
    );
    const observe = () => ok({ action: "observe", tab_id: tab });
    const ref = (view, name) =>
      view.frames.flatMap((f) => f.elements).find((e) => e.name === name).ref;
    let view = await observe();
    await ok({
      action: "fill",
      tab_id: tab,
      ref: ref(view, "Name"),
      text: "Hello Chrome",
    });
    assert.equal(await page.locator("#name").inputValue(), "Hello Chrome");
    assert.equal((await call({ action: "reload", tab_id: tab })).ok, false);
    await ok({
      action: "select",
      tab_id: tab,
      ref: ref(view, "Choice"),
      value: "b",
    });
    assert.equal(await page.locator("select").inputValue(), "b");
    await ok({ action: "click", tab_id: tab, ref: ref(view, "Send") });
    assert.equal(await page.evaluate(() => window.count), 1);
    await ok({ action: "click", tab_id: tab, ref: ref(view, "Frame action") });
    assert.equal(
      await page.frameLocator("iframe").locator("button").innerText(),
      "Frame done",
    );
    await ok({ action: "wait", tab_id: tab, text: "Done 1" });
    const shot = await ok({ action: "screenshot", tab_id: tab });
    assert.ok(shot.image_url.startsWith("data:image/jpeg;base64,"));
    // DOM replacement and identical labels do not reuse node identity.
    const old = ref(view, "Send");
    await page
      .locator("#send")
      .evaluate((el) => el.replaceWith(el.cloneNode(true)));
    assert.equal(
      (await call({ action: "click", tab_id: tab, ref: old })).ok,
      false,
    );
    view = await observe();
    await page.evaluate(() => {
      const overlay = document.createElement("div");
      overlay.id = "cover";
      overlay.style = "position:fixed;inset:0;z-index:9999;background:white";
      document.body.append(overlay);
    });
    assert.equal(
      (await call({ action: "click", tab_id: tab, ref: ref(view, "Send") })).ok,
      false,
    );
    await page.locator("#cover").evaluate((el) => el.remove());
    // A cancelled queued action must never be sent to Chrome.
    const wait = begin("browser", {
      action: "wait",
      tab_id: tab,
      text: "never appears",
      timeout_ms: 15000,
    });
    const click = begin("browser", {
      action: "click",
      tab_id: tab,
      ref: ref(view, "Send"),
    });
    socket.send(JSON.stringify({ cancel: click.id }));
    socket.send(JSON.stringify({ cancel: wait.id }));
    assert.equal((await wait.promise).ok, false);
    const cancelled = await click.promise;
    assert.equal(cancelled.ok, false);
    assert.equal(cancelled.effect_state, "none");
    assert.equal(
      (
        await begin(
          "browser",
          { action: "click", tab_id: tab, ref: ref(view, "Send") },
          "test-a",
          Date.now() - 1,
        ).promise
      ).ok,
      false,
    );
    // Navigation destroys document references even when the replacement looks identical.
    await page.reload();
    assert.equal(
      (await call({ action: "click", tab_id: tab, ref: ref(view, "Send") })).ok,
      false,
    );
    await ok({ action: "reload", tab_id: tab });
    // Disconnect while a wait owns the queue; the queued mutation must be discarded.
    view = await observe();
    const disconnectedWait = begin("browser", {
      action: "wait",
      tab_id: tab,
      text: "not present",
      timeout_ms: 15000,
    });
    const disconnectedClick = begin("browser", {
      action: "click",
      tab_id: tab,
      ref: ref(view, "Send"),
    });
    const beforeCount = await page.evaluate(() => window.count);
    socket.terminate();
    for (const id of [disconnectedWait.id, disconnectedClick.id]) {
      const p = pending.get(id);
      clearTimeout(p.timer);
      pending.delete(id);
      p.resolve({ ok: false });
    }
    await settings.evaluate(() =>
      chrome.runtime.sendMessage({ type: "CONNECT" }),
    );
    await settings.waitForFunction(async () =>
      Boolean((await chrome.runtime.sendMessage({ type: "STATUS" })).connected),
    );
    await observe();
    assert.equal(await page.evaluate(() => window.count), beforeCount);
    // Native keyboard input, clean navigation and old refs invalidation.
    view = await observe();
    await ok({
      action: "press",
      tab_id: tab,
      ref: ref(view, "Name"),
      key: "Tab",
    });
    await ok({ action: "navigate", tab_id: tab, url: origin + "/second" });
    await page.waitForURL(origin + "/second");
    assert.equal(
      (await call({ action: "click", tab_id: tab, ref: ref(view, "Send") })).ok,
      false,
    );
    await ok({ action: "release", tab_id: tab });
    await ok({ action: "attach", tab_id: tab }, "test-b");
    await ok({ action: "stop" }, "test-b");
    assert.equal(page.isClosed(), false);
    await ok({ action: "attach", tab_id: tab });
    const released = await settings.evaluate(() =>
      chrome.runtime.sendMessage({ type: "RELEASE_TABS" }),
    );
    assert.equal(released.ok, true);
    await ok({ action: "attach", tab_id: tab }, "test-b");
    await ok({ action: "release", tab_id: tab }, "test-b");
    await ok({ action: "attach", tab_id: tab });
    const opened = await ok({ action: "open", url: origin + "/new" });
    await new Promise((r) => setTimeout(r, 300));
    await ok({ action: "close", tab_id: opened.tab_id });
    assert.equal(
      (await call({ action: "open", url: "https://ungranted.test/" })).ok,
      false,
    );
    console.log(
      "PASS generic: permissions, sessions, native input, iframe, stale refs, overlay, cancellation, deadline, dirty forms, screenshot, release/close",
    );
  } finally {
    await context.close();
    if (fixture) await new Promise((r) => fixture.close(r));
    for (const p of pending.values()) clearTimeout(p.timer);
    for (const c of wss.clients) c.terminate();
    await new Promise((r) => wss.close(r));
    fs.rmSync(dir, { recursive: true, force: true });
  }
})().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
