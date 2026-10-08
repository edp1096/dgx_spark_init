// Chrome's native permission approval bubble is unavailable to headless page
// automation. Seed previously USER-APPROVED optional grants in a throwaway
// profile; all getAll/contains/remove calls, UI and script lifecycle are real.
const { chromium } = require("../../web/node_modules/playwright");
const fs = require("node:fs"),
  path = require("node:path"),
  assert = require("node:assert/strict");
(async () => {
  const dir = fs.mkdtempSync("/tmp/talk-optional-permissions-"),
    ext = path.join(dir, "extension"),
    profile = path.join(dir, "profile");
  fs.cpSync(
    path.resolve(__dirname, "../../internal/browserbridge/extension"),
    ext,
    { recursive: true },
  );
  const manifest = JSON.parse(fs.readFileSync(path.join(ext, "manifest.json")));
  assert.deepEqual(manifest.host_permissions, []);
  assert.equal(manifest.content_scripts, undefined);
  let context, id;
  const launch = async () => {
    context = await chromium.launchPersistentContext(profile, {
      executablePath:
        process.env.TALK_CHROME ||
        "/home/edp1096/.cache/ms-playwright/chromium-1243/chrome-linux-arm64/chrome",
      headless: true,
      args: [
        "--no-sandbox",
        `--disable-extensions-except=${ext}`,
        `--load-extension=${ext}`,
      ],
    });
    const worker =
      context.serviceWorkers()[0] ||
      (await context.waitForEvent("serviceworker"));
    id = new URL(worker.url()).hostname;
    const page = await context.newPage();
    await page.goto(`chrome-extension://${id}/sites.html`);
    return { worker, page };
  };
  const seedApprovedGrant = () => {
    const file = path.join(profile, "Default/Preferences"),
      prefs = JSON.parse(fs.readFileSync(file));
    const entry = prefs.extensions.settings[id];
    for (const field of ["active_permissions", "granted_permissions"])
      entry[field].explicit_host = ["https://shopping.naver.com/*"];
    fs.writeFileSync(file, JSON.stringify(prefs));
  };
  try {
    let { worker, page } = await launch();
    assert.deepEqual(
      await worker.evaluate(
        async () => (await chrome.permissions.getAll()).origins || [],
      ),
      [],
    );
    assert.equal(
      await worker.evaluate(
        async () =>
          (await chrome.scripting.getRegisteredContentScripts()).length,
      ),
      0,
    );
    await context.close();
    seedApprovedGrant();
    ({ worker, page } = await launch());
    await page.waitForFunction(
      async () =>
        (await chrome.scripting.getRegisteredContentScripts()).length === 2,
    );
    // Request an already approved site through the actual form (no native bubble).
    await page.locator("#site").fill("https://shopping.naver.com");
    await page.locator("#addSite button").click();
    await page.waitForFunction(() =>
      document
        .querySelector("#siteStatus")
        .textContent.includes("접근을 허용했습니다."),
    );
    await context.route("https://shopping.naver.com/**", (r) =>
      r.fulfill({
        contentType: "text/html",
        body: '<meta charset="utf-8"><script>window.savedConfirm=window.confirm;window.resumed=confirm("작성 중이던 리뷰가 있습니다. 이어서 작성하시겠습니까?");</script><title>Permission fixture</title>',
      }),
    );
    const naver = await context.newPage();
    await naver.goto("https://shopping.naver.com/popup/reviews/form");
    assert.equal(await naver.evaluate(() => window.resumed), true);
    await page
      .getByRole("button", {
        name: "https://shopping.naver.com/* 권한 해제",
        exact: true,
      })
      .click();
    await page.waitForFunction(
      async () =>
        !(await chrome.permissions.contains({
          origins: ["https://shopping.naver.com/*"],
        })),
    );
    await page.waitForFunction(
      async () =>
        (await chrome.scripting.getRegisteredContentScripts()).length === 0,
    );
    await naver.waitForFunction(() => !window.__sparkTalkResumeDraft);
    naver.on("dialog", (dialog) => dialog.dismiss());
    assert.equal(
      await naver.evaluate(() =>
        window.savedConfirm(
          "작성 중이던 리뷰가 있습니다. 이어서 작성하시겠습니까?",
        ),
      ),
      false,
    );
    const denied = await page.evaluate(async () => {
      const { BrowserDriver } = await import(
        chrome.runtime.getURL("browser-driver.js")
      );
      const tab = (await chrome.tabs.query({})).find((t) =>
        t.url.includes("shopping.naver.com/popup/reviews/form"),
      );
      try {
        await new BrowserDriver(
          {
            claim: () => {
              throw Error("must not claim");
            },
          },
          () => {},
        ).tab(tab.id);
        return false;
      } catch (error) {
        return error.message.includes("접근 권한");
      }
    });
    assert.equal(denied, true);
    assert.equal(await page.locator("#allowedSites li").count(), 0);
    await naver.reload();
    assert.equal(await naver.evaluate(() => window.resumed), false);
    // Reload a fresh browser with a newly approved persisted grant.
    await context.close();
    seedApprovedGrant();
    ({ worker, page } = await launch());
    await page.waitForFunction(
      async () =>
        (await chrome.scripting.getRegisteredContentScripts()).length === 2,
    );
    assert.equal(
      await worker.evaluate(async () =>
        chrome.permissions.contains({
          origins: ["https://shopping.naver.com/*"],
        }),
      ),
      true,
    );
    console.log(
      "PASS: no mandatory hosts, optional permission UI removal, unregister, live hook revocation, retained hook revocation, blocked control, no injection after reload, regrant registration",
    );
  } finally {
    if (context) await context.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
