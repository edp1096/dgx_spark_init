// Only the early resume-dialog hook needs registration; all other site code is
// injected on demand through BrowserDriver after checking current permission.
const ids = ["talk-review-resume", "talk-review-permission-guard"];
let queue = Promise.resolve();
export function syncSiteScripts() {
  queue = queue
    .catch(() => {})
    .then(async () => {
      const granted = await chrome.permissions.contains({
        origins: ["https://shopping.naver.com/*"],
      });
      const existing = await chrome.scripting.getRegisteredContentScripts({
        ids,
      });
      if (!granted) {
        if (existing.length)
          await chrome.scripting.unregisterContentScripts({
            ids: existing.map((s) => s.id),
          });
        return;
      }
      const matches = [
        "https://shopping.naver.com/popup/reviews/form*",
        "https://shopping.naver.com/popup/reviews/monthly-form*",
      ];
      const scripts = [
        {
          id: ids[0],
          matches,
          js: ["resume-draft.js"],
          runAt: "document_start",
          world: "MAIN",
          persistAcrossSessions: true,
        },
        {
          id: ids[1],
          matches,
          js: ["permission-guard.js"],
          runAt: "document_start",
          persistAcrossSessions: true,
        },
      ];
      for (const script of scripts) {
        if (existing.some((s) => s.id === script.id))
          await chrome.scripting.updateContentScripts([script]);
        else await chrome.scripting.registerContentScripts([script]);
      }
    });
  return queue;
}
export async function revokePageHooks() {
  await syncSiteScripts();
  // Existing isolated-world listeners remain reachable after host revocation.
  for (const tab of await chrome.tabs.query({})) {
    if (
      !/^https:\/\/shopping\.naver\.com(?::\d+)?\/popup\/reviews\//.test(
        tab.url || "",
      )
    )
      continue;
    const granted = await chrome.permissions.contains({
      origins: [new URL(tab.url).origin + "/*"],
    });
    if (!granted)
      await chrome.tabs
        .sendMessage(tab.id, { type: "TALK_SITE_REVOKED" })
        .catch(() => {});
  }
}
