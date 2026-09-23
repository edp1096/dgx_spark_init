// Persist ownership across MV3 restarts. DOM refs and adapter caches remain ephemeral.
const sessions = new Map();
let owners = {};
const ready = chrome.storage.session.get("browserOwners").then((v) => {
  owners = v.browserOwners || {};
});
const save = () => chrome.storage.session.set({ browserOwners: owners });
chrome.tabs.onRemoved.addListener((id) => {
  ready.then(async () => {
    delete owners[id];
    await save();
  });
});
export async function sessionFor(id) {
  await ready;
  if (typeof id !== "string" || !id || id.length > 200)
    throw Error("브라우저 세션이 필요합니다.");
  if (!sessions.has(id))
    sessions.set(id, {
      id,
      adapters: new Map(),
      adapter(name, create) {
        if (!this.adapters.has(name)) this.adapters.set(name, create());
        return this.adapters.get(name);
      },
      refs: new Map(),
      async claim(tabId, created = false) {
        const existing = owners[tabId];
        if (existing && existing.id !== id)
          throw Error(
            "다른 대화가 사용하는 탭입니다. 해당 대화에서 release 또는 stop 하세요.",
          );
        owners[tabId] = { id, created: existing?.created || created };
        await save();
      },
      owns(tabId) {
        return owners[tabId]?.id === id;
      },
      async release(tabId) {
        if (owners[tabId] && owners[tabId].id !== id)
          throw Error("이 세션에 연결된 탭이 아닙니다.");
        delete owners[tabId];
        this.refs.delete(tabId);
        for (const adapter of this.adapters.values())
          adapter.releaseTab?.(tabId);
        await save();
      },
      async stop() {
        const released = [];
        for (const [tabId, v] of Object.entries(owners))
          if (v.id === id) {
            released.push(Number(tabId));
            delete owners[tabId];
          }
        await save();
        sessions.delete(id);
        return released;
      },
      tabs() {
        return Object.entries(owners)
          .filter(([, v]) => v.id === id)
          .map(([id, v]) => ({ tab_id: Number(id), created: v.created }));
      },
    });
  return sessions.get(id);
}

export async function resetSessions() {
  await ready;
  owners = {};
  sessions.clear();
  await save();
}
