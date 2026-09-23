import { syncSiteScripts, revokePageHooks } from "./site-permissions.js";
import { BrowserDriver } from "./browser-driver.js";
import { sessionFor, resetSessions } from "./browser-sessions.js";
import { naverAdapter, createNaverState } from "./naver-adapter.js";
import { browserTool } from "./browser-tools.js";
let socket,
  connecting = false,
  heartbeat,
  authenticated = false;
// One queue across reconnects: an old in-flight Chrome command must settle first.
let serial = Promise.resolve();
const pending = new Map();
function abortConnection(ws) {
  for (const request of pending.values())
    if (request.ws === ws) request.cancelled = true;
}
async function connect() {
  if (
    connecting ||
    socket?.readyState === WebSocket.OPEN ||
    socket?.readyState === WebSocket.CONNECTING
  )
    return;
  connecting = true;
  try {
    const { server, token } = await chrome.storage.local.get([
      "server",
      "token",
    ]);
    if (!server || !token) return;
    const u = new URL("/api/browser/connect", server);
    u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(u);
    socket = ws;
    ws.onopen = () => {
      if (socket !== ws) {
        ws.close();
        return;
      }
      ws.send(JSON.stringify({ token, protocol: 14 }));
      clearInterval(heartbeat);
      heartbeat = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send("{}");
      }, 20000);
    };
    ws.onmessage = (e) => {
      if (socket !== ws) return;
      let cmd;
      try {
        cmd = JSON.parse(e.data);
      } catch {
        ws.close();
        return;
      }
      if (cmd.ready) {
        authenticated = true;
        return;
      }
      if (cmd.cancel) {
        const request = pending.get(cmd.cancel);
        if (request?.ws === ws) request.cancelled = true;
        return;
      }
      if (!authenticated || typeof cmd.id !== "string" || pending.has(cmd.id))
        return;
      const request = { ws, cancelled: false };
      pending.set(cmd.id, request);
      serial = serial
        .then(async () => {
          let result, driver;
          const check = () => {
            if (
              request.cancelled ||
              socket !== ws ||
              !authenticated ||
              ws.readyState !== WebSocket.OPEN ||
              (cmd.expires && Date.now() > cmd.expires)
            )
              throw Error("브라우저 작업이 취소되거나 연결이 끊겼습니다.");
          };
          try {
            check();
            const session = await sessionFor(cmd.session_id || "legacy");
            if (cmd.action !== "browser") await syncSiteScripts();
            check();
            driver = new BrowserDriver(session, check);
            result = await (cmd.action === "browser"
              ? browserTool(cmd, driver)
              : naverAdapter(
                  session.adapter("naver", createNaverState),
                  driver,
                )(cmd));
            result = {
              ...result,
              effect_state:
                result.effect_state ||
                (driver.effect === "none"
                  ? "none"
                  : result.ok
                    ? "committed"
                    : "unknown"),
            };
          } catch (err) {
            result = {
              ok: false,
              error: err.message,
              observation: err.observation,
              effect_state: driver?.effect || "none",
            };
          } finally {
            pending.delete(cmd.id);
          }
          if (socket === ws && ws.readyState === WebSocket.OPEN)
            ws.send(JSON.stringify({ id: cmd.id, result }));
        })
        .catch(() => {});
    };
    ws.onclose = () => {
      abortConnection(ws);
      if (socket !== ws) return;
      clearInterval(heartbeat);
      authenticated = false;
      socket = null;
    };
  } finally {
    connecting = false;
  }
}
chrome.alarms.create("bridge", { periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((a) => {
  if (a.name === "bridge") connect();
});
chrome.runtime.onStartup.addListener(connect);
chrome.runtime.onMessage.addListener((m, s, reply) => {
  if (
    s.id !== chrome.runtime.id ||
    !s.url?.startsWith(chrome.runtime.getURL(""))
  )
    return;
  if (m.type === "RELEASE_TABS") {
    for (const request of pending.values()) request.cancelled = true;
    serial = serial
      .then(async () => {
        await resetSessions();
        reply({ ok: true });
      })
      .catch((error) => reply({ ok: false, error: error.message }));
    return true;
  }
  if (m.type === "STATUS") {
    reply({
      connected: authenticated && socket?.readyState === WebSocket.OPEN,
    });
    return;
  }
  if (m.type === "CONNECT") {
    if (socket) {
      abortConnection(socket);
      socket.close();
    }
    socket = null;
    authenticated = false;
    clearInterval(heartbeat);
    connect().then(() => reply({ ok: true }));
    return true;
  }
});
connect();

chrome.permissions.onAdded.addListener(() => {
  syncSiteScripts().catch(console.error);
});
chrome.permissions.onRemoved.addListener(() => {
  for (const request of pending.values()) request.cancelled = true;
  revokePageHooks().catch(console.error);
});
syncSiteScripts().catch(console.error);
