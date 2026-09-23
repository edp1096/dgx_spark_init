// Revocation signal only. This script never grants access or reads page data.
(() => {
  if (globalThis.__talkPermissionGuard) return;
  globalThis.__talkPermissionGuard = true;
  chrome.runtime.onMessage.addListener((message, sender, reply) => {
    if (sender.id !== chrome.runtime.id || message.type !== "TALK_SITE_REVOKED")
      return;
    document.dispatchEvent(new Event("sparktalk:revoke-resume"));
    reply({ ok: true });
  });
})();
