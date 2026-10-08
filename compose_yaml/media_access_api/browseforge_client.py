import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


MEDIA_PATTERN = re.compile(r"\.(?:m3u8|mpd|mp4|webm)(?:\?|$)", re.IGNORECASE)
BLOCKED_TITLES = (
    "just a moment", "attention required", "access denied",
    "잠시만 기다리십시오", "잠시만 기다려", "しばらくお待ちください",
)
_host_locks: dict[str, threading.Lock] = {}
_host_locks_guard = threading.Lock()


class BrowseForgeError(RuntimeError):
    pass


def media_candidates(page_state: dict) -> list[str]:
    values = [
        *(page_state.get("videos") or []),
        *(page_state.get("sources") or []),
        *(page_state.get("media") or []),
    ]
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or value.startswith("blob:") or not MEDIA_PATTERN.search(value):
            continue
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def blocked_page(page_state: dict) -> bool:
    title = str(page_state.get("title") or "").lower()
    body = str(page_state.get("body") or "").lower()
    verification_markers = (
        "performing security verification", "보안 확인 수행 중",
        "セキュリティ検証を実行中", "checking if the site connection is secure",
    )
    cloudflare_challenge = "ray id:" in body and "cloudflare" in body
    return (
        any(marker in title for marker in BLOCKED_TITLES)
        or any(marker in body for marker in verification_markers)
        or cloudflare_challenge
    )


def profile_name(host: str, runtime_id: str) -> str:
    safe_host = re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-")[:60]
    runtime_suffix = "chromium" if runtime_id == "browseforge-chromium" else re.sub(r"[^a-z0-9]+", "-", runtime_id)
    return f"media-{safe_host}-{runtime_suffix}"


def host_lock(host: str) -> threading.Lock:
    with _host_locks_guard:
        return _host_locks.setdefault(host, threading.Lock())


class BrowseForgeClient:
    def __init__(self):
        self.base_url = os.getenv("BROWSEFORGE_API_URL", "").rstrip("/")
        self.token_file = Path(os.getenv(
            "BROWSEFORGE_TOKEN_FILE", "/data/browseforge/data/.api-token"
        ))
        self.runtime_id = os.getenv("BROWSEFORGE_RUNTIME", "browseforge-chromium")
        self.timeout = int(os.getenv("BROWSEFORGE_TIMEOUT_SECONDS", "45"))

    @property
    def configured(self) -> bool:
        return bool(self.base_url) and self.token_file.is_file()

    def _token(self) -> str:
        try:
            token = self.token_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise BrowseForgeError(f"token unavailable: {exc}") from exc
        if not token:
            raise BrowseForgeError("token is empty")
        return token

    def request(self, method: str, path: str, payload=None, timeout: int | None = None):
        if not self.configured:
            raise BrowseForgeError("service is not configured")
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token()}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=timeout or self.timeout) as response:
                raw = response.read()
                if not raw:
                    return None
                result = json.loads(raw.decode("utf-8"))
        except HTTPError as exc:
            try:
                error = json.loads(exc.read().decode("utf-8")).get("error", {})
                message = error.get("message") or str(exc)
            except (OSError, ValueError):
                message = str(exc)
            raise BrowseForgeError(message) from exc
        except (OSError, URLError, ValueError) as exc:
            raise BrowseForgeError(str(exc)) from exc
        return result.get("data", result)

    def find_or_create_profile(self, host: str) -> str:
        expected_name = profile_name(host, self.runtime_id)
        profiles = self.request("GET", "/api/profiles") or []
        for profile in profiles:
            if profile.get("runtime_id") != self.runtime_id:
                continue
            if profile.get("name") == expected_name or host in (profile.get("tags") or []):
                return profile["id"]
        profile = self.request("POST", "/api/profiles", {
            "name": expected_name,
            "runtime_id": self.runtime_id,
            "group": "media-access",
            "tags": [host],
        })
        return profile["id"]

    def start_session(self, profile_id: str) -> str:
        sessions = self.request("GET", "/api/sessions") or []
        for session in sessions:
            if session.get("profile_id") == profile_id:
                return session["session_id"]
        session = self.request("POST", "/api/sessions", {"profile_id": profile_id}, timeout=90)
        return session["session_id"]

    def navigate(self, session_id: str, url: str):
        error = None
        for attempt in range(3):
            try:
                return self.request("POST", f"/api/sessions/{session_id}/navigate", {
                    "url": url,
                    "wait_until": "domcontentloaded",
                }, timeout=self.timeout + 15)
            except BrowseForgeError as exc:
                error = exc
                if attempt < 2:
                    time.sleep(attempt + 1)
        raise error or BrowseForgeError("navigation failed")

    def page_state(self, session_id: str) -> dict:
        script = r'''JSON.stringify({
          title: document.title,
          url: location.href,
          body: document.body ? document.body.innerText.slice(0, 500) : "",
          userAgent: navigator.userAgent,
          videos: Array.from(document.querySelectorAll("video")).map(v => v.currentSrc || v.src).filter(Boolean),
          sources: Array.from(document.querySelectorAll("video source")).map(s => s.src).filter(Boolean),
          iframes: Array.from(document.querySelectorAll("iframe[src]")).map(f => f.src).filter(Boolean),
          servers: Array.from(document.querySelectorAll(".btn-server")).map((e, index) => ({
            index, name: (e.textContent || "").trim(), link: e.dataset.link || ""
          })),
          parts: Array.from(document.querySelectorAll(".btn-cd")).map((button, index) => {
            const group = document.querySelectorAll(".cd-server")[index];
            return {
              id: String(index + 1),
              label: (button.textContent || String(index + 1)).trim(),
              sources: group ? Array.from(group.querySelectorAll(".btn-server")).map((e, sourceIndex) => ({
                index: sourceIndex, name: (e.textContent || "").trim(), link: e.dataset.link || ""
              })) : []
            };
          }),
          media: performance.getEntriesByType("resource").map(e => e.name).filter(u => /\.(m3u8|mpd|mp4|webm)(\?|$)/i.test(u)).slice(-100)
        })'''
        encoded = self.request("POST", f"/api/sessions/{session_id}/eval", {"script": script})
        try:
            return json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise BrowseForgeError("invalid page state") from exc

    def wait_for_access(self, session_id: str, attempts: int = 12) -> dict:
        """Wait for an in-browser verification page to finish in the same profile."""
        state = {}
        for attempt in range(max(1, attempts)):
            state = self.page_state(session_id)
            if not blocked_page(state):
                return state
            if attempt + 1 < attempts:
                time.sleep(1)
        return state

    def options(self, url: str, adapter=None) -> dict:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise BrowseForgeError("URL host is missing")
        with host_lock(host):
            profile_id = self.find_or_create_profile(host)
            session_id = self.start_session(profile_id)
            parsed = urlparse(url)
            try:
                try:
                    self.navigate(session_id, f"{parsed.scheme}://{parsed.netloc}/")
                    root_state = self.wait_for_access(session_id)
                    if blocked_page(root_state):
                        raise BrowseForgeError(
                            f"access verification blocked ({root_state.get('title') or 'verification page'})"
                        )
                except BrowseForgeError:
                    # The detail URL still gets one independent chance; some
                    # sites protect only one of the two routes.
                    pass
                self.navigate(session_id, url)
                state = self.wait_for_access(session_id)
                if blocked_page(state):
                    raise BrowseForgeError(f"access verification blocked ({state.get('title') or 'verification page'})")
                return adapter.browseforge_options(state) if adapter else {"site": "generic", "parts": []}
            finally:
                try:
                    self.request("DELETE", f"/api/sessions/{session_id}", timeout=30)
                except BrowseForgeError:
                    pass

    def resolve(self, url: str, adapter=None, selection: dict | None = None) -> tuple[list[str], list[dict], dict]:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise BrowseForgeError("URL host is missing")
        with host_lock(host):
            profile_id = self.find_or_create_profile(host)
            session_id = self.start_session(profile_id)
            parsed = urlparse(url)
            root_url = f"{parsed.scheme}://{parsed.netloc}/"
            try:
                try:
                    self.navigate(session_id, root_url)
                except BrowseForgeError:
                    pass
                self.navigate(session_id, url)
                state = {}
                candidates = []
                provider_attempts = adapter.browseforge_attempts if adapter else 1
                polls_per_provider = max(4, 20 // provider_attempts)
                for provider_attempt in range(provider_attempts):
                    if adapter:
                        prepare_script = adapter.browseforge_prepare_script()
                        if prepare_script:
                            try:
                                prepared = self.request(
                                    "POST",
                                    f"/api/sessions/{session_id}/eval",
                                    {"script": prepare_script},
                                )
                                if prepared is False:
                                    break
                            except BrowseForgeError:
                                pass
                    for poll in range(polls_per_provider):
                        state = self.page_state(session_id)
                        if blocked_page(state):
                            if provider_attempt * polls_per_provider + poll >= 7:
                                raise BrowseForgeError(f"access verification blocked ({state.get('title') or 'verification page'})")
                        candidates = media_candidates(state)
                        if adapter:
                            if selection:
                                # Explicit part/source choices must not be
                                # replaced by a resource from the page's
                                # initially active player.
                                candidates = []
                            candidates = [
                                candidate for candidate in candidates
                                if adapter.browseforge_accept_candidate(candidate)
                            ]
                            headers = {
                                "User-Agent": state.get("userAgent") or "Mozilla/5.0",
                                "Referer": state.get("url") or url,
                            }
                            candidates.extend(
                                adapter.browseforge_extra_candidates(state, headers, self.timeout, selection)
                            )
                        if candidates:
                            break
                        if poll == 2 and not (adapter and adapter.browseforge_prepare_script()):
                            try:
                                self.request("POST", f"/api/sessions/{session_id}/eval", {
                                    "script": "(() => { const v=document.querySelector('video'); if(v){v.muted=true;v.play().catch(()=>{});} return true; })()"
                                })
                            except BrowseForgeError:
                                pass
                        time.sleep(1)
                    if candidates:
                        break
                    if provider_attempt + 1 < provider_attempts:
                        try:
                            self.request(
                                "POST",
                                f"/api/sessions/{session_id}/eval",
                                {"script": "performance.clearResourceTimings(); true"},
                            )
                        except BrowseForgeError:
                            pass
                if not candidates:
                    raise BrowseForgeError("no playable media was observed")
                cookies = self.request("GET", f"/api/sessions/{session_id}/cookies")
                headers = {
                    "User-Agent": state.get("userAgent") or "Mozilla/5.0",
                    "Referer": state.get("url") or url,
                }
                return candidates, cookies, headers
            finally:
                try:
                    self.request("DELETE", f"/api/sessions/{session_id}", timeout=30)
                except BrowseForgeError:
                    pass
