import json
import os
import subprocess
import threading
from urllib.parse import urlparse

from browseforge_client import media_candidates


RESULT_PREFIX = "CAMOUFOX_RESULT="
_host_locks: dict[str, threading.Lock] = {}
_host_locks_guard = threading.Lock()


class CamoufoxError(RuntimeError):
    pass


def host_lock(host: str) -> threading.Lock:
    with _host_locks_guard:
        return _host_locks.setdefault(host, threading.Lock())


class CamoufoxClient:
    def __init__(self):
        self.python = os.getenv("CAMOUFOX_WORKER_PYTHON", "/opt/camoufox-venv/bin/python")
        self.worker = os.getenv("CAMOUFOX_WORKER_SCRIPT", "/app/camoufox_worker.py")
        self.xvfb_run = os.getenv("CAMOUFOX_XVFB_RUN", "/usr/bin/xvfb-run")
        self.timeout = int(os.getenv("CAMOUFOX_WORKER_TIMEOUT_SECONDS", "180"))

    @property
    def configured(self) -> bool:
        return (
            os.path.isfile(self.python)
            and os.path.isfile(self.worker)
            and os.path.isfile(self.xvfb_run)
        )

    def inspect(self, url: str) -> dict:
        if not self.configured:
            raise CamoufoxError("Camoufox worker is not configured")
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise CamoufoxError("URL host is missing")
        with host_lock(host):
            try:
                completed = subprocess.run(
                    [
                        self.xvfb_run,
                        "-a",
                        "-s",
                        "-screen 0 1920x1080x24 -nolisten tcp +extension GLX +render",
                        self.python,
                        self.worker,
                    ],
                    input=json.dumps({"action": "inspect", "url": url}),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise CamoufoxError(str(exc)) from exc
        payload = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(RESULT_PREFIX):
                try:
                    payload = json.loads(line.removeprefix(RESULT_PREFIX))
                except ValueError as exc:
                    raise CamoufoxError("Camoufox worker returned invalid JSON") from exc
                break
        if not payload:
            detail = completed.stderr.strip().splitlines()[-1:] or ["no result"]
            raise CamoufoxError(f"Camoufox worker failed: {detail[0]}")
        if not payload.get("ok"):
            raise CamoufoxError(str(payload.get("error") or "Camoufox worker failed"))
        return payload.get("data") or {}

    def options(self, url: str, adapter=None) -> dict:
        state = self.inspect(url)
        return adapter.browseforge_options(state) if adapter else {"site": "generic", "parts": []}

    def resolve(self, url: str, adapter=None, selection: dict | None = None):
        state = self.inspect(url)
        candidates = [] if selection else media_candidates(state)
        for request in state.get("requests") or []:
            candidate = request.get("url") if isinstance(request, dict) else None
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        if adapter:
            candidates = [candidate for candidate in candidates if adapter.browseforge_accept_candidate(candidate)]
        headers = {
            "User-Agent": state.get("userAgent") or "Mozilla/5.0",
            "Referer": state.get("url") or url,
        }
        if adapter:
            candidates.extend(adapter.browseforge_extra_candidates(state, headers, self.timeout, selection))
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise CamoufoxError("no playable media was observed")
        return candidates, state.get("cookies") or [], headers
