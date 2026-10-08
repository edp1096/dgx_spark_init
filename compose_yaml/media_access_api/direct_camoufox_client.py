import json
import os
import re
import subprocess
import threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

from yt_dlp.cookies import extract_cookies_from_browser


RESULT_PREFIX = "DIRECT_CAMOUFOX_RESULT:"
MEDIA_PATTERN = re.compile(
    r"https?://[^\s'\"<>]+?\.(?:m3u8|mpd|mp4|webm)(?:\?[^\s'\"<>]*)?", re.I
)
_host_locks: dict[str, threading.Lock] = {}
_host_locks_guard = threading.Lock()


class DirectCamoufoxError(RuntimeError):
    pass


def host_lock(host: str) -> threading.Lock:
    with _host_locks_guard:
        return _host_locks.setdefault(host, threading.Lock())


class SupJavSourceParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title = ""
        self._title_depth = 0
        self.part_labels = []
        self.server_groups = []
        self.top_servers = []
        self._group_depth = 0
        self._capture = None
        self._capture_text = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set(attributes.get("class", "").split())
        if tag == "title":
            self._title_depth += 1
        if tag == "div":
            if self._group_depth:
                self._group_depth += 1
            elif "cd-server" in classes:
                self.server_groups.append([])
                self._group_depth = 1
        if tag in {"a", "button"} and "btn-cd" in classes:
            self._capture = ("part", None)
            self._capture_text = []
        elif tag in {"a", "button"} and "btn-server" in classes:
            server = {"index": 0, "name": "", "link": attributes.get("data-link", "")}
            target = self.server_groups[-1] if self._group_depth and self.server_groups else self.top_servers
            server["index"] = len(target)
            target.append(server)
            self._capture = ("server", server)
            self._capture_text = []

    def handle_endtag(self, tag):
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag in {"a", "button"} and self._capture:
            text = " ".join("".join(self._capture_text).split())
            kind, value = self._capture
            if kind == "part":
                self.part_labels.append(text or str(len(self.part_labels) + 1))
            elif value is not None:
                value["name"] = text
            self._capture = None
            self._capture_text = []
        if tag == "div" and self._group_depth:
            self._group_depth -= 1

    def handle_data(self, data):
        if self._title_depth:
            self.title += data
        if self._capture:
            self._capture_text.append(data)


def source_state(state: dict) -> dict:
    source = str(state.get("source") or "")
    parser = SupJavSourceParser()
    parser.feed(source)
    parts = []
    for index, group in enumerate(parser.server_groups):
        identifier = str(index + 1)
        parts.append({
            "id": identifier,
            "label": parser.part_labels[index] if index < len(parser.part_labels) else identifier,
            "sources": group,
        })
    return {
        **state,
        "title": " ".join(parser.title.split()) or state.get("title", ""),
        "body": "",
        "servers": parser.top_servers,
        "parts": parts,
        "videos": list(dict.fromkeys(MEDIA_PATTERN.findall(source))),
        "sources": [],
        "media": [],
    }


class DirectCamoufoxClient:
    def __init__(self):
        self.python = os.getenv("DIRECT_CAMOUFOX_WORKER_PYTHON", "/usr/bin/python3")
        self.worker = os.getenv("DIRECT_CAMOUFOX_WORKER_SCRIPT", "/app/direct_camoufox_worker.py")
        self.xvfb_run = os.getenv("DIRECT_CAMOUFOX_XVFB_RUN", "/usr/bin/xvfb-run")
        self.timeout = int(os.getenv("DIRECT_CAMOUFOX_WORKER_TIMEOUT_SECONDS", "150"))

    @property
    def configured(self) -> bool:
        return all(os.path.isfile(path) for path in (self.python, self.worker, self.xvfb_run))

    def inspect(self, url: str) -> dict:
        if not self.configured:
            raise DirectCamoufoxError("direct Camoufox worker is not configured")
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise DirectCamoufoxError("URL host is missing")
        with host_lock(host):
            try:
                completed = subprocess.run(
                    [
                        self.xvfb_run, "-a", "-s",
                        "-screen 0 1365x768x24 -nolisten tcp +extension GLX +render",
                        self.python, self.worker,
                    ],
                    input=json.dumps({"action": "inspect", "url": url}),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DirectCamoufoxError(str(exc)) from exc
        payload = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(RESULT_PREFIX):
                try:
                    payload = json.loads(line.removeprefix(RESULT_PREFIX))
                except ValueError as exc:
                    raise DirectCamoufoxError("direct Camoufox returned invalid JSON") from exc
                break
        if not payload:
            detail = completed.stderr.strip().splitlines()[-1:] or ["no result"]
            raise DirectCamoufoxError(f"direct Camoufox failed: {detail[0]}")
        if not payload.get("ok"):
            raise DirectCamoufoxError(str(payload.get("error") or "direct Camoufox failed"))
        return source_state(payload.get("data") or {})

    def export_cookies(self, state: dict, destination: Path) -> Path:
        profile = state.get("profile")
        if not profile:
            raise DirectCamoufoxError("direct Camoufox profile is unavailable")
        try:
            jar = extract_cookies_from_browser("firefox", profile=profile)
            jar.save(str(destination), ignore_discard=True, ignore_expires=True)
        except Exception as exc:
            raise DirectCamoufoxError(f"cannot export Camoufox cookies: {exc}") from exc
        return destination

    def options(self, url: str, adapter) -> dict:
        return adapter.browseforge_options(self.inspect(url))

    def resolve(self, url: str, adapter, destination: Path, selection: dict | None = None):
        state = self.inspect(url)
        headers = {
            "User-Agent": state.get("userAgent") or "Mozilla/5.0",
            "Referer": state.get("url") or url,
        }
        candidates = list(state.get("videos") or [])
        if selection:
            candidates = []
        candidates.extend(adapter.browseforge_extra_candidates(
            state, headers, max(1, self.timeout // 3), selection
        ))
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            raise DirectCamoufoxError("no playable media source was found in page source")
        return candidates, self.export_cookies(state, destination), headers
