#!/usr/bin/env python3
"""Drive a normally launched Camoufox only through X11 and the clipboard."""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Reuse the OCR and human pointer implementation with this worker's isolated
# xvfb-run display rather than the attached Chromium display.
os.environ["DIRECT_CHROMIUM_DISPLAY"] = os.environ.get("DISPLAY", ":98")
from attached_chromium import _checkbox_target, _human_x11_click, _ocr_words  # noqa: E402


RESULT_PREFIX = "DIRECT_CAMOUFOX_RESULT:"
EXECUTABLE = os.getenv(
    "DIRECT_CAMOUFOX_EXECUTABLE", "/opt/browseforge/browsers/camoufox/camoufox"
)
PROFILE_ROOT = Path(os.getenv("DIRECT_CAMOUFOX_PROFILE_DIR", "/data/direct-camoufox/profiles"))
X11_COMMAND = os.getenv("DIRECT_CAMOUFOX_X11_COMMAND", "/usr/bin/xdotool")
TIMEOUT_SECONDS = int(os.getenv("DIRECT_CAMOUFOX_TIMEOUT_SECONDS", "90"))
SCREEN_WIDTH = int(os.getenv("DIRECT_CAMOUFOX_SCREEN_WIDTH", "1365"))
SCREEN_HEIGHT = int(os.getenv("DIRECT_CAMOUFOX_SCREEN_HEIGHT", "768"))
BLOCKED_MARKERS = (
    "just a moment",
    "performing security verification",
    "verify you are human",
    "잠시만 기다리십시오",
    "보안 확인 수행 중",
)
ERROR_MARKERS = (
    "this site can’t be reached",
    "this site can't be reached",
    "the connection was reset",
    "secure connection failed",
    "pr_connect_reset_error",
    "server not found",
)


def profile_path(url: str) -> Path:
    host = (urlparse(url).hostname or "unknown").lower()
    key = hashlib.sha256(host.encode()).hexdigest()[:24]
    return PROFILE_ROOT / f"{key}.profile"


def x11(*arguments: str, timeout: float = 8) -> subprocess.CompletedProcess:
    return subprocess.run(
        [X11_COMMAND, *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def find_window() -> str | None:
    result = x11("search", "--onlyvisible", "--name", ".*")
    if result.returncode:
        return None
    windows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    # Firefox/Camoufox exposes additional visible, titleless X11 utility
    # windows. The last search result is commonly one of those rather than the
    # actual browser chrome, which made successful pages look titleless.
    for window in reversed(windows):
        if window_title(window):
            return window
    return windows[-1] if windows else None


def window_title(window: str) -> str:
    result = x11("getwindowname", window)
    return result.stdout.strip() if result.returncode == 0 else ""


def screenshot() -> bytes:
    completed = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "x11grab", "-video_size", f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
            "-i", os.environ["DISPLAY"], "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    if completed.returncode or not completed.stdout:
        raise RuntimeError(completed.stderr.decode(errors="replace").strip() or "screenshot failed")
    return completed.stdout


def copy_page_source(window: str, url: str) -> str:
    # Xvfb intentionally has no window manager, so EWMH windowactivate cannot
    # work. XSetInputFocus (xdotool windowfocus) does not require one.
    x11("windowfocus", "--sync", window)
    # Camoufox does not consistently honor Ctrl+U while a video is loading.
    # Navigating the address bar is deterministic and preserves the exact
    # response source Firefox received through its real TLS connection.
    x11("key", "--clearmodifiers", "ctrl+l")
    x11("type", "--clearmodifiers", "--delay", "1", f"view-source:{url}")
    x11("key", "--clearmodifiers", "Return")
    time.sleep(5)
    source_window = find_window() or window
    x11("windowfocus", "--sync", source_window)
    # Firefox keeps keyboard focus in the address bar after navigation. Put
    # focus into the rendered source document before selecting its contents.
    x11("mousemove", str(SCREEN_WIDTH // 2), str(SCREEN_HEIGHT // 2))
    x11("click", "1")
    time.sleep(0.5)
    x11("key", "--clearmodifiers", "ctrl+a")
    time.sleep(0.5)
    x11("key", "--clearmodifiers", "ctrl+c")
    time.sleep(1)
    completed = subprocess.run(
        ["xclip", "-selection", "clipboard", "-o"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=10,
        check=False,
    )
    source = completed.stdout.decode("utf-8", errors="replace")
    lowered = source.casefold()
    looks_like_html = len(source) >= 500 and any(
        marker in lowered for marker in ("<!doctype html", "<html", "<head", "<body")
    )
    if completed.returncode or not looks_like_html:
        preview = " ".join(source[:160].split())
        stderr = " ".join(completed.stderr.decode(errors="replace")[:160].split())
        raise RuntimeError(
            f"Camoufox page source was not copied (length={len(source)}, "
            f"preview={preview!r}, xclip={stderr!r})"
        )
    if any(marker in lowered for marker in ERROR_MARKERS):
        raise RuntimeError("Camoufox copied a browser error page")
    return source


def close_browser(process: subprocess.Popen, window: str | None) -> None:
    if window:
        x11("windowclose", window)
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def user_agent() -> str:
    completed = subprocess.run(
        [EXECUTABLE, "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=10,
        check=False,
    )
    # Sandbox diagnostics may prefix the version line with a numeric PID such
    # as "[10271]". Only accept the version following the Camoufox product
    # name; using the first number produced impossible Firefox/10271 UAs.
    match = re.search(r"Camoufox(?:\s+Camoufox)?\s+(\d+)(?:\.\d+)*", completed.stdout, re.I)
    major = match.group(1) if match else "152"
    return f"Mozilla/5.0 (X11; Linux x86_64; rv:{major}.0) Gecko/20100101 Firefox/{major}.0"


def inspect(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an http(s) URL")
    profile = profile_path(url)
    profile.mkdir(parents=True, exist_ok=True)
    for name in ("parent.lock", ".parentlock", "lock"):
        target = profile / name
        if target.exists() or target.is_symlink():
            target.unlink(missing_ok=True)
    process = subprocess.Popen(
        [
            EXECUTABLE, "-profile", str(profile),
            "-width", str(SCREEN_WIDTH), "-height", str(SCREEN_HEIGHT), url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    started = time.monotonic()
    window = None
    clicks = []
    last_click = 0.0
    stable_since = None
    last_text = ""
    connection_retries = 0
    try:
        while time.monotonic() - started < TIMEOUT_SECONDS:
            if process.poll() is not None:
                raise RuntimeError(f"Camoufox exited with code {process.returncode}")
            # During startup only Xvfb's titleless root/utility window may be
            # visible. Re-resolve every pass so it is replaced by Camoufox's
            # titled browser window as soon as that window appears.
            window = find_window() or window
            if not window:
                time.sleep(0.5)
                continue
            title = window_title(window)
            words = _ocr_words(screenshot())
            text = " ".join(word["text"] for word in words).casefold()
            last_text = text[-500:]
            connection_error = next((marker for marker in ERROR_MARKERS if marker in text), None)
            if connection_error:
                if connection_retries >= 3:
                    raise RuntimeError(connection_error)
                connection_retries += 1
                x11("windowfocus", "--sync", window)
                time.sleep((2, 5, 10)[connection_retries - 1])
                x11("key", "--clearmodifiers", "ctrl+r")
                stable_since = None
                time.sleep(3)
                continue
            target = _checkbox_target(words)
            now = time.monotonic()
            if target and now - last_click >= 12 and len(clicks) < 2:
                time.sleep(random.SystemRandom().uniform(3.0, 7.0))
                x11("windowfocus", "--sync", window)
                _human_x11_click(*target)
                clicks.append({"x": target[0], "y": target[1], "elapsed": round(now - started, 2)})
                last_click = time.monotonic()
                stable_since = None
                time.sleep(2)
                continue
            blocked = any(marker in f"{title}\n{text}".casefold() for marker in BLOCKED_MARKERS)
            if title and not blocked and not target:
                # Video pages prefix the window title with a changing load
                # percentage. Treat the normal page state, not an identical
                # title string, as the stability signal.
                if stable_since is None:
                    stable_since = now
                elif now - stable_since >= 4:
                    source = copy_page_source(window, url)
                    return {
                        "url": url,
                        "title": title,
                        "source": source,
                        "profile": str(profile),
                        "userAgent": user_agent(),
                        "clicks": clicks,
                    }
            else:
                stable_since = None
            time.sleep(1)
        detail = f" ({last_text[:180]})" if last_text else ""
        raise RuntimeError(f"direct Camoufox verification timed out{detail}")
    finally:
        close_browser(process, find_window() or window)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if payload.get("action") != "inspect":
            raise ValueError("unsupported action")
        result = {"ok": True, "data": inspect(str(payload.get("url") or ""))}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
