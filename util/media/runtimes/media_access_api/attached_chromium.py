"""Launch Chromium normally, then attach Playwright over a fixed CDP port.

Playwright's normal launcher adds automation command-line switches.  Starting
Chromium ourselves keeps navigator.webdriver false while CDP still provides
DOM and response access after launch.  Interactive Cloudflare widgets are
clicked through X11, not through a synthetic DOM click.
"""

import asyncio
import json
import os
import random
import re
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlopen


EXECUTABLE = os.getenv("DIRECT_CHROMIUM_EXECUTABLE", "/opt/direct-chromium/chrome")
DISPLAY = os.getenv("DIRECT_CHROMIUM_DISPLAY", ":98")
CDP_HOST = "127.0.0.1"
CDP_PORT = int(os.getenv("DIRECT_CHROMIUM_CDP_PORT", "19281"))
PROFILE_ROOT = Path(os.getenv("DIRECT_CHROMIUM_PROFILE_DIR", "/data/direct-chromium/profiles"))
X11_COMMAND = os.getenv("DIRECT_CHROMIUM_X11_COMMAND", "/usr/bin/xdotool")
SCREEN_WIDTH = int(os.getenv("DIRECT_CHROMIUM_SCREEN_WIDTH", "1365"))
SCREEN_HEIGHT = int(os.getenv("DIRECT_CHROMIUM_SCREEN_HEIGHT", "768"))
START_TIMEOUT = int(os.getenv("DIRECT_CHROMIUM_START_TIMEOUT_SECONDS", "20"))
CHALLENGE_TIMEOUT = int(os.getenv("DIRECT_CHROMIUM_CHALLENGE_TIMEOUT_SECONDS", "60"))
BLOCKED_MARKERS = (
    "just a moment",
    "attention required",
    "performing security verification",
    "checking if the site connection is secure",
    "잠시만 기다리십시오",
    "보안 확인 수행 중",
    "しばらくお待ちください",
)

_browser_lock = asyncio.Lock()


def profile_path(host_key: str) -> Path:
    safe = re.sub(r"[^a-z0-9._-]+", "-", host_key.casefold()).strip("-")[:80]
    return PROFILE_ROOT / f"{safe or 'generic'}.profile"


def _remove_stale_profile_locks(profile: Path) -> None:
    for name in ("SingletonLock", "SingletonCookie", "SingletonSocket"):
        target = profile / name
        if target.exists() or target.is_symlink():
            target.unlink(missing_ok=True)


async def _wait_for_cdp(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + START_TIMEOUT
    endpoint = f"http://{CDP_HOST}:{CDP_PORT}/json/version"
    last_error = "CDP endpoint unavailable"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"direct Chromium exited with code {process.returncode}")
        try:
            await asyncio.to_thread(lambda: urlopen(endpoint, timeout=1).read())
            return
        except OSError as exc:
            last_error = str(exc)
        await asyncio.sleep(0.2)
    raise RuntimeError(last_error)


async def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        await asyncio.to_thread(process.wait, 5)
    except subprocess.TimeoutExpired:
        process.kill()
        await asyncio.to_thread(process.wait)


@asynccontextmanager
async def attached_chromium(playwright, host_key: str):
    """Yield the persistent Chromium context from a manually launched browser."""
    async with _browser_lock:
        profile = profile_path(host_key)
        profile.mkdir(parents=True, exist_ok=True)
        _remove_stale_profile_locks(profile)
        environment = os.environ.copy()
        environment["DISPLAY"] = DISPLAY
        process = subprocess.Popen(
            [
                EXECUTABLE,
                "--no-sandbox",
                "--password-store=basic",
                f"--user-data-dir={profile}",
                "--no-first-run",
                "--disable-default-apps",
                f"--window-size={SCREEN_WIDTH},{SCREEN_HEIGHT}",
                "--window-position=0,0",
                f"--remote-debugging-address={CDP_HOST}",
                f"--remote-debugging-port={CDP_PORT}",
                "about:blank",
            ],
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        browser = None
        try:
            await _wait_for_cdp(process)
            browser = await playwright.chromium.connect_over_cdp(
                f"http://{CDP_HOST}:{CDP_PORT}"
            )
            if not browser.contexts:
                raise RuntimeError("direct Chromium has no persistent context")
            yield browser.contexts[0]
        finally:
            if browser is not None:
                try:
                    await browser.close()
                except Exception:
                    pass
            await _stop_process(process)
            _remove_stale_profile_locks(profile)


def _ocr_words(image: bytes) -> list[dict]:
    completed = subprocess.run(
        ["tesseract", "stdin", "stdout", "--psm", "11", "tsv"],
        input=image,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=15,
        check=False,
    )
    if completed.returncode:
        return []
    words = []
    for line in completed.stdout.decode("utf-8", errors="replace").splitlines()[1:]:
        columns = line.split("\t", 11)
        if len(columns) != 12 or not columns[11].strip():
            continue
        try:
            left, top, width, height = map(int, columns[6:10])
        except ValueError:
            continue
        words.append({
            "text": columns[11].strip(),
            "key": re.sub(r"[^a-z0-9가-힣]+", "", columns[11].casefold()),
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        })
    return words


def _checkbox_target(words: list[dict]) -> tuple[int, int] | None:
    for index, word in enumerate(words):
        nearby = " ".join(item["key"] for item in words[index:index + 5])
        if word["key"] == "verify" and "human" in nearby:
            # The Turnstile checkbox sits about 21 px left of the first word.
            # The former 38 px offset landed outside the checkbox on Camoufox.
            return max(10, word["left"] - 21), word["top"] + word["height"] // 2
        if "사람인지" in nearby and "확인" in nearby:
            return max(10, word["left"] - 21), word["top"] + word["height"] // 2
    return None


def _x11(*arguments: str) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    environment["DISPLAY"] = DISPLAY
    return subprocess.run(
        [X11_COMMAND, *arguments],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=5,
        check=False,
    )


def _human_x11_click(x: int, y: int) -> None:
    rng = random.SystemRandom()
    location = _x11("getmouselocation", "--shell")
    values = {}
    for line in location.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and value.strip().lstrip("-").isdigit():
            values[key] = int(value)
    start_x = values.get("X", rng.randint(SCREEN_WIDTH // 2, SCREEN_WIDTH - 50))
    start_y = values.get("Y", rng.randint(SCREEN_HEIGHT // 2, SCREEN_HEIGHT - 50))
    steps = rng.randint(45, 70)
    curve = rng.uniform(-80, 80)
    last = (start_x, start_y)
    for step in range(1, steps + 1):
        t = step / steps
        bend = 4 * t * (1 - t) * curve
        point = (
            round(start_x + (x - start_x) * t + bend),
            round(start_y + (y - start_y) * t - bend * 0.35),
        )
        if point != last:
            _x11("mousemove", "--sync", str(point[0]), str(point[1]))
            last = point
            time.sleep(rng.uniform(0.012, 0.032))
    time.sleep(rng.uniform(0.25, 0.65))
    _x11("mousedown", "1")
    time.sleep(rng.uniform(0.08, 0.18))
    _x11("mouseup", "1")


async def _click_visible_checkbox(page) -> bool:
    words = await asyncio.to_thread(_ocr_words, await page.screenshot(type="png"))
    target = _checkbox_target(words)
    if not target:
        return False
    geometry = await page.evaluate(
        """() => ({
          x: window.screenX + (window.outerWidth - window.innerWidth) / 2,
          y: window.screenY + window.outerHeight - window.innerHeight,
          dpr: window.devicePixelRatio || 1
        })"""
    )
    x = round((float(geometry["x"]) + target[0]) * float(geometry["dpr"]))
    y = round((float(geometry["y"]) + target[1]) * float(geometry["dpr"]))
    await asyncio.sleep(random.SystemRandom().uniform(3.0, 7.0))
    await asyncio.to_thread(_human_x11_click, x, y)
    return True


async def page_is_blocked(page) -> bool:
    try:
        state = await page.evaluate(
            """() => ({
              title: document.title || '',
              body: (document.body?.innerText || '').slice(0, 1200)
            })"""
        )
    except Exception:
        return True
    combined = f"{state['title']}\n{state['body']}".casefold()
    return (
        any(marker in combined for marker in BLOCKED_MARKERS)
        or ("ray id:" in combined and "cloudflare" in combined)
    )


async def wait_for_access(page, timeout_seconds: int = CHALLENGE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout_seconds
    clicks = 0
    next_click_check = time.monotonic() + 4
    while time.monotonic() < deadline:
        if not await page_is_blocked(page):
            await page.wait_for_timeout(1500)
            if not await page_is_blocked(page):
                return
        now = time.monotonic()
        if clicks < 2 and now >= next_click_check:
            if await _click_visible_checkbox(page):
                clicks += 1
                next_click_check = time.monotonic() + 12
            else:
                next_click_check = now + 3
        await page.wait_for_timeout(500)
    raise RuntimeError("direct Chromium access verification timed out")


async def page_state(page) -> dict:
    encoded = await page.evaluate(
        r'''() => JSON.stringify({
          title: document.title,
          url: location.href,
          body: document.body ? document.body.innerText.slice(0, 500) : "",
          userAgent: navigator.userAgent,
          webdriver: navigator.webdriver,
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
    )
    return json.loads(encoded)
