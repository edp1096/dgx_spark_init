#!/usr/bin/env python3
"""Isolated Camoufox browser worker.

This process runs in a Playwright 1.60 virtual environment because the main
Media API intentionally tracks a newer Playwright release.  Exactly one JSON
request is accepted on stdin and one prefixed JSON result is emitted on stdout.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from camoufox import AsyncCamoufox
from camoufox.addons import DefaultAddons
import camoufox.utils as camoufox_utils
from camoufox.fingerprints import load_presets
from camoufox.webgl.sample import sample_webgl
from playwright_captcha import CaptchaType, ClickSolver, FrameworkType
from playwright_captcha.solvers.click.cloudflare.utils.dom_helpers import get_ready_checkbox
from playwright_captcha.solvers.click.common.shadow_root import search_shadow_root_iframes
from playwright_captcha.utils.camoufox_add_init_script.add_init_script import get_addon_path


RESULT_PREFIX = "CAMOUFOX_RESULT="
MEDIA_SUFFIXES = (".m3u8", ".mpd", ".mp4", ".webm", ".m4a", ".mp3", ".aac")
BLOCKED_MARKERS = (
    "just a moment",
    "attention required",
    "잠시만 기다리십시오",
    "잠시만 기다려",
    "しばらくお待ちください",
)
PROFILE_ROOT = Path(os.getenv("CAMOUFOX_PROFILE_DIR", "/data/camoufox/profiles"))
EXECUTABLE = os.getenv(
    "CAMOUFOX_EXECUTABLE", "/opt/browseforge/browsers/camoufox/camoufox"
)
TIMEOUT_MS = int(os.getenv("CAMOUFOX_TIMEOUT_SECONDS", "60")) * 1000
FIREFOX_VERSION = int(os.getenv("CAMOUFOX_FIREFOX_VERSION", "152"))
FINGERPRINT_OS = os.getenv("CAMOUFOX_FINGERPRINT_OS", "linux")
FINGERPRINT_TIMEZONE = os.getenv("CAMOUFOX_FINGERPRINT_TIMEZONE", "Asia/Seoul")
X11_CLICK_COMMAND = os.getenv("CAMOUFOX_X11_CLICK_COMMAND", "/usr/bin/xdotool")

logging.basicConfig(
    stream=sys.stderr,
    level=os.getenv("CAMOUFOX_LOG_LEVEL", "WARNING"),
    format="camoufox-worker %(levelname)s %(message)s",
)


def configure_bundled_runtime() -> None:
    """Keep Camoufox support files beside the explicitly bundled executable.

    Camoufox's Python helper otherwise resolves fonts through its mutable user
    cache and downloads the newest browser even when executable_path is given.
    The BrowseForge image already supplies a complete, pinned ARM64 bundle.
    """
    browser_dir = Path(EXECUTABLE).resolve().parent

    def bundled_path(name: str) -> str:
        return str(browser_dir / name)

    camoufox_utils.get_path = bundled_path


def profile_path(url: str) -> Path:
    host = (urlparse(url).hostname or "unknown").lower()
    key = hashlib.sha256(host.encode()).hexdigest()[:24]
    return PROFILE_ROOT / f"{key}.profile"


def fingerprint_identity(url: str) -> tuple[dict, dict]:
    """Return one compatible, stable Linux identity for a target hostname."""
    host = (urlparse(url).hostname or "unknown").lower()
    digest = hashlib.sha256(("camoufox-identity:" + host).encode()).digest()
    presets = load_presets(FIREFOX_VERSION) or {}
    candidates = []
    target_os = {"linux": "lin", "windows": "win", "macos": "mac"}.get(
        FINGERPRINT_OS, "lin"
    )
    for preset in presets.get("presets", {}).get(FINGERPRINT_OS, []):
        webgl = preset.get("webgl") or {}
        try:
            sample_webgl(
                target_os,
                webgl.get("unmaskedVendor"),
                webgl.get("unmaskedRenderer"),
            )
        except Exception:
            continue
        candidates.append(preset)
    if not candidates:
        raise RuntimeError(f"no compatible Camoufox {FINGERPRINT_OS} preset is available")
    preset = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
    config = {
        "forceScopeAccess": True,
        "timezone": FINGERPRINT_TIMEZONE,
        "fonts:spacing_seed": int.from_bytes(digest[8:12], "big") or 1,
        "audio:seed": int.from_bytes(digest[12:16], "big") or 1,
        "canvas:seed": int.from_bytes(digest[16:20], "big") or 1,
        "window.history.length": 1 + digest[20] % 5,
    }
    return preset, config


async def page_is_blocked(page) -> bool:
    try:
        title = (await page.title()).lower()
        body = (await page.locator("body").inner_text(timeout=3000)).lower()
    except Exception:
        return False
    return (
        any(marker in title or marker in body for marker in BLOCKED_MARKERS)
        or ("ray id:" in body and "cloudflare" in body)
        or "performing security verification" in body
        or "보안 확인 수행 중" in body
    )


async def click_interstitial_via_x11(page) -> dict:
    """Find the checkbox in Firefox, but click it through native X11 input."""
    if not os.getenv("DISPLAY"):
        raise RuntimeError("DISPLAY is not configured for external X11 input")
    if not os.path.isfile(X11_CLICK_COMMAND):
        raise RuntimeError(f"external X11 click command is missing: {X11_CLICK_COMMAND}")
    iframes = await search_shadow_root_iframes(
        framework=FrameworkType.CAMOUFOX,
        captcha_container=page,
        src_filter="https://challenges.cloudflare.com/cdn-cgi/challenge-platform/",
    )
    if not iframes:
        raise RuntimeError("Cloudflare iframe was not found for external X11 input")
    checkbox_data = await get_ready_checkbox(
        framework=FrameworkType.CAMOUFOX,
        iframes=iframes,
        delay=2,
        attempts=8,
    )
    if not checkbox_data:
        raise RuntimeError("Cloudflare checkbox was not ready for external X11 input")
    iframe, checkbox = checkbox_data
    box = await checkbox.bounding_box()
    if not box:
        raise RuntimeError("Cloudflare checkbox has no visible bounding box")
    try:
        iframe_box = await (await iframe.frame_element()).bounding_box()
    except Exception:
        iframe_box = None
    geometry = await page.evaluate(
        """() => ({
          x: window.mozInnerScreenX ?? (window.screenX + (window.outerWidth - window.innerWidth) / 2),
          y: window.mozInnerScreenY ?? (window.screenY + window.outerHeight - window.innerHeight),
          dpr: window.devicePixelRatio || 1,
          width: window.innerWidth,
          height: window.innerHeight,
          screenWidth: window.screen.width,
          screenHeight: window.screen.height
        })"""
    )
    # ElementHandle.bounding_box() is relative to the main-frame viewport,
    # including for elements found in child frames. Firefox exposes that
    # viewport's native screen origin through mozInnerScreenX/Y.
    dpr = float(geometry.get("dpr") or 1)
    origin_x = float(geometry["x"]) * dpr
    origin_y = float(geometry["y"]) * dpr
    target_x = round((float(geometry["x"]) + box["x"] + box["width"] / 2) * dpr)
    target_y = round((float(geometry["y"]) + box["y"] + box["height"] / 2) * dpr)
    await page.bring_to_front()
    rng = random.SystemRandom()
    screen_width = max(1, round(float(geometry["screenWidth"]) * dpr))
    screen_height = max(1, round(float(geometry["screenHeight"]) * dpr))

    async def x11(*arguments: str):
        completed = await asyncio.to_thread(
            subprocess.run,
            [X11_CLICK_COMMAND, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                "external X11 click failed: "
                + (completed.stderr.strip() or str(completed.returncode))
            )
        return completed

    def clamp_point(x: float, y: float) -> tuple[int, int]:
        return (
            min(screen_width - 1, max(1, round(x))),
            min(screen_height - 1, max(1, round(y))),
        )

    location = await x11("getmouselocation", "--shell")
    location_values = {}
    for line in location.stdout.splitlines():
        key, separator, value = line.partition("=")
        if separator and value.strip().lstrip("-").isdigit():
            location_values[key] = int(value)
    current = clamp_point(
        location_values.get("X", origin_x + geometry["width"] * dpr * rng.uniform(0.65, 0.9)),
        location_values.get("Y", origin_y + geometry["height"] * dpr * rng.uniform(0.65, 0.9)),
    )
    motion_points = 0

    async def move_curve(destination: tuple[int, int], steps: int) -> None:
        nonlocal current, motion_points
        start_x, start_y = current
        last_point = current
        end_x, end_y = destination
        distance = math.hypot(end_x - start_x, end_y - start_y)
        bend = min(120.0, max(18.0, distance * rng.uniform(0.12, 0.28)))
        normal_x = -(end_y - start_y) / max(distance, 1.0)
        normal_y = (end_x - start_x) / max(distance, 1.0)
        direction = rng.choice((-1.0, 1.0))
        control_1 = (
            start_x + (end_x - start_x) * rng.uniform(0.25, 0.4) + normal_x * bend * direction,
            start_y + (end_y - start_y) * rng.uniform(0.2, 0.45) + normal_y * bend * direction,
        )
        control_2 = (
            start_x + (end_x - start_x) * rng.uniform(0.62, 0.82) - normal_x * bend * direction * 0.35,
            start_y + (end_y - start_y) * rng.uniform(0.58, 0.8) - normal_y * bend * direction * 0.35,
        )
        for step in range(1, steps + 1):
            t = step / steps
            inverse = 1.0 - t
            x = (
                inverse ** 3 * start_x
                + 3 * inverse ** 2 * t * control_1[0]
                + 3 * inverse * t ** 2 * control_2[0]
                + t ** 3 * end_x
            )
            y = (
                inverse ** 3 * start_y
                + 3 * inverse ** 2 * t * control_1[1]
                + 3 * inverse * t ** 2 * control_2[1]
                + t ** 3 * end_y
            )
            point = clamp_point(x, y)
            if point == last_point:
                continue
            await x11("mousemove", "--sync", str(point[0]), str(point[1]))
            last_point = point
            motion_points += 1
            await asyncio.sleep(rng.uniform(0.012, 0.032))
        current = clamp_point(end_x, end_y)

    if iframe_box:
        area_left = origin_x + iframe_box["x"] * dpr
        area_top = origin_y + iframe_box["y"] * dpr
        area_width = iframe_box["width"] * dpr
        area_height = iframe_box["height"] * dpr
    else:
        area_left = target_x - 145
        area_top = target_y - 45
        area_width = 290
        area_height = 90

    started_at = time.monotonic()
    # Approach the widget along a curved path, then inspect several points in
    # its vicinity before committing to the checkbox itself.
    near = clamp_point(
        area_left + area_width * rng.uniform(0.2, 0.8),
        area_top + area_height * rng.uniform(0.15, 0.85),
    )
    await move_curve(near, rng.randint(24, 38))
    hover_count = rng.randint(3, 5)
    for _ in range(hover_count):
        hover = clamp_point(
            area_left + area_width * rng.uniform(0.08, 0.92),
            area_top + area_height * rng.uniform(0.1, 0.9),
        )
        await move_curve(hover, rng.randint(5, 10))
        await asyncio.sleep(rng.uniform(0.08, 0.28))

    overshoot = clamp_point(
        target_x + rng.choice((-1, 1)) * rng.uniform(5, 13),
        target_y + rng.choice((-1, 1)) * rng.uniform(3, 9),
    )
    await move_curve(overshoot, rng.randint(12, 20))
    await asyncio.sleep(rng.uniform(0.06, 0.18))
    await move_curve((target_x, target_y), rng.randint(4, 7))
    await asyncio.sleep(rng.uniform(0.18, 0.52))

    await x11("mousedown", "1")
    await asyncio.sleep(rng.uniform(0.08, 0.19))
    await x11("mouseup", "1")
    details = {
        "display": os.getenv("DISPLAY"),
        "x": target_x,
        "y": target_y,
        "dpr": dpr,
        "box": box,
        "iframe_box": iframe_box,
        "viewport": geometry,
        "hover_count": hover_count,
        "motion_points": motion_points,
        "motion_seconds": round(time.monotonic() - started_at, 3),
    }
    logging.info("external X11 checkbox click: %s", details)
    return details


async def solve_interstitial(page, solver) -> bool:
    if not await page_is_blocked(page):
        return True
    try:
        await click_interstitial_via_x11(page)
    except Exception as exc:
        logging.warning("external X11 click failed, using browser input fallback: %s", exc)
        try:
            await solver.solve_captcha(
                captcha_container=page,
                captcha_type=CaptchaType.CLOUDFLARE_INTERSTITIAL,
                solve_click_delay=10,
                wait_checkbox_attempts=8,
                wait_checkbox_delay=2,
                checkbox_click_attempts=3,
            )
        except Exception as fallback_exc:
            logging.warning("playwright-captcha fallback failed: %s", fallback_exc)
    # The iframe can disappear before the resulting full-page navigation ends.
    # Evaluate the page state rather than a widget-specific success node.
    for _ in range(30):
        if not await page_is_blocked(page):
            return True
        await page.wait_for_timeout(1000)
    return False


async def navigate_with_challenge(page, solver, url: str):
    response = None
    last_error = None
    for attempt in range(3):
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 < 3 and any(marker in str(exc) for marker in (
                "NS_ERROR_NET_RESET", "NS_ERROR_CONNECTION_REFUSED", "net::ERR_CONNECTION_RESET"
            )):
                await page.wait_for_timeout((attempt + 1) * 1000)
                continue
            raise
    if response is None and last_error:
        raise last_error
    await page.wait_for_timeout(2500)
    if await page_is_blocked(page):
        if not await solve_interstitial(page, solver):
            raise RuntimeError(f"access verification blocked ({await page.title() or 'Cloudflare'})")
        await page.wait_for_load_state("domcontentloaded", timeout=TIMEOUT_MS)
    return response


async def collect_state(page, observed: list[dict]) -> dict:
    script = r'''() => ({
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
      media: performance.getEntriesByType("resource").map(e => e.name)
        .filter(u => /\.(m3u8|mpd|mp4|webm)(\?|$)/i.test(u)).slice(-200)
    })'''
    state = await page.evaluate(script)
    state["requests"] = observed[-200:]
    state["cookies"] = await page.context.cookies()
    return state


async def inspect(payload: dict) -> dict:
    url = str(payload.get("url") or "")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an http(s) URL")
    root_url = f"{parsed.scheme}://{parsed.netloc}/"
    user_data_dir = profile_path(url)
    user_data_dir.mkdir(parents=True, exist_ok=True)
    configure_bundled_runtime()
    fingerprint_preset, fingerprint_config = fingerprint_identity(url)
    addon_path = os.path.abspath(get_addon_path())
    observed: list[dict] = []

    async with AsyncCamoufox(
        executable_path=EXECUTABLE,
        persistent_context=True,
        user_data_dir=str(user_data_dir),
        headless=False,
        humanize=True,
        i_know_what_im_doing=True,
        config=fingerprint_config,
        disable_coop=True,
        main_world_eval=True,
        addons=[addon_path],
        exclude_addons=[DefaultAddons.UBO],
        os=FINGERPRINT_OS,
        fingerprint_preset=fingerprint_preset,
        ff_version=FIREFOX_VERSION,
    ) as context:
        page = context.pages[0] if context.pages else await context.new_page()

        async def observe(response):
            try:
                clean_url = response.url.split("?", 1)[0].lower()
                content_type = (response.headers.get("content-type") or "").lower()
                if not (
                    clean_url.endswith(MEDIA_SUFFIXES)
                    or "mpegurl" in content_type
                    or "dash+xml" in content_type
                    or content_type.startswith("video/")
                ):
                    return
                headers = await response.request.all_headers()
                observed.append({
                    "url": response.url,
                    "status": response.status,
                    "content_type": content_type,
                    "request_headers": headers,
                })
            except Exception:
                return

        page.on("response", observe)
        async with ClickSolver(
            framework=FrameworkType.CAMOUFOX,
            page=page,
            max_attempts=2,
            attempt_delay=2,
        ) as solver:
            if root_url != url:
                try:
                    await navigate_with_challenge(page, solver, root_url)
                except Exception as exc:
                    logging.info("root preflight failed: %s", exc)
            observed.clear()
            await navigate_with_challenge(page, solver, url)
            await page.wait_for_timeout(min(12000, TIMEOUT_MS // 3))
            for selector in (
                "button[aria-label*='play' i]",
                ".plyr__control--overlaid",
                ".vjs-big-play-button",
                ".play",
                "video",
            ):
                try:
                    await page.locator(selector).first.click(timeout=1500)
                    await page.wait_for_timeout(2500)
                    break
                except Exception:
                    pass
            for opened_page in list(context.pages):
                if opened_page != page:
                    await opened_page.close()
            await page.bring_to_front()
            state = await collect_state(page, observed)
            if await page_is_blocked(page):
                raise RuntimeError(f"access verification blocked ({state.get('title') or 'Cloudflare'})")
            return state


async def main():
    payload = json.loads(sys.stdin.read() or "{}")
    if payload.get("action") != "inspect":
        raise ValueError("unsupported action")
    result = await inspect(payload)
    print(RESULT_PREFIX + json.dumps({"ok": True, "data": result}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(RESULT_PREFIX + json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
