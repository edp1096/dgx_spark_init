import asyncio
import base64
import hashlib
import ipaddress
import json
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse, Response
from playwright.async_api import async_playwright

from attached_chromium import (
    EXECUTABLE as DIRECT_CHROMIUM_EXECUTABLE,
    attached_chromium,
    page_state as attached_page_state,
    wait_for_access as wait_for_attached_access,
)
from browseforge_client import BrowseForgeClient, BrowseForgeError
from camoufox_client import CamoufoxClient, CamoufoxError
from direct_camoufox_client import DirectCamoufoxClient, DirectCamoufoxError
from site_adapters import adapter_for_url


DATA_DIR = Path(os.getenv("MEDIA_DATA_DIR", "/data"))
SESSION_DIR = DATA_DIR / "sessions"
ASSET_DIR = DATA_DIR / "media"
PROGRESS_DIR = DATA_DIR / "progress"
MAX_UPLOAD_BYTES = int(os.getenv("MEDIA_MAX_UPLOAD_MB", "16384")) << 20
BROWSER_TIMEOUT_MS = int(os.getenv("MEDIA_BROWSER_TIMEOUT_SECONDS", "45")) * 1000
MEDIA_SUFFIXES = (".m3u8", ".mpd", ".mp4", ".webm", ".m4a", ".mp3", ".aac")

app = FastAPI(title="Media Access API", version="1")
browseforge = BrowseForgeClient()
camoufox = CamoufoxClient()
direct_camoufox = DirectCamoufoxClient()
active_prepare_dirs: set[Path] = set()
active_prepare_lock = threading.Lock()
active_prepare_processes: dict[str, subprocess.Popen] = {}
cancelled_prepare_ids: set[str] = set()


class PrepareCancelled(RuntimeError):
    pass


@app.on_event("startup")
def startup():
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health():
    browseforge_online = False
    pot_provider_online = False
    try:
        browseforge_online = bool(browseforge.request("GET", "/api/status"))
    except BrowseForgeError:
        pass
    try:
        with urlopen("http://127.0.0.1:4416/ping", timeout=2) as response:
            pot_provider_online = response.status == 200
    except OSError:
        pass
    return {
        "status": "ok",
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "yt_dlp": shutil.which("yt-dlp") is not None,
        "browsers": ["chromium", "firefox"],
        "direct_chromium": Path(DIRECT_CHROMIUM_EXECUTABLE).is_file(),
        "direct_camoufox": direct_camoufox.configured,
        "browseforge": browseforge_online,
        "camoufox": camoufox.configured,
        "pot_provider": pot_provider_online,
    }


def progress_path(request_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request_id):
        raise ValueError("invalid request id")
    return PROGRESS_DIR / f"{request_id}.json"


def set_progress(request_id: str | None, stage: str, **values):
    if not request_id:
        return
    destination = progress_path(request_id)
    payload = {"stage": stage, **values}
    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)


def request_work_dir(request_id: str | None) -> Path:
    if not request_id:
        return Path(tempfile.mkdtemp(prefix="prepare-", dir=DATA_DIR))
    progress_path(request_id)  # validate before using it as a directory name
    return DATA_DIR / f"prepare-{request_id}"


def ensure_prepare_active(request_id: str | None):
    if not request_id:
        return
    with active_prepare_lock:
        if request_id in cancelled_prepare_ids:
            raise PrepareCancelled("media preparation cancelled")


def register_prepare_process(request_id: str | None, process: subprocess.Popen):
    if not request_id:
        return
    with active_prepare_lock:
        if request_id in cancelled_prepare_ids:
            process.terminate()
            process.wait(timeout=5)
            raise PrepareCancelled("media preparation cancelled")
        active_prepare_processes[request_id] = process


def unregister_prepare_process(request_id: str | None, process: subprocess.Popen):
    if not request_id:
        return
    with active_prepare_lock:
        if active_prepare_processes.get(request_id) is process:
            active_prepare_processes.pop(request_id, None)


def finish_prepare(request_id: str | None, work_dir: Path):
    with active_prepare_lock:
        active_prepare_dirs.discard(work_dir)
        if request_id:
            active_prepare_processes.pop(request_id, None)
            cancelled_prepare_ids.discard(request_id)


def begin_prepare(request_id: str | None, work_dir: Path) -> None:
    """Reserve one durable work directory for exactly one active request."""
    with active_prepare_lock:
        if work_dir in active_prepare_dirs:
            raise HTTPException(409, "media preparation is already active for this request")
        if request_id:
            cancelled_prepare_ids.discard(request_id)
        work_dir.mkdir(parents=True, exist_ok=True)
        active_prepare_dirs.add(work_dir)


def recovery_path(work_dir: Path) -> Path:
    return work_dir / "recovery.json"


def read_recovery(work_dir: Path) -> dict:
    try:
        value = json.loads(recovery_path(work_dir).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_recovery(work_dir: Path, **values):
    current = read_recovery(work_dir)
    current.update(values)
    destination = recovery_path(work_dir)
    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)


def reusable_source(work_dir: Path, source_name: str) -> Path | None:
    recovery = read_recovery(work_dir)
    if recovery and recovery.get("source_name") != source_name:
        return None
    recorded = Path(str(recovery.get("source_file") or "")).name
    candidates = [work_dir / recorded] if recorded else []
    candidates.extend(sorted(work_dir.glob("source.*"), key=lambda item: item.stat().st_size, reverse=True))
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if candidate.name.endswith((".part", ".ytdl")) or ".part-Frag" in candidate.name:
            continue
        try:
            if candidate.stat().st_size > 0 and probe_duration(candidate) > 0:
                validate_audio_decode(candidate)
                return candidate
        except Exception:
            # ffprobe only reads container metadata and can accept an MP4 whose
            # AAC payload is truncated or byte-shifted. Never resume from such
            # a file; a URL retry must resolve and download a fresh source.
            candidate.unlink(missing_ok=True)
            continue
    return None


def reusable_asset(work_dir: Path) -> dict | None:
    asset_id = str(read_recovery(work_dir).get("asset_id") or "")
    if not asset_id:
        return None
    try:
        _, _, metadata = asset_paths(asset_id)
        return metadata
    except HTTPException:
        return None


def prepare_temp_entries() -> list[dict]:
    entries = []
    with active_prepare_lock:
        active = set(active_prepare_dirs)
        directories = list(DATA_DIR.glob("prepare-*"))
    for directory in directories:
        if not directory.is_dir():
            continue
        try:
            size = 0
            newest = directory.stat().st_mtime
            for root, _, files in os.walk(directory):
                for name in files:
                    stat = (Path(root) / name).stat()
                    size += stat.st_size
                    newest = max(newest, stat.st_mtime)
        except OSError:
            continue
        entries.append({
            "path": directory,
            "size": size,
            "modified_at": newest,
            "active": directory in active,
        })
    return entries


@app.get("/v1/media/storage")
def media_storage():
    entries = prepare_temp_entries()
    inactive = [entry for entry in entries if not entry["active"]]
    return {
        "temporary_directories": len(entries),
        "temporary_bytes": sum(entry["size"] for entry in entries),
        "active_directories": sum(1 for entry in entries if entry["active"]),
        "reclaimable_directories": len(inactive),
        "reclaimable_bytes": sum(entry["size"] for entry in inactive),
    }


@app.delete("/v1/media/storage/temp")
def cleanup_media_storage(older_than_hours: int = 0):
    if older_than_hours < 0 or older_than_hours > 8760:
        raise HTTPException(400, "older_than_hours must be between 0 and 8760")
    cutoff = time.time() - older_than_hours * 3600
    removed_directories = 0
    removed_bytes = 0
    for entry in prepare_temp_entries():
        if entry["active"] or (older_than_hours and entry["modified_at"] > cutoff):
            continue
        shutil.rmtree(entry["path"], ignore_errors=False)
        removed_directories += 1
        removed_bytes += entry["size"]
    return {"removed_directories": removed_directories, "removed_bytes": removed_bytes}


@app.get("/v1/media/progress/{request_id}")
def get_media_progress(request_id: str):
    try:
        path = progress_path(request_id)
    except ValueError as exc:
        raise HTTPException(404, "progress not found") from exc
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(404, "progress not found") from None


@app.delete("/v1/media/progress/{request_id}", status_code=204)
def delete_media_progress(request_id: str):
    try:
        progress_path(request_id).unlink(missing_ok=True)
    except ValueError as exc:
        raise HTTPException(404, "progress not found") from exc


@app.delete("/v1/media/jobs/{request_id}", status_code=204)
def delete_media_job_artifacts(request_id: str):
    """Remove durable preparation state owned by one Spark Media job.

    Deletion is deliberately scoped to the validated request ID. Browser
    profiles, shared caches and persisted media assets are managed separately.
    """
    try:
        progress = progress_path(request_id)
        work_dir = request_work_dir(request_id)
    except ValueError as exc:
        raise HTTPException(404, "media job artifacts not found") from exc

    with active_prepare_lock:
        active = work_dir in active_prepare_dirs
        process = active_prepare_processes.get(request_id)
        if active:
            cancelled_prepare_ids.add(request_id)
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if active:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            with active_prepare_lock:
                if work_dir not in active_prepare_dirs:
                    break
            time.sleep(0.05)
        with active_prepare_lock:
            if work_dir in active_prepare_dirs:
                raise HTTPException(409, "media preparation is still stopping")

    shutil.rmtree(work_dir, ignore_errors=True)
    progress.unlink(missing_ok=True)
    with active_prepare_lock:
        active_prepare_processes.pop(request_id, None)
        cancelled_prepare_ids.discard(request_id)


@app.delete("/v1/media/prepare/{request_id}", status_code=202)
def cancel_media_prepare(request_id: str):
    try:
        progress_path(request_id)
    except ValueError as exc:
        raise HTTPException(404, "media preparation not found") from exc
    work_dir = request_work_dir(request_id)
    with active_prepare_lock:
        if work_dir not in active_prepare_dirs:
            return {"status": "not_active", "request_id": request_id}
        cancelled_prepare_ids.add(request_id)
        process = active_prepare_processes.get(request_id)
    set_progress(request_id, "cancelled")
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    # Let the active handler observe cancellation and release its durable work
    # directory before a restarted client submits the same request ID again.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with active_prepare_lock:
            if work_dir not in active_prepare_dirs:
                break
        time.sleep(0.05)
    return {"status": "cancelling", "request_id": request_id}


def run(command: list[str], timeout: int | None = None, request_id: str | None = None):
    if request_id:
        ensure_prepare_active(request_id)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        register_prepare_process(request_id, process)
        try:
            output, _ = process.communicate(timeout=timeout)
        except Exception:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            unregister_prepare_process(request_id, process)
        ensure_prepare_active(request_id)
        if process.returncode:
            message = output.strip()[-4000:]
            raise RuntimeError(message or f"command failed with exit code {process.returncode}")
        return output
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        message = completed.stdout.strip()[-4000:]
        raise RuntimeError(message or f"command failed with exit code {completed.returncode}")
    return completed.stdout


def run_stdout(command: list[str], timeout: int | None = None):
    """Run a machine-readable command without mixing diagnostics into stdout."""
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        message = (completed.stderr or completed.stdout).strip()[-4000:]
        raise RuntimeError(message or f"command failed with exit code {completed.returncode}")
    return completed.stdout


def run_prepare_command(command: list[str], timeout: int, request_id: str | None):
    if request_id:
        return run(command, timeout=timeout, request_id=request_id)
    return run(command, timeout=timeout)


def progress_number(value: str) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return 0


def run_download(command: list[str], request_id: str | None, timeout: int | None = None):
    if not request_id:
        return run(command, timeout=timeout)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    register_prepare_process(request_id, process)
    output = []
    try:
        assert process.stdout is not None
        for line in process.stdout:
            line = line.rstrip()
            if line.startswith("MEDIA_PROGRESS:"):
                parts = line.split(":", 4)
                downloaded = progress_number(parts[1] if len(parts) > 1 else "0")
                total = progress_number(parts[2] if len(parts) > 2 else "0") or progress_number(parts[3] if len(parts) > 3 else "0")
                eta = progress_number(parts[4] if len(parts) > 4 else "0")
                percent = round(downloaded * 100 / total, 1) if total else 0
                set_progress(request_id, "downloading", downloaded_bytes=downloaded, total_bytes=total, percent=percent, eta_seconds=eta)
            else:
                output.append(line)
                if len(output) > 200:
                    output = output[-200:]
        return_code = process.wait(timeout=timeout)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait()
        raise
    finally:
        unregister_prepare_process(request_id, process)
    ensure_prepare_active(request_id)
    if return_code:
        message = "\n".join(output).strip()[-4000:]
        raise RuntimeError(message or f"command failed with exit code {return_code}")
    return "\n".join(output)


def validate_public_url(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("url must be an http(s) URL")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(parsed.hostname, parsed.port or 443)}
    except OSError as exc:
        raise ValueError(f"cannot resolve URL host: {exc}") from exc
    for value in addresses:
        address = ipaddress.ip_address(value)
        if not address.is_global:
            raise ValueError("private, loopback, and link-local URL hosts are not allowed")
    return parsed.geturl()


async def save_upload(upload: UploadFile, destination: Path):
    written = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(4 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise ValueError("uploaded media is too large")
            output.write(chunk)
    if written == 0:
        raise ValueError("uploaded media is empty")


def cookie_file_from_state(state: dict, destination: Path):
    lines = ["# Netscape HTTP Cookie File"]
    for cookie in state.get("cookies", []):
        domain = cookie.get("domain", "")
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        secure = "TRUE" if cookie.get("secure") else "FALSE"
        expires = int(cookie.get("expires", 0))
        lines.append("\t".join([
            domain, include_subdomains, cookie.get("path", "/"), secure,
            str(max(0, expires)), cookie.get("name", ""), cookie.get("value", ""),
        ]))
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def session_paths(url: str) -> tuple[Path, Path]:
    host = urlparse(url).hostname or "unknown"
    key = hashlib.sha256(host.encode()).hexdigest()[:24]
    return SESSION_DIR / f"{key}.json", SESSION_DIR / f"{key}.cookies.txt"


def browser_session_paths(url: str, browser_name: str) -> tuple[Path, Path, Path]:
    host = urlparse(url).hostname or "unknown"
    key = hashlib.sha256(host.encode()).hexdigest()[:24]
    prefix = SESSION_DIR / f"{key}-{browser_name}"
    return prefix.with_suffix(".json"), SESSION_DIR / f"{key}-{browser_name}.cookies.txt", SESSION_DIR / f"{key}-{browser_name}.profile"


@asynccontextmanager
async def resolver_browser_context(playwright, url: str, browser_name: str, profile_path: Path):
    if browser_name == "chromium":
        host = (urlparse(url).hostname or "unknown").lower()
        async with attached_chromium(playwright, host) as context:
            yield context
        return
    browser_type = getattr(playwright, browser_name)
    context = await browser_type.launch_persistent_context(str(profile_path), headless=True)
    try:
        yield context
    finally:
        await context.close()


def yt_dlp_error_summary(message: str) -> str:
    statuses = sorted(set(re.findall(r"HTTP Error (\d{3})", message)))
    fragment_failures = len(re.findall(r"fragment (?:not found|\d+)", message, re.IGNORECASE))
    if statuses and fragment_failures:
        return f"media fragments failed with HTTP {','.join(statuses)} ({fragment_failures} failures)"
    errors = [line.strip() for line in message.splitlines() if line.strip().startswith("ERROR:")]
    if errors:
        return errors[-1].removeprefix("ERROR:").strip()
    return message.strip().splitlines()[-1] if message.strip() else "download failed"


def recover_corrupt_partial_download(work_dir: Path, error: str, request_id: str | None = None) -> Path | None:
    """Keep a playable HLS download when yt-dlp's final FFmpeg pass hits bad AAC frames."""
    decode_markers = (
        "error submitting packet to decoder",
        "decoding error",
        "prediction is not allowed in aac",
        "sample rate index in program config element",
        "too large remapped id",
    )
    if not any(marker in error.lower() for marker in decode_markers):
        return None
    candidates = sorted(
        (
            path for path in work_dir.glob("source*.part")
            if path.is_file() and ".part-Frag" not in path.name and path.stat().st_size >= (16 << 20)
        ),
        key=lambda path: path.stat().st_size,
        reverse=True,
    )
    for partial in candidates:
        try:
            probe = probe_media(partial)
            streams = probe.get("streams") or []
            if not any(stream.get("codec_type") in {"video", "audio"} for stream in streams):
                continue
            if probe_duration(partial) < 60:
                continue
            destination = work_dir / "source.recovered.mp4"
            destination.unlink(missing_ok=True)
            run_prepare_command([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-fflags", "+discardcorrupt", "-err_detect", "ignore_err", "-i", str(partial),
                "-map", "0:v?", "-map", "0:a?", "-map_metadata", "0", "-c", "copy",
                "-movflags", "+faststart", str(destination),
            ], 7200, request_id)
            if destination.stat().st_size > 0 and probe_duration(destination) >= 60:
                return destination
        except PrepareCancelled:
            raise
        except Exception:
            destination.unlink(missing_ok=True)
    return None


def validate_audio_decode(source: Path, request_id: str | None = None) -> None:
    """Decode the complete primary audio stream before accepting a download.

    ffprobe can report a plausible duration for a damaged MP4 because it only
    reads container metadata. The providers used by browser resolution can
    occasionally return a nearly complete file with malformed AAC near the
    end, so a full null decode is the integrity check that matters for ASR.
    """
    probe = probe_media(source)
    if not any(stream.get("codec_type") == "audio" for stream in probe.get("streams", [])):
        raise RuntimeError("media contains no audio stream")
    run_prepare_command([
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", str(source), "-map", "0:a:0", "-vn", "-f", "null", "-",
    ], 7200, request_id)


def yt_dlp_download(url: str, work_dir: Path, cookies: Path | None = None, headers: dict | None = None, request_id: str | None = None) -> Path:
    output = str(work_dir / "source.%(ext)s")
    host = (urlparse(url).hostname or "").lower()
    youtube_clients = ("web_embedded", "mweb", None) if host == "youtu.be" or host.endswith("youtube.com") else (None,)
    is_hls = urlparse(url).path.lower().endswith(".m3u8")
    formats = (
        ("bestvideo*+bestaudio/best", "original"),
        ("best[height<=720]/best", "720p fallback"),
        ("best[height<=480]/worst", "480p fallback"),
    ) if is_hls else (("bestvideo*+bestaudio/best", "default"),)
    errors = []
    for client, (format_selector, quality_label) in (
        (client, format_item) for client in youtube_clients for format_item in formats
    ):
        command = [
            "yt-dlp", "--no-playlist", "--newline", "--no-colors",
            "--impersonate", "chrome", "-f", format_selector,
            "--merge-output-format", "mp4", "-o", output,
        ]
        if is_hls:
            command += [
                "--concurrent-fragments", "4",
                "--fragment-retries", "20",
                "--retry-sleep", "fragment:linear=1:5:1",
                "--abort-on-unavailable-fragments",
            ]
        if request_id:
            command += [
                "--progress-template",
                "download:MEDIA_PROGRESS:%(progress.downloaded_bytes)s:%(progress.total_bytes)s:%(progress.total_bytes_estimate)s:%(progress.eta)s",
            ]
        else:
            command += ["--no-progress"]
        if client:
            command += ["--extractor-args", f"youtube:player_client={client}"]
        if cookies and cookies.exists():
            command += ["--cookies", str(cookies)]
        for key, value in (headers or {}).items():
            if value:
                command += ["--add-header", f"{key}:{value}"]
        command += ["--", url]
        try:
            set_progress(request_id, "downloading", downloaded_bytes=0, total_bytes=0, percent=0, eta_seconds=0)
            run_download(command, request_id, timeout=7200)
            candidates = [path for path in work_dir.glob("source.*") if path.is_file() and path.suffix != ".part"]
            if candidates:
                result = max(candidates, key=lambda path: path.stat().st_size)
                probe_duration(result)
                validate_audio_decode(result, request_id)
                return result
            errors.append(f"{quality_label}: completed without a media file")
        except PrepareCancelled:
            raise
        except Exception as exc:
            error = str(exc)
            recovered = recover_corrupt_partial_download(work_dir, error, request_id)
            if recovered is not None:
                return recovered
            summary = yt_dlp_error_summary(error)
            errors.append(f"{quality_label}: {summary}")
        for path in work_dir.glob("source.*"):
            if path.is_file():
                path.unlink(missing_ok=True)
    raise RuntimeError("; ".join(errors))


def media_candidate_score(candidate: dict) -> int:
    url = candidate["url"].lower().split("?", 1)[0]
    content_type = candidate["content_type"]
    score = 0
    if url.endswith(".m3u8") or "mpegurl" in content_type:
        score += 10000
    elif url.endswith(".mpd") or "dash+xml" in content_type:
        score += 8000
    elif url.endswith((".mp4", ".webm")) or content_type.startswith("video/"):
        score += 5000
    if candidate["status"] in {200, 206}:
        score += 1000
    dimensions = re.findall(r"(?:^|[/_-])(\d{3,4})x(\d{3,4})(?:[/_.-]|$)", url)
    if dimensions:
        width, height = map(int, dimensions[-1])
        score += min(width * height // 1000, 5000)
    if "alternative" in url or "-alt." in url:
        score -= 3000
    return score


def download_url_to_file(url: str, destination, headers: dict, written: int) -> int:
    request = Request(url, headers={key: value for key, value in headers.items() if value})
    with urlopen(request, timeout=60) as response:
        while chunk := response.read(4 << 20):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise RuntimeError("downloaded Vimeo media is too large")
            destination.write(chunk)
    return written


def assemble_vimeo_playlist(playlist_url: str, playlist: dict, work_dir: Path, headers: dict, request_id: str | None = None) -> Path:
    videos = playlist.get("video") or []
    audios = playlist.get("audio") or []
    if not videos or not audios:
        raise RuntimeError("Vimeo playlist contains no video or audio tracks")
    video = max(videos, key=lambda item: (int(item.get("width") or 0) * int(item.get("height") or 0), int(item.get("bitrate") or 0)))
    audio = max(audios, key=lambda item: (bool(item.get("audio_primary")), int(item.get("bitrate") or 0)))
    common_base = urljoin(playlist_url, playlist.get("base_url") or "")
    expected_size = sum(int(segment.get("size") or 0) for track in (video, audio) for segment in track.get("segments") or [])
    if expected_size > MAX_UPLOAD_BYTES:
        raise RuntimeError("downloaded Vimeo media is too large")
    downloaded_size = 0
    progress_lock = threading.Lock()
    set_progress(request_id, "downloading", downloaded_bytes=0, total_bytes=expected_size, percent=0, eta_seconds=0)

    def assemble_track(kind: str, track: dict, destination: Path) -> None:
        track_base = urljoin(common_base, track.get("base_url") or "")
        fragments = work_dir / f"vimeo-{kind}-fragments"
        fragments.mkdir()
        segment_items = list(enumerate(track.get("segments") or []))

        def fetch_segment(item) -> Path:
            nonlocal downloaded_size
            index, segment = item
            path = fragments / f"{index:06d}.m4s"
            if not segment.get("url"):
                path.touch()
                return path
            with path.open("wb") as output:
                download_url_to_file(urljoin(track_base, segment["url"]), output, headers, 0)
            with progress_lock:
                downloaded_size += path.stat().st_size
                percent = round(downloaded_size * 100 / expected_size, 1) if expected_size else 0
                set_progress(request_id, "downloading", downloaded_bytes=downloaded_size, total_bytes=expected_size, percent=percent, eta_seconds=0)
            return path

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(segment_items)))) as executor:
            segment_paths = list(executor.map(fetch_segment, segment_items))

        written = 0
        with destination.open("wb") as output:
            init_segment = track.get("init_segment") or ""
            if init_segment:
                initial = base64.b64decode(init_segment)
                output.write(initial)
                written += len(initial)
            elif track.get("init_segment_url"):
                written = download_url_to_file(urljoin(track_base, track["init_segment_url"]), output, headers, written)
            else:
                raise RuntimeError("Vimeo track contains no initialization segment")
            for segment_path in segment_paths:
                with segment_path.open("rb") as fragment:
                    while chunk := fragment.read(4 << 20):
                        written += len(chunk)
                        if written > MAX_UPLOAD_BYTES:
                            raise RuntimeError("downloaded Vimeo media is too large")
                        output.write(chunk)

    video_path = work_dir / "vimeo-video.mp4"
    audio_path = work_dir / "vimeo-audio.m4a"
    assemble_track("video", video, video_path)
    assemble_track("audio", audio, audio_path)
    output = work_dir / "source.mp4"
    run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c", "copy",
        "-movflags", "+faststart", str(output),
    ], timeout=7200)
    validate_audio_decode(output, request_id)
    return output


async def browser_resolve(
    url: str, work_dir: Path, request_id: str | None = None, selection: dict | None = None
) -> tuple[str | Path, Path, dict]:
    legacy_state_path, generic_cookie_path = session_paths(url)
    last_error = "no playable media request was observed"
    adapter = adapter_for_url(url)
    async with async_playwright() as playwright:
        browser_order = ("chromium", *(name for name in adapter.browser_order if name != "chromium"))
        for browser_name in browser_order:
            for attempt in range(2):
                state_path, cookie_path, profile_path = browser_session_paths(url, browser_name)
                profile_path.mkdir(parents=True, exist_ok=True)
                try:
                    async with resolver_browser_context(
                        playwright, url, browser_name, profile_path
                    ) as context:
                        seed_path = state_path
                        if not seed_path.exists() and browser_name == "firefox" and legacy_state_path.exists():
                            seed_path = legacy_state_path
                        if browser_name != "chromium" and seed_path.exists():
                            try:
                                seed = json.loads(seed_path.read_text(encoding="utf-8"))
                                if seed.get("cookies"):
                                    await context.add_cookies(seed["cookies"])
                            except (OSError, ValueError):
                                pass
                        page = context.pages[0] if context.pages else await context.new_page()
                        candidates: list[dict] = []
                        special_responses: dict[str, list] = {}

                        def observe(response):
                            clean_url = response.url.split("?", 1)[0].lower()
                            content_type = (response.headers.get("content-type") or "").lower()
                            special_kind = adapter.special_response(response.url, content_type)
                            if special_kind:
                                special_responses.setdefault(special_kind, []).append(response)
                            if clean_url.endswith((".m4s", ".ts")):
                                return
                            if clean_url.endswith(MEDIA_SUFFIXES) or "mpegurl" in content_type or "dash+xml" in content_type or content_type.startswith("video/"):
                                candidates.append({
                                    "url": response.url,
                                    "status": response.status,
                                    "content_type": content_type,
                                    "response": response,
                                })

                        page.on("response", observe)
                        await adapter.before_detail(page, url, BROWSER_TIMEOUT_MS)
                        candidates.clear()
                        response = await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT_MS)
                        if browser_name == "chromium":
                            await wait_for_attached_access(page)
                        response = await adapter.after_detail(
                            page, url, response, BROWSER_TIMEOUT_MS, candidates
                        )
                        if browser_name == "chromium":
                            await wait_for_attached_access(page)
                        await page.wait_for_timeout(min(12000, BROWSER_TIMEOUT_MS // 2))
                        original_page = page
                        for selector in ("button[aria-label*='play' i]", ".plyr__control--overlaid", ".vjs-big-play-button", ".play", "video"):
                            try:
                                await original_page.locator(selector).first.click(timeout=2000)
                                await original_page.wait_for_timeout(3000)
                                break
                            except Exception:
                                pass
                        for opened_page in list(context.pages):
                            if opened_page != original_page:
                                await opened_page.close()
                        await original_page.bring_to_front()
                        if selection:
                            candidates.clear()
                            current_state = await attached_page_state(original_page)
                            selection_headers = {
                                "User-Agent": current_state.get("userAgent") or "Mozilla/5.0",
                                "Referer": current_state.get("url") or url,
                            }
                            for candidate in adapter.browseforge_extra_candidates(
                                current_state, selection_headers, max(1, BROWSER_TIMEOUT_MS // 1000), selection
                            ):
                                suffix = urlparse(candidate).path.lower()
                                candidates.append({
                                    "url": candidate,
                                    "status": 200,
                                    "content_type": "application/vnd.apple.mpegurl" if suffix.endswith(".m3u8") else "video/mp4",
                                    "response": None,
                                })
                        if special_responses.get("vimeo_playlist"):
                            state = await context.storage_state(path=str(state_path))
                            cookie_file_from_state(state, cookie_path)
                            shutil.copyfile(cookie_path, generic_cookie_path)
                            playlist_response = special_responses["vimeo_playlist"][-1]
                            playlist = await playlist_response.json()
                            user_agent = await original_page.evaluate("navigator.userAgent")
                            headers = {"User-Agent": user_agent, "Referer": url}
                            source = await asyncio.to_thread(
                                assemble_vimeo_playlist, playlist_response.url, playlist, work_dir, headers, request_id
                            )
                            return source, cookie_path, headers
                        if candidates:
                            selected = max(candidates, key=media_candidate_score)
                            request_headers = (
                                await selected["response"].request.all_headers()
                                if selected.get("response") is not None else {}
                            )
                            user_agent = request_headers.get("user-agent") or await original_page.evaluate("navigator.userAgent")
                            headers = {
                                "User-Agent": user_agent,
                                "Referer": request_headers.get("referer") or url,
                            }
                            if request_headers.get("origin"):
                                headers["Origin"] = request_headers["origin"]
                            if request_headers.get("accept-language"):
                                headers["Accept-Language"] = request_headers["accept-language"]
                            state = await context.storage_state(path=str(state_path))
                            cookie_file_from_state(state, cookie_path)
                            shutil.copyfile(cookie_path, generic_cookie_path)
                            return selected["url"], cookie_path, headers
                        title = await original_page.title()
                        if response and response.status == 403:
                            last_error = f"{browser_name}: access verification blocked ({title or 'HTTP 403'})"
                        else:
                            last_error = f"{browser_name}: no playable media request was observed"
                except Exception as exc:
                    last_error = f"{browser_name} attempt {attempt + 1}: {exc}"
    raise RuntimeError(last_error)


async def download_via_browser(
    url: str, work_dir: Path, request_id: str | None = None, selection: dict | None = None
) -> Path:
    media_result, cookies, headers = await browser_resolve(url, work_dir, request_id, selection)
    if isinstance(media_result, Path):
        return media_result
    return await asyncio.to_thread(yt_dlp_download, media_result, work_dir, cookies, headers, request_id)


async def options_via_browser(url: str, adapter) -> dict:
    host = (urlparse(url).hostname or "unknown").lower()
    async with async_playwright() as playwright:
        async with attached_chromium(playwright, host) as context:
            page = context.pages[0] if context.pages else await context.new_page()
            await adapter.before_detail(page, url, BROWSER_TIMEOUT_MS)
            await wait_for_attached_access(page)
            await page.goto(url, wait_until="domcontentloaded", timeout=BROWSER_TIMEOUT_MS)
            await wait_for_attached_access(page)
            state = await attached_page_state(page)
            return adapter.browseforge_options(state)


async def download_via_browseforge(
    url: str, work_dir: Path, request_id: str | None = None, selection: dict | None = None
) -> Path:
    adapter = adapter_for_url(url)
    candidates, browseforge_cookies, headers = await asyncio.to_thread(
        browseforge.resolve, url, adapter, selection
    )
    ranked = []
    for candidate in candidates:
        suffix = urlparse(candidate).path.lower()
        content_type = (
            "application/vnd.apple.mpegurl" if suffix.endswith(".m3u8")
            else "application/dash+xml" if suffix.endswith(".mpd")
            else "video/mp4" if suffix.endswith(".mp4")
            else "video/webm" if suffix.endswith(".webm")
            else ""
        )
        ranked.append({"url": candidate, "status": 200, "content_type": content_type})
    if not ranked:
        raise BrowseForgeError("no playable media was observed")
    selected = max(ranked, key=media_candidate_score)
    await asyncio.to_thread(validate_public_url, selected["url"])
    _, cookies = session_paths(url)
    await asyncio.to_thread(cookie_file_from_state, {"cookies": browseforge_cookies}, cookies)
    return await asyncio.to_thread(
        yt_dlp_download, selected["url"], work_dir, cookies, headers, request_id
    )


async def download_via_camoufox(
    url: str, work_dir: Path, request_id: str | None = None, selection: dict | None = None
) -> Path:
    adapter = adapter_for_url(url)
    candidates, session_cookies, headers = await asyncio.to_thread(
        camoufox.resolve, url, adapter, selection
    )
    ranked = []
    for candidate in candidates:
        suffix = urlparse(candidate).path.lower()
        content_type = (
            "application/vnd.apple.mpegurl" if suffix.endswith(".m3u8")
            else "application/dash+xml" if suffix.endswith(".mpd")
            else "video/mp4" if suffix.endswith(".mp4")
            else "video/webm" if suffix.endswith(".webm")
            else ""
        )
        ranked.append({"url": candidate, "status": 200, "content_type": content_type})
    if not ranked:
        raise CamoufoxError("no playable media was observed")
    selected = max(ranked, key=media_candidate_score)
    await asyncio.to_thread(validate_public_url, selected["url"])
    _, cookies = session_paths(url)
    await asyncio.to_thread(cookie_file_from_state, {"cookies": session_cookies}, cookies)
    return await asyncio.to_thread(
        yt_dlp_download, selected["url"], work_dir, cookies, headers, request_id
    )


async def download_via_direct_camoufox(
    url: str, work_dir: Path, request_id: str | None = None, selection: dict | None = None
) -> Path:
    adapter = adapter_for_url(url)
    _, cookie_path = session_paths(url)
    candidates, cookies, headers = await asyncio.to_thread(
        direct_camoufox.resolve, url, adapter, cookie_path, selection
    )
    ranked = []
    for candidate in candidates:
        suffix = urlparse(candidate).path.lower()
        content_type = (
            "application/vnd.apple.mpegurl" if suffix.endswith(".m3u8")
            else "application/dash+xml" if suffix.endswith(".mpd")
            else "video/webm" if suffix.endswith(".webm")
            else "video/mp4"
        )
        ranked.append({"url": candidate, "status": 200, "content_type": content_type})
    selected = max(ranked, key=media_candidate_score)
    await asyncio.to_thread(validate_public_url, selected["url"])
    return await asyncio.to_thread(
        yt_dlp_download, selected["url"], work_dir, cookies, headers, request_id
    )


def probe_duration(path: Path) -> float:
    output = run_stdout([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    try:
        return max(0.0, float(output.strip()))
    except ValueError:
        return 0.0


def probe_media(path: Path) -> dict:
    output = run_stdout([
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ])
    try:
        probe = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc
    if not isinstance(probe, dict):
        raise RuntimeError("ffprobe returned an invalid media description")
    return probe


def persist_media_asset(source: Path, source_name: str, request_id: str | None = None) -> dict | None:
    probe = probe_media(source)
    video_streams = [
        stream for stream in probe.get("streams", [])
        if stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")
    ]
    audio_streams = [stream for stream in probe.get("streams", []) if stream.get("codec_type") == "audio"]
    if not video_streams and not audio_streams:
        return None

    media_type = "video" if video_streams else "audio"
    asset_id = uuid.uuid4().hex
    staging = ASSET_DIR / f".{asset_id}.tmp"
    destination_dir = ASSET_DIR / asset_id
    staging.mkdir(parents=True, exist_ok=False)
    try:
        if media_type == "video":
            destination = staging / "video.mp4"
            # 웹 재생과 Range 탐색을 위해 가능한 경우 재인코딩 없이 MP4로 remux한다.
            try:
                run_prepare_command([
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-fflags", "+discardcorrupt", "-err_detect", "ignore_err", "-i", str(source),
                    "-map", "0:v:0", "-map", "0:a?", "-map_metadata", "0",
                    "-c", "copy", "-movflags", "+faststart", str(destination),
                ], 7200, request_id)
            except PrepareCancelled:
                raise
            except Exception:
                suffix = source.suffix.lower() if source.suffix else ".bin"
                destination = staging / f"video{suffix}"
                shutil.copy2(source, destination)
            content_type = mimetypes.guess_type(destination.name)[0] or "application/octet-stream"
            width = int(video_streams[0].get("width") or 0)
            height = int(video_streams[0].get("height") or 0)
        else:
            destination = staging / "audio.m4a"
            codec = str(audio_streams[0].get("codec_name") or "").lower()
            codec_args = ["-c:a", "copy"] if codec == "aac" else ["-c:a", "aac", "-b:a", "192k"]
            run_prepare_command([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                "-map", "0:a:0", "-vn", "-map_metadata", "0", *codec_args,
                "-movflags", "+faststart", str(destination),
            ], 7200, request_id)
            content_type = "audio/mp4"
            width = 0
            height = 0
        metadata = {
            "id": asset_id,
            "filename": destination.name,
            "source_name": source_name,
            "media_type": media_type,
            "content_type": content_type,
            "size": destination.stat().st_size,
            "duration": probe_duration(destination),
            "width": width,
            "height": height,
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.rename(destination_dir)
        return metadata
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def asset_paths(asset_id: str) -> tuple[Path, Path, dict]:
    if not re.fullmatch(r"[0-9a-f]{32}", asset_id):
        raise HTTPException(404, "media asset not found")
    directory = ASSET_DIR / asset_id
    metadata_path = directory / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(404, "media asset not found") from None
    media_path = directory / Path(str(metadata.get("filename", ""))).name
    if not media_path.is_file():
        raise HTTPException(404, "media asset not found")
    return directory, media_path, metadata


def prepare_segments(source: Path, work_dir: Path, segment_seconds: int, request_id: str | None = None) -> list[dict]:
    segment_dir = work_dir / "segments"
    # A killed ffmpeg can leave empty or partial WAV files. They are never safe
    # to append to, so restart segmentation from the preserved source media.
    shutil.rmtree(segment_dir, ignore_errors=True)
    segment_dir.mkdir()
    duration = probe_duration(source)
    silence_output = run_prepare_command([
        "ffmpeg", "-hide_banner", "-nostats",
        "-fflags", "+discardcorrupt", "-err_detect", "ignore_err", "-i", str(source),
        "-map", "0:a:0", "-af", "silencedetect=noise=-35dB:d=0.6", "-f", "null", "-",
    ], 7200, request_id)
    silence_starts = [float(value) for value in re.findall(r"silence_start:\s*([0-9.]+)", silence_output)]
    silence_ends = [float(value) for value in re.findall(r"silence_end:\s*([0-9.]+)", silence_output)]
    silence_midpoints = []
    for index, start in enumerate(silence_starts):
        end = silence_ends[index] if index < len(silence_ends) else duration
        if end > start:
            silence_midpoints.append((start + end) / 2)
    cut_points = []
    cursor = 0.0
    while duration - cursor > segment_seconds:
        candidates = [point for point in silence_midpoints if cursor + 2 <= point <= cursor + segment_seconds]
        cut = candidates[-1] if candidates else cursor + segment_seconds
        cut_points.append(cut)
        cursor = cut

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-fflags", "+discardcorrupt", "-err_detect", "ignore_err", "-i", str(source),
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        "-f", "segment", "-reset_timestamps", "1",
    ]
    if cut_points:
        command += ["-segment_times", ",".join(f"{point:.3f}" for point in cut_points)]
    else:
        command += ["-segment_time", str(segment_seconds)]
    command += [str(segment_dir / "segment-%05d.wav")]
    run_prepare_command(command, 7200, request_id)
    segments = []
    cursor = 0.0
    for path in sorted(segment_dir.glob("segment-*.wav")):
        duration = probe_duration(path)
        segments.append({"name": path.name, "start": cursor, "end": cursor + duration, "duration": duration})
        cursor += duration
    if not segments:
        raise RuntimeError("media contains no decodable audio stream")
    return segments


def build_archive(work_dir: Path, source_name: str, segments: list[dict], asset: dict | None) -> Path:
    manifest = {"source_name": source_name, "segments": segments, "asset": asset}
    (work_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    archive = Path(tempfile.mkstemp(prefix="media-prepared-", suffix=".zip", dir=DATA_DIR)[1])
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.write(work_dir / "manifest.json", "manifest.json")
        for segment in segments:
            bundle.write(work_dir / "segments" / segment["name"], segment["name"])
    return archive


def remove_path(path: Path):
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@app.api_route("/v1/media/assets/{asset_id}", methods=["GET", "HEAD"])
def get_media_asset(asset_id: str):
    _, media_path, metadata = asset_paths(asset_id)
    source_name = str(metadata.get("source_name") or "")
    download_name = Path(urlparse(source_name).path).name or media_path.name
    return FileResponse(
        media_path,
        media_type=metadata.get("content_type") or "application/octet-stream",
        filename=download_name,
        content_disposition_type="inline",
    )


@app.post("/v1/media/thumbnails")
async def create_video_thumbnails(video: UploadFile = File(...)):
    """Build one 10x5 JPEG timeline sprite for a locally supplied video."""
    work_dir = Path(tempfile.mkdtemp(prefix="video-thumbnails-", dir=DATA_DIR))
    source = work_dir / (Path(video.filename or "video.mp4").name or "video.mp4")
    sprite = work_dir / "timeline.jpg"
    total = 0
    try:
        with source.open("wb") as output:
            while chunk := await video.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "video is too large")
                output.write(chunk)
        if total == 0:
            raise HTTPException(400, "video is empty")
        duration = await asyncio.to_thread(probe_duration, source)
        if duration <= 0:
            raise HTTPException(422, "video duration is unavailable")
        sample_fps = 50.0 / duration
        video_filter = (
            f"fps={sample_fps:.10f},"
            "scale=160:90:force_original_aspect_ratio=decrease,"
            "pad=160:90:(ow-iw)/2:(oh-ih)/2:black,"
            "tile=10x5"
        )
        await asyncio.to_thread(run_prepare_command, [
            "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error", "-y",
            "-i", str(source), "-an", "-vf", video_filter,
            "-frames:v", "1", "-q:v", "5", str(sprite),
        ], 600, None)
        if not sprite.is_file() or sprite.stat().st_size == 0:
            raise HTTPException(500, "thumbnail sprite was not created")
        return Response(
            content=sprite.read_bytes(),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
                "X-Thumbnail-Count": "50",
                "X-Thumbnail-Columns": "10",
                "X-Thumbnail-Width": "160",
                "X-Thumbnail-Height": "90",
            },
        )
    finally:
        await video.close()
        shutil.rmtree(work_dir, ignore_errors=True)


def extract_video_frame(source: Path, time_seconds: float, destination: Path) -> None:
    if time_seconds < 0:
        raise HTTPException(400, "time_seconds must not be negative")
    duration = probe_duration(source)
    attempts = [time_seconds]
    # Container duration often extends one frame beyond the final decodable
    # presentation timestamp. Seeking to that exact end boundary succeeds but
    # produces no image, so walk back until a real frame is found.
    if duration > 0 and time_seconds >= duration - 0.001:
        attempts.append(max(0.0, duration - 0.05))
    attempts.extend(max(0.0, time_seconds - offset) for offset in (0.05, 0.1, 0.25, 0.5))
    unique_attempts = list(dict.fromkeys(round(value, 6) for value in attempts))
    for attempt in unique_attempts:
        destination.unlink(missing_ok=True)
        try:
            run_prepare_command([
                "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error", "-y",
                "-ss", f"{attempt:.6f}", "-i", str(source), "-an",
                "-frames:v", "1", "-q:v", "2", str(destination),
            ], 600, None)
        except RuntimeError:
            continue
        if destination.is_file() and destination.stat().st_size > 0:
            return
    raise HTTPException(422, "a frame could not be extracted near that time")


@app.post("/v1/media/frame")
async def create_video_frame(time_seconds: float = Form(...), video: UploadFile = File(...)):
    """Extract one full-resolution JPEG frame from a locally supplied video."""
    work_dir = Path(tempfile.mkdtemp(prefix="video-frame-", dir=DATA_DIR))
    source = work_dir / (Path(video.filename or "video.mp4").name or "video.mp4")
    frame = work_dir / "frame.jpg"
    total = 0
    try:
        with source.open("wb") as output:
            while chunk := await video.read(1 << 20):
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "video is too large")
                output.write(chunk)
        if total == 0:
            raise HTTPException(400, "video is empty")
        await asyncio.to_thread(extract_video_frame, source, time_seconds, frame)
        return Response(
            content=frame.read_bytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Frame-Time": f"{time_seconds:.6f}"},
        )
    finally:
        await video.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/v1/media/assets/{asset_id}/frame")
async def create_asset_video_frame(asset_id: str, time_seconds: float = 0):
    """Extract one full-resolution JPEG frame from a persisted transcription source."""
    _, source, metadata = asset_paths(asset_id)
    if str(metadata.get("media_type") or "") == "audio":
        raise HTTPException(409, "audio assets do not contain video frames")
    duration = float(metadata.get("duration") or 0)
    if duration > 0 and time_seconds > duration:
        raise HTTPException(400, "time_seconds exceeds media duration")
    work_dir = Path(tempfile.mkdtemp(prefix="asset-frame-", dir=DATA_DIR))
    frame = work_dir / "frame.jpg"
    try:
        await asyncio.to_thread(extract_video_frame, source, time_seconds, frame)
        return Response(
            content=frame.read_bytes(),
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store", "X-Frame-Time": f"{time_seconds:.6f}"},
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/v1/media/assets/{asset_id}/metadata")
def get_media_asset_metadata(asset_id: str):
    _, _, metadata = asset_paths(asset_id)
    return metadata


@app.delete("/v1/media/assets/{asset_id}", status_code=204)
def delete_media_asset(asset_id: str):
    directory, _, _ = asset_paths(asset_id)
    shutil.rmtree(directory)


@app.post("/v1/media/options")
async def media_options(url: str = Form(...)):
    try:
        target_url = await asyncio.to_thread(validate_public_url, url)
        adapter = adapter_for_url(target_url)
        if not adapter.prefer_browseforge:
            return {"url": target_url, "site": adapter.name, "parts": []}
        browser_error = None
        direct_camoufox_error = None
        if adapter.name == "supjav.com":
            try:
                options = await asyncio.to_thread(direct_camoufox.options, target_url, adapter)
            except Exception as exc:
                direct_camoufox_error = exc
            else:
                return {"url": target_url, **options}
        try:
            options = await options_via_browser(target_url, adapter)
        except Exception as exc:
            browser_error = exc
            if adapter.name != "supjav.com":
                try:
                    options = await asyncio.to_thread(direct_camoufox.options, target_url, adapter)
                except Exception as direct_exc:
                    direct_camoufox_error = direct_exc
                else:
                    return {"url": target_url, **options}
            try:
                options = await asyncio.to_thread(browseforge.options, target_url, adapter)
            except Exception as browseforge_error:
                try:
                    options = await asyncio.to_thread(camoufox.options, target_url, adapter)
                except Exception as camoufox_error:
                    raise RuntimeError(
                        f"direct Camoufox: {direct_camoufox_error}; browser: {browser_error}; "
                        f"BrowseForge: {browseforge_error}; Camoufox: {camoufox_error}"
                    ) from (direct_camoufox_error or browser_error)
        return {"url": target_url, **options}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/v1/media/prepare")
async def prepare_media(
    background: BackgroundTasks,
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    segment_seconds: int = Form(180),
    request_id: str | None = Form(None),
    media_part: str | None = Form(None),
    media_source: str | None = Form(None),
):
    if (file is None) == (not url or not url.strip()):
        raise HTTPException(400, "provide exactly one of file or url")
    if segment_seconds < 5 or segment_seconds > 180:
        raise HTTPException(400, "segment_seconds must be between 5 and 180")
    if request_id:
        try:
            progress_path(request_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    work_dir = request_work_dir(request_id)
    begin_prepare(request_id, work_dir)
    source_name = file.filename if file else url.strip()
    asset = None
    try:
        set_progress(request_id, "starting")
        ensure_prepare_active(request_id)
        source = reusable_source(work_dir, source_name)
        if source is not None:
            set_progress(request_id, "resuming")
        elif file:
            set_progress(request_id, "receiving")
            suffix = Path(file.filename or "media.bin").suffix or ".bin"
            source = work_dir / f"source{suffix.lower()}"
            await save_upload(file, source)
        else:
            target_url = await asyncio.to_thread(validate_public_url, url)
            selection = {
                "part": (media_part or "").strip(),
                "source": (media_source or "").strip(),
            }
            selection = selection if selection["part"] or selection["source"] else None
            adapter = adapter_for_url(target_url)
            if selection and adapter.name != "supjav.com":
                raise ValueError("media part/source selection is not supported for this site")
            _, session_cookies = session_paths(target_url)
            if selection:
                set_progress(request_id, "resolving")
                selection_errors = []
                for resolver_name, resolver in (
                    ("direct Camoufox", download_via_direct_camoufox),
                    ("direct Camoufox retry", download_via_direct_camoufox),
                    ("browser", download_via_browser),
                    ("BrowseForge", download_via_browseforge),
                    ("Camoufox", download_via_camoufox),
                ):
                    ensure_prepare_active(request_id)
                    try:
                        source = await resolver(target_url, work_dir, request_id, selection)
                        break
                    except PrepareCancelled:
                        raise
                    except Exception as resolver_error:
                        selection_errors.append(f"{resolver_name}: {resolver_error}")
                else:
                    raise RuntimeError("; ".join(selection_errors))
            else:
                try:
                    source = await asyncio.to_thread(
                        yt_dlp_download, target_url, work_dir, session_cookies, {"Referer": target_url}, request_id
                    )
                except PrepareCancelled:
                    raise
                except Exception as primary_error:
                    set_progress(request_id, "resolving")
                    adapter = adapter_for_url(target_url)
                    fallback_order = (
                        (
                            ("direct Camoufox", download_via_direct_camoufox),
                            ("browser", download_via_browser),
                            ("BrowseForge", download_via_browseforge),
                            ("Camoufox", download_via_camoufox),
                        )
                        if adapter.name == "supjav.com"
                        else (
                            ("browser", download_via_browser),
                            ("direct Camoufox", download_via_direct_camoufox),
                            ("BrowseForge", download_via_browseforge),
                            ("Camoufox", download_via_camoufox),
                        )
                    )
                    fallback_errors = []
                    for fallback_name, fallback in fallback_order:
                        ensure_prepare_active(request_id)
                        try:
                            source = await fallback(target_url, work_dir, request_id)
                            break
                        except PrepareCancelled:
                            raise
                        except Exception as fallback_error:
                            fallback_errors.append(f"{fallback_name}: {fallback_error}")
                    else:
                        raise RuntimeError(
                            f"yt-dlp failed: {primary_error}; fallback failed: {'; '.join(fallback_errors)}"
                        ) from primary_error
        ensure_prepare_active(request_id)
        write_recovery(work_dir, source_name=source_name, source_file=source.name, stage="downloaded")
        asset = reusable_asset(work_dir)
        if asset is None:
            set_progress(request_id, "storing")
            asset = await asyncio.to_thread(persist_media_asset, source, source_name, request_id)
            if asset:
                write_recovery(work_dir, asset_id=asset["id"], stage="stored")
        ensure_prepare_active(request_id)
        set_progress(request_id, "extracting_audio")
        write_recovery(work_dir, stage="extracting_audio")
        segments = await asyncio.to_thread(prepare_segments, source, work_dir, segment_seconds, request_id)
        archive = await asyncio.to_thread(build_archive, work_dir, source_name, segments, asset)
    except PrepareCancelled as exc:
        set_progress(request_id, "cancelled")
        if asset:
            shutil.rmtree(ASSET_DIR / asset["id"], ignore_errors=True)
        finish_prepare(request_id, work_dir)
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        set_progress(request_id, "failed", error=str(exc))
        if asset:
            shutil.rmtree(ASSET_DIR / asset["id"], ignore_errors=True)
        if not request_id or not any(work_dir.glob("source*")):
            shutil.rmtree(work_dir, ignore_errors=True)
        finish_prepare(request_id, work_dir)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        set_progress(request_id, "failed", error=str(exc)[-1000:])
        if asset:
            shutil.rmtree(ASSET_DIR / asset["id"], ignore_errors=True)
        if not request_id or not any(work_dir.glob("source*")):
            shutil.rmtree(work_dir, ignore_errors=True)
        finish_prepare(request_id, work_dir)
        raise HTTPException(422, str(exc)) from exc
    shutil.rmtree(work_dir, ignore_errors=True)
    finish_prepare(request_id, work_dir)
    set_progress(request_id, "complete")
    background.add_task(remove_path, archive)
    return FileResponse(archive, media_type="application/zip", filename="prepared.zip", background=background)
