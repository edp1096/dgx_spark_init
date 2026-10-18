"""Fetch only the 44 revision-pinned, previously audited Huihui Q8 tensor ranges."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import requests

from download import atomic, sha256


SOURCES = {
    "original": ["unsloth/GLM-5.3-Flash-GGUF", "621d456e93e926e4b52f85cff5f634358c1828f9"],
    "huihui": ["huihui-ai/GLM-5.3-Flash-abliterated-GGUF", "50d39f500cbd1d0478295da89d417607521d378b"],
}


def audited(root):
    report = json.loads((root / "metadata/gguf-audit.json").read_text())
    if (report["status"] != "complete" or report["tensor_count"] != 1412
            or len(report["tensors"]) != 1412 or report["changed_count"] != 44
            or report["sources"] != SOURCES):
        raise ValueError("Unexpected full GGUF audit or source revisions")
    changed = [r for r in report["tensors"] if r["changed"]]
    expected = {f"blk.{i}.{suffix}.weight" for i in range(15, 37)
                for suffix in ("attn_output", "ffn_down_shexp")}
    if len(changed) != 44 or {r["name"] for r in changed} != expected:
        raise ValueError("Changed projection set does not match the audited model")
    if any(r["type"] != 8 for r in changed):
        raise ValueError("All audited projection ranges must be Q8_0")
    return report, changed


def fetch(session, root, row, progress):
    destination = root / "ranges/huihui" / (row["name"] + ".bin")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != row["bytes"] or sha256(destination) != row["huihui_sha256"]:
            raise ValueError("Existing Huihui tensor failed its audited hash")
        return
    temporary = destination.with_suffix(".partial")
    repo, revision = SOURCES["huihui"]
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{row['pair']}"
    for attempt in range(6):
        done = temporary.stat().st_size if temporary.exists() else 0
        if done > row["bytes"]:
            raise ValueError("Partial tensor exceeds audited range")
        try:
            if done < row["bytes"]:
                start = row["huihui_offset"] + done
                end = row["huihui_offset"] + row["bytes"] - 1
                with session.get(url, params={"audit_range": f"{start}-{end}"},
                                 headers={"Range": f"bytes={start}-{end}"},
                                 stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                        raise ValueError("Server did not honor the exact audited tensor range")
                    with temporary.open("ab") as out:
                        for block in response.iter_content(4 << 20):
                            if done + len(block) > row["bytes"]:
                                raise ValueError("Server exceeded the audited tensor range")
                            out.write(block)
                            done += len(block)
                            progress(row["name"], done)
            if temporary.stat().st_size != row["bytes"]:
                raise ValueError("Truncated Huihui tensor")
            if sha256(temporary) != row["huihui_sha256"]:
                raise ValueError("Huihui tensor does not match the previous full audit")
            temporary.replace(destination)
            return
        except requests.RequestException as exc:
            print("RANGE RETRY", row["name"], attempt + 1, type(exc).__name__, flush=True)
            if attempt == 5:
                raise RuntimeError("Audited range download failed") from None
            time.sleep(min(30, 2 ** attempt))


def run(root):
    lock = (root / "delta-download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    report, changed = audited(root)
    complete = []
    started = time.time()
    total = sum(r["bytes"] for r in changed)
    last_progress = 0

    def status(phase, **extra):
        atomic(root / "delta-progress.json", {
            "phase": phase, "verified_tensors": len(complete), "total_tensors": 44,
            "verified_bytes": sum(r["bytes"] for r in complete), "total_bytes": total,
            "started": started, "time": time.time(), **extra,
        })

    for row in changed:
        path = root / "ranges/original" / (row["name"] + ".bin")
        status("verifying_original", current_tensor=row["name"])
        if path.stat().st_size != row["bytes"] or sha256(path) != row["original_sha256"]:
            raise ValueError("Copied original tensor failed its audited hash")
    print("ORIGINAL RANGES VERIFIED", len(changed), total, flush=True)

    def progress(name, done):
        nonlocal last_progress
        if time.monotonic() - last_progress > 5:
            status("downloading_huihui", current_tensor=name, current_bytes=done)
            last_progress = time.monotonic()

    with requests.Session() as session:
        for row in changed:
            status("downloading_huihui", current_tensor=row["name"], current_bytes=0)
            fetch(session, root, row, progress)
            complete.append(row)
            print("DELTA RANGE VERIFIED", row["name"], flush=True)
    atomic(root / "delta-download-receipt.json", {
        "status": "verified", "sources": report["sources"],
        "audit_sha256": sha256(root / "metadata/gguf-audit.json"),
        "tensor_count": len(complete), "bytes_per_side": total,
        "ranges": [{k: r[k] for k in ("name", "bytes", "original_sha256", "huihui_sha256")}
                   for r in complete], "started": started, "finished": time.time(),
    })
    status("verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
