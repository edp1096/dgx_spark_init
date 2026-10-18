"""Download and verify the pinned GLM FP4-Local source without touching other models."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


REPO = "nvidia/GLM-5.3-Flash-FP4-Local"
REVISION = "76348ddb2be2e35e39ecfc00e75dc07703f4f3d5"


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path):
    with path.open("rb") as stream:
        result = hashlib.file_digest(stream, "sha256").hexdigest()
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return result


def verify(path, record):
    if path.stat().st_size != record["size"]:
        raise ValueError("Pinned file size mismatch: " + record["rfilename"])
    actual = sha256(path)
    if "lfs" in record:
        if actual != record["lfs"]["sha256"]:
            raise ValueError("Pinned file SHA256 mismatch: " + record["rfilename"])
    else:
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != record["blobId"]:
            raise ValueError("Pinned metadata blob mismatch: " + record["rfilename"])
    return actual


def run(root, destination, max_xet_ranges=8):
    root.mkdir(parents=True, exist_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    lock = (root / "base-download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update({
        "HF_HOME": str(root / "hf-home"),
        "HF_XET_CACHE": str(root / "xet-cache"),
        "HF_HUB_DISABLE_XET": "0",
        "HF_XET_HIGH_PERFORMANCE": "0",
        "HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS": "1",
        "HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY": str(min(4, max_xet_ranges)),
        "HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY": "1",
        "HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY": str(max_xet_ranges),
        "HF_XET_NUM_CONCURRENT_RANGE_GETS": str(max_xet_ranges),
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "HF_HUB_VERBOSITY": "error",
    })
    import requests
    import hf_xet
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils._runtime import is_xet_available

    if not is_xet_available():
        raise RuntimeError("Official Xet downloader is required")
    if hf_xet.XetConfig().get("client.ac_max_download_concurrency") != max_xet_ranges:
        raise RuntimeError("Xet concurrency bound was not applied")
    response = requests.get(
        f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}",
        params={"blobs": "true"}, timeout=60,
    )
    response.raise_for_status()
    manifest = response.json()
    if manifest["id"] != REPO or manifest["sha"] != REVISION:
        raise ValueError("Unexpected source revision")
    api_path = root / "metadata/base-api.json"
    if api_path.exists() and json.loads(api_path.read_text())["sha"] != REVISION:
        raise ValueError("Existing source manifest belongs to a different revision")
    atomic(api_path, manifest)
    records = sorted(manifest["siblings"], key=lambda r: (
        r["rfilename"].endswith(".safetensors"), r["rfilename"],
    ))
    total = sum(r["size"] for r in records)
    complete = []
    previous = root / "base-download-receipt.json"
    started = json.loads(previous.read_text())["started"] if previous.exists() else time.time()

    def status(phase, **extra):
        atomic(root / "download-progress.json", {
            "phase": phase, "repo": REPO, "revision": REVISION,
            "engine": "huggingface_hub + hf_xet", "concurrent_files": 1,
            "maximum_xet_range_requests": max_xet_ranges,
            "verified_files": len(complete), "total_files": len(records),
            "verified_bytes": sum(r["size"] for r in complete),
            "total_bytes": total, "started": started, "time": time.time(), **extra,
        })
        print(phase, extra.get("current_file", ""), len(complete), flush=True)

    for record in records:
        name = record["rfilename"]
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Unsafe source path")
        target = destination / name
        status("downloading", current_file=name, current_file_bytes=record["size"])
        if not target.exists():
            for attempt in range(6):
                try:
                    downloaded = Path(hf_hub_download(
                        repo_id=REPO, filename=name, revision=REVISION,
                        local_dir=str(destination),
                    ))
                    if downloaded.resolve() != target.resolve():
                        raise ValueError("Downloader returned an unexpected destination")
                    break
                except Exception as exc:
                    print("DOWNLOAD RETRY", name, attempt + 1, type(exc).__name__, flush=True)
                    if attempt == 5:
                        raise RuntimeError("Source download failed: " + name) from None
                    time.sleep(min(30, 2 ** attempt))
        actual = verify(target, record)
        complete.append({"rfilename": name, "size": record["size"], "sha256": actual})
        atomic(root / "base-download-receipt.json", {
            "status": "downloading", "repo": REPO, "revision": REVISION,
            "files": complete, "started": started,
        })
        status("file_verified", current_file=name)
    atomic(root / "base-download-receipt.json", {
        "status": "verified", "repo": REPO, "revision": REVISION,
        "files": complete, "started": started, "finished": time.time(),
    })
    status("verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--max-xet-ranges", type=int, choices=(1, 2, 4, 8), default=8)
    args = parser.parse_args()
    run(args.root, args.destination, args.max_xet_ranges)
