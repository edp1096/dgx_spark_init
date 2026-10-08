#!/usr/bin/env python3
"""Explicit, persistent yt-dlp updates; never update during startup/downloads."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(os.getenv("MEDIA_DATA_DIR", "/data")) / "runtimes" / "yt-dlp"
BASE_LOCK = Path(__file__).with_name("ytdlp-lock.json")


def fetch_json(url):
    with urlopen(Request(url, headers={"User-Agent": "media-access-api/1.0.2"}), timeout=30) as response:
        return json.load(response)


def latest_lock(version=None):
    endpoint = "latest"
    if version is not None:
        if not re.fullmatch(r"\d{4}\.\d{1,2}\.\d{1,2}", version):
            raise ValueError("expected YYYY.MM.DD version")
        year, month, day = map(int, version.split("."))
        endpoint = f"tags/{year:04d}.{month:02d}.{day:02d}"
    release = fetch_json("https://api.github.com/repos/yt-dlp/yt-dlp/releases/" + endpoint)
    tag = release["tag_name"]
    if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", tag):
        raise ValueError("unexpected stable release tag")
    commit = fetch_json("https://api.github.com/repos/yt-dlp/yt-dlp/commits/" + tag)["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("invalid release commit")
    version = ".".join(str(int(part)) for part in tag.split("."))
    package = fetch_json(f"https://pypi.org/pypi/yt-dlp/{version}/json")
    wheel = next(item for item in package["urls"] if item["filename"] == f"yt_dlp-{version}-py3-none-any.whl")
    if not wheel["url"].startswith("https://files.pythonhosted.org/"):
        raise ValueError("unexpected wheel host")
    return dict(version=version, tag=tag, commit=commit, url=wheel["url"], sha256=wheel["digests"]["sha256"])


def current_lock():
    path = ROOT / "current" / "lock.json"
    return json.loads((path if path.is_file() else BASE_LOCK).read_text())


def status(check=False):
    result = {"current": current_lock(), "overridden": (ROOT / "current").is_symlink(),
              "rollback_available": (ROOT / "previous").is_symlink()}
    if check:
        result["latest"] = latest_lock()
        result["update_available"] = result["current"]["sha256"] != result["latest"]["sha256"]
    return result


def link(name, destination):
    temporary = ROOT / (name + ".next")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(destination)
    temporary.replace(ROOT / name)


def install(lock):
    release = Path(tempfile.mkdtemp(prefix=lock["tag"] + "-", dir=ROOT))
    wheel = release / f"yt_dlp-{lock['version']}-py3-none-any.whl"
    with urlopen(lock["url"], timeout=60) as response:
        data = response.read(32 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != lock["sha256"]:
        raise ValueError("yt-dlp wheel SHA256 mismatch; current version unchanged")
    wheel.write_bytes(data)
    # Separate environment keeps updates out of the running API's dependencies.
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(release / "venv")], check=True, timeout=60)
    python = release / "venv/bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "--no-cache-dir", "--ignore-installed",
                    str(wheel) + "[default,curl-cffi]"], check=True, timeout=300, stdout=sys.stderr)
    binary = release / "venv/bin/yt-dlp"
    version = subprocess.check_output([str(binary), "--version"], text=True, timeout=30).strip()
    if version != lock["tag"]:
        raise ValueError("installed yt-dlp version mismatch; current version unchanged")
    (release / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    (release / "dependencies.txt").write_text(subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True))
    wheel.unlink()
    return release


def mutate(action):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "update.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if action == "update":
            lock = latest_lock()
            if current_lock()["sha256"] == lock["sha256"]:
                return {**status(), "changed": False}
            release = install(lock)
            previous = ROOT / "current"
            # A missing base sentinel means use the image's pinned version.
            link("previous", previous.resolve() if previous.is_symlink() else ROOT / "base")
            link("current", release)
        elif action == "rollback":
            previous = ROOT / "previous"
            if not previous.is_symlink():
                raise ValueError("no previous version available")
            target = previous.resolve()
            if target == ROOT / "base":
                (ROOT / "current").unlink(missing_ok=True)
            else:
                link("current", target)
            previous.unlink()
        else:
            raise ValueError("unknown operation")
        return {**status(), "changed": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["status", "check", "update", "rollback"])
    action = parser.parse_args().action
    result = status(action == "check") if action in ("status", "check") else mutate(action)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
