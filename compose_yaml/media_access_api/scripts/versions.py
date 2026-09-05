#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "versions.env"
YOUTUBE_TEST_URL = "https://www.youtube.com/watch?v=AK_Tq3QyYVM"
COMPONENTS = {
    "bgutil": {
        "version": "POT_PROVIDER_VERSION",
        "digest": "POT_PROVIDER_DIGEST",
        "latest": "https://pypi.org/pypi/bgutil-ytdlp-pot-provider/json",
        "image": lambda version: f"brainicism/bgutil-ytdlp-pot-provider:{version}-deno",
    },
    "browseforge": {
        "version": "BROWSEFORGE_VERSION",
        "digest": "BROWSEFORGE_DIGEST",
        "latest": "https://api.github.com/repos/nczz/BrowseForge/tags?per_page=100",
        "image": lambda version: f"ghcr.io/nczz/browseforge:{version}",
    },
    "playwright": {
        "version": "PLAYWRIGHT_VERSION",
        "digest": "PLAYWRIGHT_DIGEST",
        "latest": "https://pypi.org/pypi/playwright/json",
        "image": lambda version: f"mcr.microsoft.com/playwright/python:v{version}-noble",
    },
    "yt-dlp": {
        "version": "YTDLP_VERSION",
        "latest": "https://pypi.org/pypi/yt-dlp/json",
    },
    "camoufox": {
        "version": "CAMOUFOX_VERSION",
        "latest": "https://pypi.org/pypi/camoufox/json",
    },
    "playwright-captcha": {
        "version": "PLAYWRIGHT_CAPTCHA_VERSION",
        "latest": "https://pypi.org/pypi/playwright-captcha/json",
    },
}


def load_env():
    values = {}
    for raw_line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def run(*args, capture=False, env=None):
    return subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )


def fetch_json(url):
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "media-access-version-check/1"})
    with urlopen(request, timeout=20) as response:
        return json.load(response)


def latest_version(component):
    spec = COMPONENTS[component]
    payload = fetch_json(spec["latest"])
    if component == "browseforge":
        versions = []
        for tag in payload:
            match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag.get("name", ""))
            if match:
                versions.append((tuple(map(int, match.groups())), tag["name"]))
        if not versions:
            raise RuntimeError("BrowseForge did not publish a semantic version tag")
        return max(versions)[1]
    return payload["info"]["version"]


def normalize(version):
    return version.removeprefix("v")


def show():
    values = load_env()
    for component, spec in COMPONENTS.items():
        suffix = f"  {values[spec['digest']]}" if spec.get("digest") else ""
        print(f"{component:12} {values[spec['version']]}{suffix}")
    print(f"{'image tag':12} {values['MEDIA_ACCESS_IMAGE_TAG']}")


def check():
    values = load_env()
    updates = 0
    for component, spec in COMPONENTS.items():
        current = values[spec["version"]]
        latest = latest_version(component)
        state = "current" if normalize(current) == normalize(latest) else "update available"
        updates += state != "current"
        print(f"{component:12} current={current:12} latest={latest:12} {state}")
    if updates:
        print("\nNo files were changed. Apply an approved version with:")
        print("  make set-version COMPONENT=<name> VERSION=<version>")


def image_digest(image):
    result = run(
        "docker", "buildx", "imagetools", "inspect", image,
        "--format", "{{json .Manifest}}", capture=True,
    )
    payload = json.loads(result.stdout)
    digest = payload.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RuntimeError(f"could not resolve manifest digest for {image}")
    return digest


def replace_values(changes):
    content = ENV_FILE.read_text(encoding="utf-8")
    for key, value in changes.items():
        content, count = re.subn(rf"(?m)^{re.escape(key)}=.*$", f"{key}={value}", content)
        if count != 1:
            raise RuntimeError(f"missing or duplicate setting: {key}")
    temporary = ENV_FILE.with_suffix(".env.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(ENV_FILE)


def set_version(component, version):
    if component not in COMPONENTS or not version:
        names = ", ".join(COMPONENTS)
        raise SystemExit(f"usage: set <component> <version>; components: {names}")
    spec = COMPONENTS[component]
    changes = {spec["version"]: version}
    if component == "yt-dlp":
        sys.path.insert(0, str(ROOT))
        from ytdlp_runtime import latest_lock
        lock = latest_lock(version)
        changes[spec["version"]] = lock["version"]
        (ROOT / "ytdlp-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
        replace_values(changes)
        print("Pinned yt-dlp release commit and wheel SHA256; run make rebuild.")
        return
    if spec.get("image"):
        image = spec["image"](version)
        print(f"Resolving {image} ...")
        changes[spec["digest"]] = image_digest(image)
    else:
        package = {
            "yt-dlp": "yt-dlp",
            "camoufox": "camoufox",
            "playwright-captcha": "playwright-captcha",
        }[component]
        payload = fetch_json(f"https://pypi.org/pypi/{package}/{version}/json")
        if payload["info"]["version"] != version:
            raise RuntimeError(f"PyPI returned an unexpected version for {version}")
    replace_values(changes)
    print(f"Updated versions.env for {component} {version}.")
    print("Review the diff, then run: make rebuild && make verify")


def verify():
    with urlopen("http://127.0.0.1:8697/health", timeout=10) as response:
        health = json.load(response)
    required = ("ffmpeg", "yt_dlp", "browseforge", "camoufox", "pot_provider")
    missing = [name for name in required if not health.get(name)]
    if missing:
        raise RuntimeError("unhealthy components: " + ", ".join(missing))

    names = run("docker", "ps", "--format", "{{.Names}}", capture=True).stdout.splitlines()
    if "media-access-api" not in names or "media-pot-provider" in names:
        raise RuntimeError("expected one media-access-api container and no media-pot-provider container")
    run(
        "docker", "exec", "media-access-api", "sh", "-c",
        "test -s /usr/share/licenses/bgutil-ytdlp-pot-provider/GPL-3 "
        "&& test -s /usr/share/licenses/browseforge/LICENSE "
        "&& test -s /usr/share/licenses/camoufox/MPL-2.0 "
        "&& test -s /usr/share/licenses/playwright-captcha/LICENSE",
    )
    result = run(
        "docker", "exec", "media-access-api", "yt-dlp", "-v", "--simulate",
        "--extractor-args", "youtube:player_client=mweb", YOUTUBE_TEST_URL,
        capture=True,
    )
    expected = ("PO Token Providers: bgutil:http", "Retrieved a gvs PO Token", "Downloading 1 format(s)")
    absent = [marker for marker in expected if marker not in result.stdout]
    if absent:
        print(result.stdout)
        raise RuntimeError("YouTube verification missed: " + ", ".join(absent))
    print("Verified health, licenses, single-container topology, Deno JS solver, and YouTube PO Token generation.")


def checked_tag(tag):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", tag or ""):
        raise SystemExit("a valid TAG is required")
    return tag


def snapshot(tag):
    tag = checked_tag(tag)
    image = "ghcr.io/edp1096/media-access-api"
    run("docker", "tag", f"{image}:latest", f"{image}:{tag}")
    print(f"Saved {image}:{tag}")


def rollback(tag):
    tag = checked_tag(tag)
    environment = os.environ.copy()
    environment["MEDIA_ACCESS_IMAGE_TAG"] = tag
    run("docker", "compose", "--env-file", "versions.env", "up", "-d", "--no-build", env=environment)
    print(f"Running ghcr.io/edp1096/media-access-api:{tag}")


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else "show"
    if command == "show":
        show()
    elif command == "check":
        check()
    elif command == "set":
        set_version(sys.argv[2] if len(sys.argv) > 2 else "", sys.argv[3] if len(sys.argv) > 3 else "")
    elif command == "verify":
        verify()
    elif command == "snapshot":
        snapshot(sys.argv[2] if len(sys.argv) > 2 else "")
    elif command == "rollback":
        rollback(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    main()
