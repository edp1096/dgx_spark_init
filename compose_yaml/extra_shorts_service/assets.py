"""Project-scoped animation assets, with source and license evidence attached."""
import base64
import datetime
import hashlib
import html
import io
import json
import re
import uuid
import zipfile
from pathlib import Path
from urllib.parse import quote, urlparse, urljoin
from xml.etree import ElementTree

import httpx

import store

LICENSE = "https://lottiefiles.com/page/license"
PAGE_HOSTS = {"lottiefiles.com", "www.lottiefiles.com"}
FILE_HOSTS = {"assets-v2.lottiefiles.com", "lottie.host", *(f"assets{i}.lottiefiles.com" for i in range(1, 11))}
LIMIT = 12 << 20


def _fetch(url, hosts, limit=LIMIT):
    with httpx.Client(timeout=30, headers={"User-Agent": "Mozilla/5.0 SparkTalk/0.2"}) as client:
        for _ in range(5):
            parsed = urlparse(url)
            if parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.port not in (None, 443):
                raise ValueError("허용된 공개 소재 주소가 아닙니다")
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["location"])
                    continue
                response.raise_for_status()
                chunks, total = [], 0
                for part in response.iter_bytes():
                    total += len(part)
                    if total > limit:
                        raise ValueError("소재 파일 크기 한도 초과")
                    chunks.append(part)
                return b"".join(chunks)
    raise ValueError("소재 주소 리디렉션이 너무 많습니다")


def _objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def metadata(page):
    result = []
    for raw in re.findall(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', page, re.S):
        try:
            result.extend(_objects(json.loads(raw)))
        except json.JSONDecodeError:
            continue
    return result


def search(query, limit=8):
    query = re.sub(r"[^\w -]", "", query.strip())[:70]
    if not query:
        raise ValueError("검색어가 필요합니다")
    url = "https://lottiefiles.com/free-animations/" + quote(query.replace(" ", "-"))
    page = _fetch(url, PAGE_HOSTS, 4 << 20).decode("utf-8")
    rows, seen = [], set()
    for item in metadata(page):
        source = item.get("acquireLicensePage", "")
        if item.get("@type") != "ImageObject" or item.get("license") != LICENSE or not valid_page(source) or source in seen:
            continue
        seen.add(source)
        rows.append({"title": item.get("name", ""), "source_url": source,
                     "preview_url": item.get("thumbnailUrl", item.get("contentUrl", "")),
                     "license": "Lottie Simple License", "license_url": LICENSE,
                     "creator": (item.get("creator") or {}).get("name", ""), "commercial_use": True})
        if len(rows) >= min(12, max(1, limit)):
            break
    return {"query": query, "search_url": url, "results": rows,
            "guidance": "Choose relevant assets, then import their source_url. Search English keywords or use web_search for other public free-animation pages."}


def valid_page(url):
    parsed = urlparse(url)
    return parsed.scheme == "https" and parsed.hostname in PAGE_HOSTS and re.fullmatch(r"/(?:[a-z]{2}/)?free-animation/[\w-]+/?", parsed.path) is not None


def validate_lottie(animation):
    if not isinstance(animation, dict) or not isinstance(animation.get("layers"), list):
        raise ValueError("Lottie layers가 없습니다")
    for key in ("w", "h", "fr", "ip", "op"):
        if not isinstance(animation.get(key), (int, float)):
            raise ValueError(f"Lottie {key} 값이 잘못됐습니다")
    if not (1 <= animation["fr"] <= 120 and 0 < animation["w"] <= 4096 and 0 < animation["h"] <= 4096 and 0 < (animation["op"]-animation["ip"])/animation["fr"] <= 180):
        raise ValueError("Lottie 크기·길이 한도 초과")
    if len(animation["layers"]) > 400 or len(json.dumps(animation)) > LIMIT:
        raise ValueError("Lottie 복잡도 한도 초과")
    for item in _objects(animation):
        if isinstance(item.get("x"), str):
            raise ValueError("실행식이 포함된 Lottie는 지원하지 않습니다. 벡터 키프레임으로 내보내세요.")
    for item in animation.get("assets", []):
        if item.get("p") and not item["p"].startswith(("data:image/png;base64,", "data:image/jpeg;base64,", "data:image/webp;base64,")):
            raise ValueError("Lottie 외부 이미지가 남아 있습니다")
    return animation


def unpack_lottie(data):
    if not data.startswith(b"PK"):
        return validate_lottie(json.loads(data))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 300 or sum(e.file_size for e in entries) > 24 << 20:
            raise ValueError("dotLottie 압축 해제 크기 한도 초과")
        for entry in entries:
            if entry.filename.startswith("/") or ".." in Path(entry.filename).parts:
                raise ValueError("잘못된 dotLottie 경로")
        candidates = [e.filename for e in entries if e.filename.startswith("animations/") and e.filename.endswith(".json")]
        if not candidates:
            raise ValueError("dotLottie에 animation JSON이 없습니다")
        manifest = json.loads(archive.read("manifest.json")) if "manifest.json" in archive.namelist() else {}
        preferred = "animations/" + str((manifest.get("animations") or [{}])[0].get("id", "")) + ".json"
        selected = preferred if preferred in candidates else candidates[0]
        animation = json.loads(archive.read(selected))
        for item in animation.get("assets", []):
            if not item.get("p") or item["p"].startswith("data:"):
                continue
            filename = "images/" + Path(item["p"]).name
            if filename not in archive.namelist():
                raise ValueError("dotLottie 이미지가 없습니다")
            ext = Path(filename).suffix.lower()
            mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(ext)
            if not mime:
                raise ValueError("dotLottie 이미지 형식 미지원")
            item.update(u="", p="data:"+mime+";base64,"+base64.b64encode(archive.read(filename)).decode(), e=1)
    return validate_lottie(animation)


def session_root(session_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", session_id):
        raise ValueError("잘못된 대화 ID")
    return store.ROOT / "_assets" / session_id


def save_asset(session_id, name, kind, data, evidence):
    asset_id = uuid.uuid4().hex
    root = session_root(session_id) / asset_id
    root.mkdir(parents=True)
    filename = "animation.json" if kind == "lottie" else "drawing.svg"
    (root / filename).write_bytes(data)
    record = {"id": asset_id, "name": name[:120], "kind": kind, "filename": filename,
              "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
              "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **evidence}
    (root / "source.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def import_lottie(session_id, source_url):
    if not valid_page(source_url):
        raise ValueError("LottieFiles의 공개 무료 animation 페이지 주소가 필요합니다")
    page = _fetch(source_url, PAGE_HOSTS, 4 << 20).decode("utf-8")
    item = next((x for x in metadata(page) if x.get("@type") == "ImageObject" and x.get("acquireLicensePage", "").rstrip("/") == source_url.rstrip("/")), None)
    if not item or item.get("license") != LICENSE:
        raise ValueError("이 소재 페이지에서 상용 사용 가능한 무료 라이선스를 확인하지 못했습니다")
    decoded = html.unescape(page.replace('\\"', '"').replace("\\/", "/"))
    match = re.search(r'"lottiePath"\s*[:,]\s*"(https://[^" ]+\.lottie)"', decoded)
    if not match:
        raise ValueError("공개 다운로드 파일을 찾지 못했습니다. 다른 무료 소재를 선택하세요.")
    download_url = match.group(1)
    animation = unpack_lottie(_fetch(download_url, FILE_HOSTS))
    record = save_asset(session_id, item.get("name", "Lottie"), "lottie", json.dumps(animation).encode(),
                        {"source_url": source_url, "download_url": download_url, "license": "Lottie Simple License",
                         "license_url": LICENSE, "commercial_use": True, "creator": (item.get("creator") or {}).get("name", ""),
                         "attribution": item.get("creditText", ""), "license_verified_from": "animation page JSON-LD"})
    # Keep the applicable terms with every selected asset, not a scraped library.
    (session_root(session_id) / record["id"] / "license.html").write_bytes(_fetch(LICENSE, PAGE_HOSTS, 2 << 20))
    return record


def create_asset(session_id, name, kind, content):
    if kind == "lottie":
        data = json.dumps(validate_lottie(content if isinstance(content, dict) else json.loads(content))).encode()
    elif kind == "svg":
        if not isinstance(content, str) or len(content) > 200000 or re.search(r"<!DOCTYPE|<!ENTITY", content, re.I):
            raise ValueError("잘못된 SVG")
        root = ElementTree.fromstring(content)
        for element in root.iter():
            tag = element.tag.rsplit("}", 1)[-1].lower()
            if tag in ("script", "foreignobject", "image", "use", "style") or any(k.lower().startswith("on") or k.rsplit("}", 1)[-1] in ("href", "src") or "url(" in v.lower() for k, v in element.attrib.items()):
                raise ValueError("SVG에는 외부 참조·스크립트를 넣을 수 없습니다")
        data = content.encode()
    else:
        raise ValueError("새 벡터 소재는 lottie 또는 svg 형식이어야 합니다")
    return save_asset(session_id, name, kind, data, {"source_url": "", "license": "Project original", "commercial_use": True, "creator": "Local model", "attribution": "Created for this project"})


def list_assets(session_id):
    return [json.loads(p.read_text()) for p in sorted(session_root(session_id).glob("*/source.json"), reverse=True)][:100]


def get_asset(session_id, asset_id):
    if not re.fullmatch(r"[a-f0-9]{32}", asset_id):
        raise ValueError("잘못된 소재 ID")
    path = session_root(session_id) / asset_id
    if not (path / "source.json").exists():
        raise ValueError("이 대화에 소재가 없습니다")
    return json.loads((path / "source.json").read_text()), path
