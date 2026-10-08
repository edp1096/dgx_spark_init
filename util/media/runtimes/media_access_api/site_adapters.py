import re
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


def _is_host(host: str, domain: str, include_subdomains: bool = False) -> bool:
    host = host.rstrip(".").lower()
    domain = domain.rstrip(".").lower()
    return host == domain or (include_subdomains and host.endswith(f".{domain}"))


def _source_name(name: str) -> str:
    """Return the page-provided provider name after an optional SERVER: prefix."""
    return re.sub(r"^\s*server\s*:\s*", "", name, count=1, flags=re.IGNORECASE).strip()


def _source_key(name: str) -> str:
    """Normalize provider spelling only for matching, never as an allowlist."""
    return re.sub(r"[^a-z0-9]+", "", _source_name(name).casefold())


@dataclass(frozen=True)
class SiteAdapter:
    name: str = "generic"
    domains: tuple[str, ...] = ()
    include_subdomains: bool = False
    browser_order: tuple[str, ...] = ("chromium", "firefox")
    prefer_browseforge: bool = False
    browseforge_attempts: int = 1

    def matches(self, host: str) -> bool:
        return any(_is_host(host, domain, self.include_subdomains) for domain in self.domains)

    async def before_detail(self, page, url: str, timeout_ms: int) -> None:
        return None

    async def after_detail(self, page, url: str, response, timeout_ms: int, candidates: list) -> object:
        return response

    def special_response(self, response_url: str, content_type: str) -> str | None:
        return None

    def browseforge_prepare_script(self) -> str | None:
        return None

    def browseforge_accept_candidate(self, candidate: str) -> bool:
        return True

    def browseforge_options(self, page_state: dict) -> dict:
        return {"site": self.name, "parts": []}

    def browseforge_extra_candidates(
        self, page_state: dict, headers: dict, timeout: int, selection: dict | None = None
    ) -> list[str]:
        return []



@dataclass(frozen=True)
class RootPreflightAdapter(SiteAdapter):
    async def before_detail(self, page, url: str, timeout_ms: int) -> None:
        parsed = urlparse(url)
        try:
            await page.goto(
                f"{parsed.scheme}://{parsed.netloc}/",
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            await page.wait_for_timeout(min(8000, timeout_ms // 3))
        except Exception:
            # A failed warm-up must not prevent a direct visit to the requested URL.
            pass


@dataclass(frozen=True)
class RetryAfterRoot403Adapter(SiteAdapter):
    async def after_detail(self, page, url: str, response, timeout_ms: int, candidates: list) -> object:
        if not response or response.status != 403:
            return response
        parsed = urlparse(url)
        try:
            await page.goto(
                f"{parsed.scheme}://{parsed.netloc}/",
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            await page.wait_for_timeout(min(10000, timeout_ms // 3))
            candidates.clear()
            return await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except Exception:
            return response


@dataclass(frozen=True)
class EnterGateAdapter(SiteAdapter):
    async def after_detail(self, page, url: str, response, timeout_ms: int, candidates: list) -> object:
        body = (await page.locator("body").inner_text()).lower()
        if not ((response and response.status >= 500) or 'please click "continue"' in body):
            return response
        parsed = urlparse(url)
        await page.goto(
            f"{parsed.scheme}://{parsed.netloc}/enter",
            wait_until="domcontentloaded",
            timeout=timeout_ms,
        )
        if parsed.path in {"", "/"}:
            return response
        candidates.clear()
        return await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)


@dataclass(frozen=True)
class VimeoAdapter(SiteAdapter):
    def special_response(self, response_url: str, content_type: str) -> str | None:
        clean_url = response_url.split("?", 1)[0].lower()
        if "/playlist/" in clean_url and clean_url.endswith("playlist.json"):
            return "vimeo_playlist"
        return None


@dataclass(frozen=True)
class SupJavAdapter(RootPreflightAdapter):
    def browseforge_options(self, page_state: dict) -> dict:
        parts = []
        raw_parts = page_state.get("parts") or []
        if not raw_parts and page_state.get("servers"):
            raw_parts = [{"id": "1", "label": "1", "sources": page_state["servers"]}]
        for fallback_index, part in enumerate(raw_parts, start=1):
            if not isinstance(part, dict):
                continue
            sources = []
            seen = set()
            for source in part.get("sources") or []:
                name = _source_name(str(source.get("name") or "")) if isinstance(source, dict) else ""
                if name and name.casefold() not in seen:
                    seen.add(name.casefold())
                    sources.append({"id": name, "label": name})
            if sources:
                identifier = str(part.get("id") or fallback_index)
                parts.append({
                    "id": identifier,
                    "label": str(part.get("label") or identifier),
                    "sources": sources,
                })
        return {"site": self.name, "parts": parts}

    def browseforge_accept_candidate(self, candidate: str) -> bool:
        host = (urlparse(candidate).hostname or "").lower()
        # These are live-cam widgets rendered beside the real player.
        return not (
            _is_host(host, "growcdnssedge.com", include_subdomains=True)
            or _is_host(host, "stripchat.mov", include_subdomains=True)
        )

    def browseforge_extra_candidates(
        self, page_state: dict, headers: dict, timeout: int, selection: dict | None = None
    ) -> list[str]:
        parts = [part for part in page_state.get("parts") or [] if isinstance(part, dict)]
        selected_part = str((selection or {}).get("part") or "1")
        part = next((item for item in parts if str(item.get("id")) == selected_part), None)
        if part is None and not selection and parts:
            part = parts[0]
        raw_servers = part.get("sources") if part else page_state.get("servers") or []
        servers = [server for server in raw_servers if isinstance(server, dict) and server.get("link")]
        selected_source = _source_key(str((selection or {}).get("source") or ""))
        if selected_source:
            servers = [
                server for server in servers
                if _source_key(str(server.get("name") or "")) == selected_source
            ]
        servers.sort(key=lambda server: (
            _source_key(str(server.get("name") or "")) not in {"st", "streamtape"},
            int(server.get("index") or 0),
        ))
        request_headers = {
            "User-Agent": headers.get("User-Agent", "Mozilla/5.0"),
            "Referer": page_state.get("url") or headers.get("Referer", ""),
        }
        for server in servers:
            wrapper_url = (
                "https://lk1.supremejav.com/supjav.php?"
                + urlencode({"c": str(server["link"])[::-1]})
            )
            try:
                with urlopen(Request(wrapper_url, headers=request_headers), timeout=timeout) as response:
                    html = response.read(2 << 20).decode("utf-8", errors="replace")
            except OSError:
                continue

            # StreamTape deliberately assembles get_video in JavaScript. The
            # final script literal contains the usable token; earlier DOM values
            # are decoys.
            matches = re.findall(
                r"[?&]id=([^&'\"<>]+)&expires=([^&'\"<>]+)&ip=([^&'\"<>]+)&token=([A-Za-z0-9_-]+)",
                html,
            )
            if matches:
                file_id, expires, ip_value, token = matches[-1]
                return [
                    "https://streamtape.com/get_video?"
                    + urlencode({"id": file_id, "expires": expires, "ip": ip_value, "token": token})
                ]

            direct_media = re.findall(
                r'''https?://[^\s'"<>]+?\.(?:m3u8|mpd|mp4|webm)(?:\?[^\s'"<>]*)?''',
                html,
                flags=re.IGNORECASE,
            )
            if direct_media:
                return list(dict.fromkeys(direct_media))

            embed = re.search(
                r'''<meta[^>]+(?:name|property)=["']og:url["'][^>]+content=["']([^"']+)["']''',
                html,
                flags=re.IGNORECASE,
            )
            if embed:
                return [embed.group(1)]
        return []


# Similar names do not imply a shared service. Each host family is deliberately
# registered independently, and adapters never rewrite a request to another host.
SITE_ADAPTERS: tuple[SiteAdapter, ...] = (
    VimeoAdapter(
        name="vimeo",
        domains=("vimeo.com",),
        include_subdomains=True,
        browser_order=("firefox", "chromium"),
    ),
    SupJavAdapter(
        name="supjav.com",
        domains=("supjav.com",),
        include_subdomains=True,
        prefer_browseforge=True,
    ),
    SiteAdapter(
        name="missav123.com",
        domains=("missav123.com",),
        include_subdomains=True,
        prefer_browseforge=True,
    ),
    RetryAfterRoot403Adapter(
        name="missav888.com",
        domains=("missav888.com",),
        include_subdomains=True,
        browser_order=("firefox", "chromium"),
    ),
    EnterGateAdapter(name="missav888.net", domains=("missav888.net",), include_subdomains=True),
    EnterGateAdapter(name="missav888.org", domains=("missav888.org",), include_subdomains=True),
)

GENERIC_ADAPTER = SiteAdapter()


def adapter_for_url(url: str) -> SiteAdapter:
    host = (urlparse(url).hostname or "").lower()
    return next((adapter for adapter in SITE_ADAPTERS if adapter.matches(host)), GENERIC_ADAPTER)
