from __future__ import annotations

import html
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("search-assistant-domestic-rss")
_DEFAULT_RSSHUB_URL = "http://127.0.0.1:1200"
_DEFAULT_TIMEOUT_SECONDS = 8.0
_USER_AGENT = "search-assistant-domestic-rss/0.1"
_PLACEHOLDER = re.compile(r"{([A-Za-z_][A-Za-z0-9_]*)}")
_SITE_DIRECTIVE = re.compile(r"(?<!\S)site:[^\s]+", flags=re.IGNORECASE)
_MAX_ITEMS_PER_FEED = 80


@dataclass(frozen=True)
class DomesticFeedSource:
    source_id: str
    platform: str
    kind: str
    enabled: bool
    requires_login: bool
    requires_browser: bool
    requires_external_subscription: bool
    route_template: str | None
    feed_url: str | None
    domains: tuple[str, ...]
    parameters: dict[str, str]


class DomesticRssFeedSearcher:
    """Read only public RSS/Atom/JSON feeds produced by a local RSSHub instance."""

    def __init__(
        self,
        catalog_path: str | Path,
        rsshub_base_url: str = _DEFAULT_RSSHUB_URL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        transport: Callable[[str], str] | None = None,
    ):
        self.catalog_path = Path(catalog_path).resolve()
        self.rsshub_base_url = _normalize_base_url(rsshub_base_url)
        self.timeout_seconds = max(0.5, timeout_seconds)
        self.transport = transport or _http_get
        self.sources = _load_sources(self.catalog_path)

    @classmethod
    def from_environment(cls) -> "DomesticRssFeedSearcher":
        project_root = Path(__file__).resolve().parents[3]
        catalog_path = Path(
            os.environ.get("SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG")
            or project_root / "configs" / "domestic-rss.sources.json"
        )
        return cls(
            catalog_path=catalog_path,
            rsshub_base_url=os.environ.get("RSSHUB_BASE_URL", _DEFAULT_RSSHUB_URL),
            timeout_seconds=_positive_float(os.environ.get("DOMESTIC_RSS_TIMEOUT_SECONDS"), _DEFAULT_TIMEOUT_SECONDS),
        )

    def search(self, query: str, limit: int = 10) -> list[dict[str, str]]:
        clean_query = _query_without_site_directives(query)
        if not clean_query:
            return []

        bounded_limit = max(1, min(limit, 30))
        requested_domains = _site_domains(query)
        feeds: list[tuple[int, DomesticFeedSource, str]] = []
        for index, source in enumerate(self.sources):
            if not _source_is_eligible(source, requested_domains):
                continue
            feed_url = _feed_url_for_source(source, clean_query, self.rsshub_base_url)
            if feed_url:
                feeds.append((index, source, feed_url))

        if not feeds:
            return []

        fetched: list[tuple[int, DomesticFeedSource, list[dict[str, str]]]] = []
        with ThreadPoolExecutor(max_workers=min(6, len(feeds))) as executor:
            futures = {
                executor.submit(_fetch_feed_items, self.transport, feed_url): (index, source)
                for index, source, feed_url in feeds
            }
            for future, (index, source) in futures.items():
                try:
                    fetched.append((index, source, future.result()))
                except OSError:
                    continue

        scored: list[tuple[int, int, int, dict[str, str]]] = []
        checked_at = datetime.now(UTC).isoformat()
        for source_index, source, items in fetched:
            for item_index, item in enumerate(items):
                url = _normalized_http_url(item.get("url", ""))
                if not url or not _matches_domains(url, source.domains):
                    continue
                if requested_domains and not _matches_domains(url, requested_domains):
                    continue
                title = _clean_text(item.get("title", ""))
                snippet = _clean_text(item.get("snippet", ""))
                relevance = _relevance_score(f"{title} {snippet}", clean_query)
                if relevance <= 0:
                    continue
                scored.append(
                    (
                        relevance,
                        -source_index,
                        -item_index,
                        {
                            "title": title or url,
                            "url": url,
                            "snippet": snippet,
                            "provider": f"{source.platform}:{source.source_id}",
                            "checked_at": checked_at,
                        },
                    )
                )

        merged: list[dict[str, str]] = []
        seen: set[str] = set()
        for _, _, _, item in sorted(scored, reverse=True):
            if item["url"] in seen:
                continue
            seen.add(item["url"])
            merged.append(item)
            if len(merged) >= bounded_limit:
                break
        return merged

    def source_status(self) -> dict[str, object]:
        sources: list[dict[str, object]] = []
        for source in self.sources:
            reason = "enabled"
            if not source.enabled:
                reason = "disabled"
            elif source.requires_login:
                reason = "requires_login"
            elif _feed_url_for_source(source, "probe", self.rsshub_base_url) is None:
                reason = "missing_required_parameters"
            sources.append(
                {
                    "id": source.source_id,
                    "platform": source.platform,
                    "status": reason,
                    "requires_browser": source.requires_browser,
                    "requires_external_subscription": source.requires_external_subscription,
                    "domains": list(source.domains),
                }
            )
        return {
            "catalog_path": str(self.catalog_path),
            "rsshub_base_url": self.rsshub_base_url,
            "sources": sources,
        }


@mcp.tool()
def search_domestic_rss(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Search configured domestic public RSS feeds and return original platform URLs only."""
    try:
        return DomesticRssFeedSearcher.from_environment().search(query, limit)
    except (OSError, ValueError, json.JSONDecodeError):
        return []


@mcp.tool()
def list_domestic_rss_sources() -> dict[str, object]:
    """Show configured domestic sources and clearly flag disabled or login-gated routes."""
    try:
        return DomesticRssFeedSearcher.from_environment().source_status()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"error": str(exc), "sources": []}


def _load_sources(path: Path) -> list[DomesticFeedSource]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise ValueError("Domestic RSS catalog must contain a sources array")

    sources: list[DomesticFeedSource] = []
    for item in raw["sources"]:
        if not isinstance(item, dict):
            continue
        source = _source_from_mapping(item)
        if source is not None:
            sources.append(source)
    return sources


def _source_from_mapping(item: dict[str, Any]) -> DomesticFeedSource | None:
    source_id = str(item.get("id") or "").strip()
    platform = str(item.get("platform") or "").strip().lower()
    kind = str(item.get("kind") or "subscription").strip().lower()
    route_template = _optional_string(item.get("route_template"))
    feed_url = _optional_string(item.get("feed_url"))
    if not source_id or not platform or (not route_template and not feed_url):
        return None

    raw_domains = item.get("domains", [])
    if isinstance(raw_domains, str):
        raw_domains = [raw_domains]
    if not isinstance(raw_domains, list):
        return None
    domains = tuple(domain for domain in (_normalize_domain(str(value)) for value in raw_domains) if domain)
    if not domains:
        return None

    raw_parameters = item.get("parameters", {})
    if not isinstance(raw_parameters, dict):
        return None
    parameters = {str(key): str(value).strip() for key, value in raw_parameters.items() if str(value).strip()}
    return DomesticFeedSource(
        source_id=source_id,
        platform=platform,
        kind=kind,
        enabled=bool(item.get("enabled", False)),
        requires_login=bool(item.get("requires_login", False)),
        requires_browser=bool(item.get("requires_browser", False)),
        requires_external_subscription=bool(item.get("requires_external_subscription", False)),
        route_template=route_template,
        feed_url=feed_url,
        domains=domains,
        parameters=parameters,
    )


def _source_is_eligible(source: DomesticFeedSource, requested_domains: tuple[str, ...]) -> bool:
    if not source.enabled or source.requires_login:
        return False
    if not requested_domains:
        return True
    return any(
        configured == requested
        or configured.endswith(f".{requested}")
        or requested.endswith(f".{configured}")
        for configured in source.domains
        for requested in requested_domains
    )


def _feed_url_for_source(source: DomesticFeedSource, query: str, rsshub_base_url: str) -> str | None:
    if source.feed_url:
        return source.feed_url
    if not source.route_template:
        return None
    values = {"query": query, **source.parameters}
    placeholders = _PLACEHOLDER.findall(source.route_template)
    if any(not values.get(placeholder, "").strip() for placeholder in placeholders):
        return None
    route = source.route_template.format(**{key: quote(value, safe="") for key, value in values.items()})
    if not route.startswith("/"):
        route = f"/{route}"
    return f"{rsshub_base_url}{route}"


def _fetch_feed_items(transport: Callable[[str], str], feed_url: str) -> list[dict[str, str]]:
    payload = transport(feed_url)
    return _parse_feed(payload)


def _parse_feed(payload: str) -> list[dict[str, str]]:
    stripped = payload.lstrip()
    if stripped.startswith("{"):
        return _parse_json_feed(stripped)
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        return []
    root_name = _local_name(root.tag)
    if root_name == "feed":
        return _parse_atom_feed(root)
    return _parse_rss_feed(root)


def _parse_rss_feed(root: ElementTree.Element) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for node in root.iter():
        if _local_name(node.tag) != "item":
            continue
        title = _child_text(node, "title")
        url = _child_text(node, "link") or _child_text(node, "guid")
        snippet = _child_text(node, "description") or _child_text(node, "encoded")
        items.append({"title": title, "url": url, "snippet": snippet})
        if len(items) >= _MAX_ITEMS_PER_FEED:
            break
    return items


def _parse_atom_feed(root: ElementTree.Element) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for node in root:
        if _local_name(node.tag) != "entry":
            continue
        url = ""
        for child in node:
            if _local_name(child.tag) != "link":
                continue
            href = str(child.attrib.get("href") or "").strip()
            if href and str(child.attrib.get("rel") or "alternate") in {"alternate", ""}:
                url = href
                break
            if href and not url:
                url = href
        items.append(
            {
                "title": _child_text(node, "title"),
                "url": url or _child_text(node, "id"),
                "snippet": _child_text(node, "summary") or _child_text(node, "content"),
            }
        )
        if len(items) >= _MAX_ITEMS_PER_FEED:
            break
    return items


def _parse_json_feed(payload: str) -> list[dict[str, str]]:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return []
    raw_items = data.get("items", []) if isinstance(data, dict) else []
    if not isinstance(raw_items, list):
        return []
    items: list[dict[str, str]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        items.append(
            {
                "title": str(raw_item.get("title") or ""),
                "url": str(raw_item.get("external_url") or raw_item.get("url") or ""),
                "snippet": str(raw_item.get("content_text") or raw_item.get("summary") or raw_item.get("content_html") or ""),
            }
        )
        if len(items) >= _MAX_ITEMS_PER_FEED:
            break
    return items


def _child_text(node: ElementTree.Element, name: str) -> str:
    for child in node:
        if _local_name(child.tag) == name:
            return "".join(child.itertext())
    return ""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _http_get(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "application/atom+xml, application/rss+xml, application/feed+json, application/xml, text/xml",
            "User-Agent": _USER_AGENT,
        },
        method="GET",
    )
    with urlopen(request, timeout=_positive_float(os.environ.get("DOMESTIC_RSS_TIMEOUT_SECONDS"), _DEFAULT_TIMEOUT_SECONDS)) as response:
        raw = response.read()
        encoding = response.headers.get_content_charset() or "utf-8"
        return raw.decode(encoding, errors="replace")


def _query_without_site_directives(query: str) -> str:
    return " ".join(_SITE_DIRECTIVE.sub(" ", query).split())


def _site_domains(query: str) -> tuple[str, ...]:
    domains: list[str] = []
    for raw_domain in re.findall(r"(?<!\S)site:([^\s]+)", query, flags=re.IGNORECASE):
        domain = _normalize_domain(raw_domain)
        if domain and domain not in domains:
            domains.append(domain)
    return tuple(domains)


def _relevance_score(text: str, query: str) -> int:
    lowered = text.lower()
    score = 0
    for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", query.lower()):
        if len(term) >= 2 and re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", lowered):
            score += 3
    ignored_cjk = set("的了和及与在是一个这那相关发展内容平台网站")
    for character in re.findall(r"[\u4e00-\u9fff]", query):
        if character not in ignored_cjk and character in text:
            score += 1
    return score


def _matches_domains(url: str, domains: tuple[str, ...]) -> bool:
    host = _normalize_domain(urlparse(url).hostname or "")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _normalized_http_url(value: str) -> str:
    url = value.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def _normalize_base_url(value: str) -> str:
    candidate = value.strip().rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _DEFAULT_RSSHUB_URL
    return candidate


def _normalize_domain(value: str) -> str:
    return value.lower().strip().strip(".").removeprefix("www.")


def _clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return " ".join(without_tags.split())


def _optional_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _positive_float(value: object, default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
