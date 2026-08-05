from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from mcp.server.fastmcp import FastMCP


mcp = FastMCP("search-assistant-public-sources")
_USER_AGENT = "search-assistant-public-sources/0.1"
_TIMEOUT_SECONDS = 8.0


@mcp.tool()
def search_public_sources(query: str, limit: int = 10) -> list[dict[str, str]]:
    """Search public technical communities without login and return source URLs with excerpts."""
    normalized_query = " ".join(query.split())
    if not normalized_query:
        return []

    bounded_limit = max(1, min(limit, 20))
    per_source = max(1, math.ceil(bounded_limit / 4))
    source_query = _source_query(normalized_query)
    fetchers = (_search_hacker_news, _search_arxiv, _search_github, _search_stack_exchange)
    with ThreadPoolExecutor(max_workers=len(fetchers)) as executor:
        results = list(executor.map(lambda fetcher: fetcher(source_query, per_source), fetchers))

    merged: list[dict[str, str]] = []
    seen: set[str] = set()
    for source_results in results:
        for result in source_results:
            url = result.get("url", "")
            if not url or url in seen or not _matches_query(result, source_query):
                continue
            seen.add(url)
            merged.append(result)
            if len(merged) >= bounded_limit:
                return merged
    return merged


def _search_hacker_news(query: str, limit: int) -> list[dict[str, str]]:
    payload = _get_json(
        "https://hn.algolia.com/api/v1/search?"
        + urlencode({"query": query, "tags": "story", "hitsPerPage": str(limit)})
    )
    results: list[dict[str, str]] = []
    for item in payload.get("hits", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("objectID") or "").strip()
        url = str(item.get("url") or f"https://news.ycombinator.com/item?id={item_id}").strip()
        title = str(item.get("title") or item.get("story_title") or "").strip()
        snippet = str(item.get("story_text") or item.get("comment_text") or "").strip()
        if title and url:
            results.append(_result(title, url, snippet, "hackernews-api"))
    return results


def _search_arxiv(query: str, limit: int) -> list[dict[str, str]]:
    url = "https://export.arxiv.org/api/query?" + urlencode(
        {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(limit),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    try:
        root = ElementTree.fromstring(_get_text(url))
    except ElementTree.ParseError:
        return []

    atom = "{http://www.w3.org/2005/Atom}"
    results: list[dict[str, str]] = []
    for entry in root.findall(f"{atom}entry"):
        title = _clean(entry.findtext(f"{atom}title") or "")
        source_url = _clean(entry.findtext(f"{atom}id") or "")
        summary = _clean(entry.findtext(f"{atom}summary") or "")
        if title and source_url:
            results.append(_result(title, source_url, summary, "arxiv-api"))
    return results


def _search_github(query: str, limit: int) -> list[dict[str, str]]:
    payload = _get_json(
        "https://api.github.com/search/repositories?"
        + urlencode({"q": query, "sort": "updated", "order": "desc", "per_page": str(limit)})
    )
    results: list[dict[str, str]] = []
    for item in payload.get("items", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("full_name") or "").strip()
        url = str(item.get("html_url") or "").strip()
        description = str(item.get("description") or "").strip()
        updated = str(item.get("updated_at") or "").strip()
        snippet = _clean(f"{description} Updated: {updated}")
        if title and url:
            results.append(_result(title, url, snippet, "github-api"))
    return results


def _search_stack_exchange(query: str, limit: int) -> list[dict[str, str]]:
    payload = _get_json(
        "https://api.stackexchange.com/2.3/search/advanced?"
        + urlencode(
            {
                "site": "stackoverflow",
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "pagesize": str(limit),
            }
        )
    )
    results: list[dict[str, str]] = []
    for item in payload.get("items", []) if isinstance(payload, dict) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("link") or "").strip()
        tags = ", ".join(str(tag) for tag in item.get("tags", []) if str(tag).strip())
        if title and url:
            results.append(_result(title, url, tags, "stackexchange-api"))
    return results


def _get_json(url: str) -> dict[str, Any]:
    try:
        payload = json.loads(_get_text(url))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _get_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "Accept": "application/json, application/atom+xml;q=0.9, text/xml;q=0.8",
            "User-Agent": _USER_AGENT,
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            raw = response.read()
            encoding = response.headers.get_content_charset() or "utf-8"
            return raw.decode(encoding, errors="replace")
    except OSError:
        return ""


def _result(title: str, url: str, snippet: str, provider: str) -> dict[str, str]:
    return {
        "title": _clean(title),
        "url": url,
        "snippet": _clean(snippet),
        "provider": provider,
        "checked_at": datetime.now(UTC).isoformat(),
    }


def _clean(value: str) -> str:
    return " ".join(value.split())


def _matches_query(result: dict[str, str], query: str) -> bool:
    terms = [term for term in re.findall(r"[a-z0-9]{3,}", query.lower()) if term not in {"and", "the", "for", "with"}]
    if not terms:
        return True
    haystack = f"{result.get('title', '')} {result.get('snippet', '')} {result.get('url', '')}".lower()
    if any(term in {"industrial", "manufacturing"} for term in terms):
        industrial_anchors = ("industrial", "manufactur", "factory", "maintenance", "quality inspection", "defect detection")
        acronym_anchors = ("mes", "erp", "scada", "plc")
        if not any(anchor in haystack for anchor in industrial_anchors) and not any(
            re.search(rf"\b{re.escape(anchor)}\b", haystack) for anchor in acronym_anchors
        ):
            return False
    matched_terms = sum(term in haystack for term in terms)
    return matched_terms >= min(2, len(terms))


def _source_query(query: str) -> str:
    english_terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9._-]*", query)
    lowered = query.lower()
    aliases: list[str] = []
    alias_rules = (
        (("工业", "制造", "工厂", "产线", "mes", "erp", "scada", "plc"), ("industrial", "manufacturing")),
        (("人工智能", "智能", "ai", "大模型", "llm"), ("AI",)),
        (("智能体", "agent"), ("agent",)),
        (("数字孪生", "仿真", "digital twin", "simulation"), ("digital twin", "simulation")),
        (("机器视觉", "视觉", "质检", "缺陷", "aoi"), ("machine vision", "quality inspection")),
        (("预测性维护", "设备维护", "oee", "停机"), ("predictive maintenance",)),
    )
    for markers, additions in alias_rules:
        if any(marker in query or marker in lowered for marker in markers):
            for addition in additions:
                if addition.lower() not in {term.lower() for term in english_terms + aliases}:
                    aliases.append(addition)
    return " ".join(english_terms + aliases) or query
