from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class SourceCapabilityContract:
    """A local contract for one public source family.

    The project intentionally does not automate login-gated or write actions.
    This registry records what each source family is allowed to do so provider
    fallback and platform coverage can be audited without reading provider code.
    """

    slug: str
    display_name: str
    aliases: tuple[str, ...]
    url_hosts: tuple[str, ...]
    provider_families: tuple[str, ...]
    public_access: bool = True
    requires_login: bool = False
    allowed_actions: tuple[str, ...] = ("search", "metadata")
    unsupported_actions: tuple[str, ...] = ("login", "private_feed", "write")
    default_budget_share: float = 1.0
    risk_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SOURCE_CAPABILITY_CONTRACTS: tuple[SourceCapabilityContract, ...] = (
    SourceCapabilityContract(
        slug="general-web",
        display_name="General public web",
        aliases=("web", "browser", "bing", "baidu", "google", "duckduckgo", "searxng", "brave"),
        url_hosts=(),
        provider_families=("browser-bing", "browser-baidu", "browser-google", "duckduckgo", "searxng", "brave"),
        default_budget_share=3.0,
        risk_notes="Public result pages may rate-limit or return challenge pages; record provider status instead of bypassing.",
    ),
    SourceCapabilityContract(
        slug="mcp-public",
        display_name="Read-only MCP public sources",
        aliases=("mcp", "github", "arxiv", "hackernews", "stackexchange"),
        url_hosts=("github.com", "arxiv.org", "news.ycombinator.com", "stackoverflow.com", "stackexchange.com"),
        provider_families=("mcp:",),
        default_budget_share=2.0,
        risk_notes="Only read-only MCP tools are supported.",
    ),
    SourceCapabilityContract(
        slug="wechat-public-index",
        display_name="WeChat public index",
        aliases=("wechat", "weixin", "mp.weixin.qq.com"),
        url_hosts=("mp.weixin.qq.com", "weixin.qq.com"),
        provider_families=("browser-baidu", "mcp:domestic-rss"),
        allowed_actions=("public_index_search", "metadata"),
        default_budget_share=1.0,
        risk_notes="Use public-index snippets or original public URLs only; do not fetch login-gated account data.",
    ),
    SourceCapabilityContract(
        slug="bilibili",
        display_name="Bilibili public search",
        aliases=("bilibili", "bili", "b站"),
        url_hosts=("bilibili.com",),
        provider_families=("bilibili-public-api", "browser-baidu", "browser-bing", "browser-google"),
        default_budget_share=1.0,
    ),
    SourceCapabilityContract(
        slug="zhihu",
        display_name="Zhihu public index",
        aliases=("zhihu", "知乎"),
        url_hosts=("zhihu.com",),
        provider_families=("browser-baidu", "mcp:domestic-rss"),
        allowed_actions=("public_index_search", "metadata"),
        default_budget_share=1.0,
    ),
    SourceCapabilityContract(
        slug="toutiao-public-index",
        display_name="Toutiao public index",
        aliases=("toutiao", "今日头条"),
        url_hosts=("toutiao.com",),
        provider_families=("browser-baidu", "mcp:domestic-rss"),
        allowed_actions=("public_index_search", "metadata"),
        default_budget_share=1.0,
    ),
    SourceCapabilityContract(
        slug="xiaohongshu-public-index",
        display_name="Xiaohongshu public index",
        aliases=("xiaohongshu", "xhs", "小红书", "xhslink"),
        url_hosts=("xiaohongshu.com", "xhslink.com"),
        provider_families=("browser-baidu",),
        allowed_actions=("public_index_search", "metadata"),
        default_budget_share=1.0,
        risk_notes="Do not use logged-in feeds, comments, or private account automation.",
    ),
    SourceCapabilityContract(
        slug="youtube",
        display_name="YouTube public result pages",
        aliases=("youtube", "yt"),
        url_hosts=("youtube.com", "youtu.be"),
        provider_families=("browser-bing", "browser-google", "mcp:public"),
        default_budget_share=1.0,
    ),
    SourceCapabilityContract(
        slug="reddit",
        display_name="Reddit public result pages",
        aliases=("reddit",),
        url_hosts=("reddit.com",),
        provider_families=("browser-bing", "browser-google", "mcp:public"),
        default_budget_share=1.0,
    ),
    SourceCapabilityContract(
        slug="x-public-index",
        display_name="X public result pages",
        aliases=("x", "twitter"),
        url_hosts=("x.com", "twitter.com"),
        provider_families=("browser-bing", "browser-google"),
        allowed_actions=("public_index_search", "metadata"),
        default_budget_share=1.0,
    ),
    SourceCapabilityContract(
        slug="linkedin-public-index",
        display_name="LinkedIn public result pages",
        aliases=("linkedin",),
        url_hosts=("linkedin.com",),
        provider_families=("browser-bing", "browser-google"),
        allowed_actions=("public_index_search", "metadata"),
        default_budget_share=1.0,
        risk_notes="Do not automate logged-in LinkedIn browsing or profile scraping.",
    ),
)


def list_source_contracts() -> list[SourceCapabilityContract]:
    return list(DEFAULT_SOURCE_CAPABILITY_CONTRACTS)


def source_contracts_as_dicts() -> list[dict[str, Any]]:
    return [contract.to_dict() for contract in DEFAULT_SOURCE_CAPABILITY_CONTRACTS]


def source_contract_by_slug(slug_or_alias: str) -> SourceCapabilityContract | None:
    normalized = _normalize_slug(slug_or_alias)
    for contract in DEFAULT_SOURCE_CAPABILITY_CONTRACTS:
        if normalized == contract.slug or normalized in {_normalize_slug(alias) for alias in contract.aliases}:
            return contract
    return None


def source_contract_for_url(url: str) -> SourceCapabilityContract | None:
    host = _normalize_host(urlparse(url).hostname or "")
    if not host:
        return None
    for contract in DEFAULT_SOURCE_CAPABILITY_CONTRACTS:
        if any(host == expected or host.endswith(f".{expected}") for expected in contract.url_hosts):
            return contract
    return source_contract_by_slug("general-web")


def source_slug_for_provider(provider: str) -> str:
    lowered = provider.lower()
    for contract in DEFAULT_SOURCE_CAPABILITY_CONTRACTS:
        if any(lowered.startswith(family.lower()) for family in contract.provider_families if family):
            return contract.slug
    return "general-web"


def default_source_recipe() -> dict[str, float]:
    return {contract.slug: contract.default_budget_share for contract in DEFAULT_SOURCE_CAPABILITY_CONTRACTS}


def normalize_source_recipe(raw_recipe: dict[str, float] | None) -> dict[str, float]:
    if not raw_recipe:
        return default_source_recipe()
    normalized: dict[str, float] = {}
    for key, value in raw_recipe.items():
        contract = source_contract_by_slug(str(key))
        if contract is None:
            continue
        try:
            share = float(value)
        except (TypeError, ValueError):
            continue
        if share <= 0:
            continue
        normalized[contract.slug] = share
    return normalized or default_source_recipe()


def source_recipe_summary(recipe: dict[str, float] | None = None) -> dict[str, Any]:
    normalized = normalize_source_recipe(recipe)
    total = sum(normalized.values()) or 1.0
    return {
        "total_weight": total,
        "sources": [
            {
                "slug": contract.slug,
                "display_name": contract.display_name,
                "weight": normalized.get(contract.slug, 0.0),
                "share": round(normalized.get(contract.slug, 0.0) / total, 4),
                "public_access": contract.public_access,
                "requires_login": contract.requires_login,
                "allowed_actions": list(contract.allowed_actions),
                "unsupported_actions": list(contract.unsupported_actions),
            }
            for contract in DEFAULT_SOURCE_CAPABILITY_CONTRACTS
        ],
    }


def _normalize_slug(value: str) -> str:
    return "-".join(value.lower().strip().replace("_", "-").split())


def _normalize_host(value: str) -> str:
    return value.lower().strip().strip(".").removeprefix("www.")
