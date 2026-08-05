import json

from search_assistant.mcp.domestic_rss import DomesticRssFeedSearcher, _parse_feed


def _write_catalog(tmp_path, sources):
    path = tmp_path / "domestic-rss.sources.json"
    path.write_text(json.dumps({"version": 1, "sources": sources}), encoding="utf-8")
    return path


def test_domestic_rss_search_preserves_original_platform_urls_and_scores_query(tmp_path):
    path = _write_catalog(
        tmp_path,
        [
            {
                "id": "weibo-keyword",
                "platform": "weibo",
                "kind": "query",
                "route_template": "/weibo/keyword/{query}",
                "domains": ["weibo.com"],
                "enabled": True,
            }
        ],
    )
    requested_urls = []

    def transport(url):
        requested_urls.append(url)
        return """<?xml version='1.0'?><rss><channel>
        <item><title>Industrial AI factory update</title><link>https://m.weibo.com/status/1</link><description>MES and equipment data</description></item>
        <item><title>Unrelated</title><link>https://m.weibo.com/status/2</link><description>Entertainment</description></item>
        </channel></rss>"""

    searcher = DomesticRssFeedSearcher(path, "http://rsshub.test:1200", transport=transport)
    results = searcher.search("industrial AI", limit=5)

    assert requested_urls == ["http://rsshub.test:1200/weibo/keyword/industrial%20AI"]
    assert [(item["url"], item["provider"]) for item in results] == [
        ("https://m.weibo.com/status/1", "weibo:weibo-keyword")
    ]


def test_domestic_rss_honors_site_scope_and_never_crosses_platform_domains(tmp_path):
    path = _write_catalog(
        tmp_path,
        [
            {
                "id": "weibo-keyword",
                "platform": "weibo",
                "kind": "query",
                "route_template": "/weibo/keyword/{query}",
                "domains": ["weibo.com"],
                "enabled": True,
            },
            {
                "id": "zhihu-hot",
                "platform": "zhihu",
                "kind": "hot",
                "route_template": "/zhihu/hot",
                "domains": ["zhihu.com"],
                "enabled": True,
            },
        ],
    )
    requested_urls = []

    def transport(url):
        requested_urls.append(url)
        return """<rss><channel><item><title>Industrial AI</title><link>https://m.weibo.com/status/1</link><description>Factory AI</description></item></channel></rss>"""

    searcher = DomesticRssFeedSearcher(path, "http://rsshub.test", transport=transport)
    results = searcher.search("site:weibo.com industrial AI", limit=5)

    assert requested_urls == ["http://rsshub.test/weibo/keyword/industrial%20AI"]
    assert [item["url"] for item in results] == ["https://m.weibo.com/status/1"]


def test_domestic_rss_skips_login_gated_and_incomplete_subscriptions(tmp_path):
    path = _write_catalog(
        tmp_path,
        [
            {
                "id": "login-only",
                "platform": "weibo",
                "kind": "subscription",
                "route_template": "/weibo/user/{uid}",
                "domains": ["weibo.com"],
                "parameters": {"uid": "123"},
                "enabled": True,
                "requires_login": True,
            },
            {
                "id": "missing-id",
                "platform": "wechat",
                "kind": "subscription",
                "route_template": "/wechat/wechat2rss/{id}",
                "domains": ["mp.weixin.qq.com"],
                "enabled": True,
            },
        ],
    )

    searcher = DomesticRssFeedSearcher(path, "http://rsshub.test", transport=lambda url: (_ for _ in ()).throw(AssertionError(url)))

    assert searcher.search("industrial AI", limit=5) == []
    status = searcher.source_status()["sources"]
    assert [item["status"] for item in status] == ["requires_login", "missing_required_parameters"]


def test_domestic_rss_parses_atom_and_json_feed_formats():
    atom = """<feed xmlns='http://www.w3.org/2005/Atom'><entry><title>Industrial AI</title><link href='https://www.zhihu.com/question/1'/><summary>Factory systems</summary></entry></feed>"""
    json_feed = json.dumps(
        {
            "items": [
                {
                    "title": "Industrial AI video",
                    "external_url": "https://www.bilibili.com/video/BV1test",
                    "content_text": "MES workflow",
                }
            ]
        }
    )

    assert _parse_feed(atom) == [
        {"title": "Industrial AI", "url": "https://www.zhihu.com/question/1", "snippet": "Factory systems"}
    ]
    assert _parse_feed(json_feed) == [
        {"title": "Industrial AI video", "url": "https://www.bilibili.com/video/BV1test", "snippet": "MES workflow"}
    ]
