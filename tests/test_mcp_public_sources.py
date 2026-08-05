from search_assistant.mcp import public_sources


def test_public_source_mcp_merges_open_sources_and_preserves_provider(monkeypatch):
    monkeypatch.setattr(
        public_sources,
        "_search_hacker_news",
        lambda query, limit: [
            {"title": "HN", "url": "https://news.ycombinator.com/item?id=1", "snippet": "HN", "provider": "hackernews-api"}
        ],
    )
    monkeypatch.setattr(
        public_sources,
        "_search_arxiv",
        lambda query, limit: [
            {"title": "Paper", "url": "https://arxiv.org/abs/1", "snippet": "Paper", "provider": "arxiv-api"}
        ],
    )
    monkeypatch.setattr(
        public_sources,
        "_search_github",
        lambda query, limit: [
            {"title": "Repository", "url": "https://github.com/example/repo", "snippet": "Repo", "provider": "github-api"}
        ],
    )
    monkeypatch.setattr(
        public_sources,
        "_search_stack_exchange",
        lambda query, limit: [
            {"title": "Question", "url": "https://stackoverflow.com/q/1", "snippet": "Q", "provider": "stackexchange-api"}
        ],
    )

    results = public_sources.search_public_sources("AI", limit=3)

    assert [result["url"] for result in results] == [
        "https://news.ycombinator.com/item?id=1",
        "https://arxiv.org/abs/1",
        "https://github.com/example/repo",
    ]
    assert [result["provider"] for result in results] == ["hackernews-api", "arxiv-api", "github-api"]


def test_public_source_mcp_filters_results_that_do_not_match_an_english_query(monkeypatch):
    monkeypatch.setattr(
        public_sources,
        "_search_hacker_news",
        lambda query, limit: [
            {"title": "Industrial AI", "url": "https://news.ycombinator.com/item?id=1", "snippet": "Manufacturing", "provider": "hackernews-api"},
            {"title": "Unrelated", "url": "https://news.ycombinator.com/item?id=2", "snippet": "Web layout", "provider": "hackernews-api"},
        ],
    )
    monkeypatch.setattr(public_sources, "_search_arxiv", lambda query, limit: [])
    monkeypatch.setattr(public_sources, "_search_github", lambda query, limit: [])
    monkeypatch.setattr(public_sources, "_search_stack_exchange", lambda query, limit: [])

    results = public_sources.search_public_sources("industrial AI manufacturing", limit=3)

    assert [result["url"] for result in results] == ["https://news.ycombinator.com/item?id=1"]


def test_public_source_mcp_requires_an_industrial_anchor_for_industrial_queries(monkeypatch):
    monkeypatch.setattr(
        public_sources,
        "_search_hacker_news",
        lambda query, limit: [
            {"title": "AI orchestration market", "url": "https://news.ycombinator.com/item?id=1", "snippet": "AI agent platforms", "provider": "hackernews-api"},
            {"title": "Industrial manufacturing maintenance", "url": "https://news.ycombinator.com/item?id=2", "snippet": "Factory maintenance platform", "provider": "hackernews-api"},
        ],
    )
    monkeypatch.setattr(public_sources, "_search_arxiv", lambda query, limit: [])
    monkeypatch.setattr(public_sources, "_search_github", lambda query, limit: [])
    monkeypatch.setattr(public_sources, "_search_stack_exchange", lambda query, limit: [])

    results = public_sources.search_public_sources("industrial AI manufacturing", limit=3)

    assert [result["url"] for result in results] == ["https://news.ycombinator.com/item?id=2"]


def test_public_source_mcp_expands_chinese_industrial_ai_topic_before_search(monkeypatch):
    received_queries = []

    def record(query, limit):
        received_queries.append(query)
        return []

    monkeypatch.setattr(public_sources, "_search_hacker_news", record)
    monkeypatch.setattr(public_sources, "_search_arxiv", record)
    monkeypatch.setattr(public_sources, "_search_github", record)
    monkeypatch.setattr(public_sources, "_search_stack_exchange", record)

    public_sources.search_public_sources("AI+工业界发展", limit=4)

    assert received_queries == ["AI industrial manufacturing"] * 4
