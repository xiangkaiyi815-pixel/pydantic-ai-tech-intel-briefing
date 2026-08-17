from __future__ import annotations

import os
import tempfile
from datetime import date, UTC, datetime

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    CollectedSource,
    DailyBriefing,
)
from search_assistant.knowledge_graph.extractor import (
    extract_entities,
    extract_relations,
    extend_graph_from_briefing,
)
from search_assistant.memory.store import MemoryStore


def _source(url: str, title: str, snippet: str = "") -> CollectedSource:
    return CollectedSource(
        id=f"src-{url}",
        topic_id="t-1",
        user_id="u-1",
        title=title,
        url=url,
        snippet=snippet,
        platform="test",
        provider="test",
        query="memory architectures",
        retrieved_at=datetime.now(UTC).isoformat(),
    )


def _briefing(topic: str = "AI agent memory architectures") -> DailyBriefing:
    now = datetime.now(UTC).isoformat()
    return DailyBriefing(
        id="brief-1",
        topic_id="t-1",
        user_id="u-1",
        chat_id="c-1",
        topic=topic,
        run_date=date(2026, 8, 17),
        search_directions=["memory architectures", "persistent memory systems"],
        keywords=["persistent memory", "vector store"],
        sources=[
            _source(
                "https://example.com/1",
                "Persistent memory systems for AI agents",
                "vector store and retrieval architectures",
            ),
            _source(
                "https://example.com/2",
                "Memory architectures in agent frameworks",
                "persistent memory and tool use",
            ),
        ],
        synthesis=BriefingSynthesis(
            search_content_summary=(
                "AI agents increasingly rely on persistent memory and vector store "
                "retrieval to maintain long-horizon context."
            ),
            short_summary="Persistent memory and vector store are core to agent architectures.",
            detailed_summary="",
            themes=[
                BriefingTheme(
                    name="记忆架构",
                    analysis=(
                        "persistent memory 与 vector store 的组合是 agent 记忆架构的关键路径，"
                        "tool use 需要跨会话上下文。"
                    ),
                    what_is_happening="行业内正在用 vector store 构建持久化记忆层。",
                    source_urls=["https://example.com/1", "https://example.com/2"],
                )
            ],
            key_signal_interpretation="memory + retrieval 成为标配",
            analysis_judgment="persistent memory 将决定 agent 的长期能力。",
            next_search_directions=["retrieval evaluation", "memory compression"],
            landing_suggestions=["评估 vector store 延迟", "建立记忆分层策略"],
        ),
        markdown="",
        created_at=now,
    )


def _store() -> MemoryStore:
    db_path = os.path.join(tempfile.mkdtemp(), "extractor-test.sqlite3")
    store = MemoryStore(db_path)
    store.initialize()
    return store


def test_extract_entities_finds_recurring_terms():
    entities = extract_entities(_briefing())
    names = {entity.name for entity in entities}
    assert "persistent memory" in names
    assert "vector store" in names
    assert all(entity.metadata.get("auto_extracted") for entity in entities)
    assert all(entity.evidence_refs == ["briefing:brief-1"] for entity in entities)


def test_extract_entities_filters_stop_words():
    entities = extract_entities(_briefing())
    names = {entity.name for entity in entities}
    for stop in ("the", "and", "系统", "技术"):
        assert stop not in names


def test_extract_relations_uses_co_occurrence():
    entities = extract_entities(_briefing())
    relations = extract_relations(entities, _briefing())
    by_name = {entity.name: entity for entity in entities}
    pairs = {
        (relation.source_entity_id, relation.target_entity_id) for relation in relations
    }
    if "persistent memory" in by_name and "vector store" in by_name:
        pair = (by_name["persistent memory"].id, by_name["vector store"].id)
        assert pair in pairs or (pair[1], pair[0]) in pairs


def test_extend_graph_creates_auto_graph_and_merges():
    store = _store()
    briefing = _briefing()
    result = extend_graph_from_briefing(store, briefing)
    assert result["graph_id"].startswith("auto-")
    assert result["added_entities"] > 0
    graphs = store.list_domain_knowledge_graphs()
    auto_graphs = [g for g in graphs if str(g["id"]).startswith("auto-")]
    assert len(auto_graphs) == 1

    # A second run on the SAME topic merges into the same graph without
    # duplicating entities.
    second_briefing = _briefing("AI agent memory architectures")
    second_briefing.id = "brief-2"
    result2 = extend_graph_from_briefing(store, second_briefing)
    assert result2["graph_id"] == result["graph_id"]
    graph = store.get_domain_knowledge_graph(result["graph_id"])
    assert graph is not None
    # "vector store" already exists from the first run -> not duplicated.
    names = [entity.name for entity in graph.entities]
    assert names.count("vector store") <= 1


def test_extend_graph_isolates_auto_graphs_by_topic():
    """Different topics must land in different auto graphs (no cross-topic merge)."""
    store = _store()
    first = extend_graph_from_briefing(store, _briefing("AI agent memory architectures"))
    second = extend_graph_from_briefing(store, _briefing("vector store latency"))

    assert first["graph_id"] != second["graph_id"]
    auto_graphs = [g for g in store.list_domain_knowledge_graphs() if str(g["id"]).startswith("auto-")]
    assert len(auto_graphs) == 2
    first_graph = store.get_domain_knowledge_graph(first["graph_id"])
    second_graph = store.get_domain_knowledge_graph(second["graph_id"])
    assert first_graph is not None and second_graph is not None
    # Every entity stays in the graph of the topic that produced it: the entity
    # summary records the source briefing topic.
    assert all("AI agent memory architectures" in (entity.summary or "") for entity in first_graph.entities)
    assert all("vector store latency" in (entity.summary or "") for entity in second_graph.entities)


def test_extend_graph_with_empty_briefing_returns_noop():
    store = _store()
    briefing = _briefing()
    briefing.sources = []
    briefing.synthesis.themes = []
    briefing.search_directions = []
    briefing.keywords = []
    # Synthesis fields are the only remaining text source; when they are blank
    # too, there is nothing to extract and the call must be a no-op.
    briefing.synthesis.search_content_summary = ""
    briefing.synthesis.short_summary = ""
    briefing.synthesis.detailed_summary = ""
    briefing.synthesis.key_signal_interpretation = ""
    briefing.synthesis.analysis_judgment = ""
    result = extend_graph_from_briefing(store, briefing)
    assert result["added_entities"] == 0
