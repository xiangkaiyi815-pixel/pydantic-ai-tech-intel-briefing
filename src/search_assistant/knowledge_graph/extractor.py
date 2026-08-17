from __future__ import annotations

import re
import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from search_assistant.contracts import (
    DailyBriefing,
    DomainKnowledgeEntity,
    DomainKnowledgeEntityType,
    DomainKnowledgeGraph,
    DomainKnowledgeRelation,
)
from search_assistant.knowledge_graph.bm25 import tokenize
from search_assistant.memory.store import MemoryStore


# Terms that appear too often across topics to be meaningful graph entities.
_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "been",
    "into",
    "over",
    "than",
    "then",
    "they",
    "them",
    "their",
    "have",
    "has",
    "had",
    "will",
    "would",
    "could",
    "should",
    "about",
    "after",
    "before",
    "between",
    "through",
    "during",
    "using",
    "used",
    "use",
    "new",
    "also",
    "one",
    "two",
    "three",
    "based",
    "via",
    "yet",
    "its",
    "our",
    "your",
    "如何",
    "一个",
    "以及",
    "可以",
    "能够",
    "进行",
    "通过",
    "相关",
    "系统",
    "技术",
    "领域",
    "目前",
    "当前",
    "未来",
    "已经",
    "我们",
    "他们",
    "这些",
    "那些",
    "其中",
    "关于",
    "由于",
    "因为",
    "所以",
    "但是",
    "以及",
    "不是",
    "没有",
    "需要",
    "可能",
    "主要",
    "重要",
    "关键",
    "解决",
    "问题",
    "方案",
    "实现",
    "应用",
    "发展",
    "研究",
    "分析",
    "报告",
    "平台",
    "工具",
    "框架",
}

_MIN_TERM_LEN = 2
_MAX_TERM_LEN = 40
_MIN_OCCURRENCES = 2


def _entity_type(name: str) -> DomainKnowledgeEntityType:
    lowered = name.lower()
    if any(marker in lowered for marker in ("metric", "指标", "延迟", "吞吐", "成本", "精度", "召回")):
        return "metric"
    if any(marker in lowered for marker in ("risk", "风险", "威胁", "隐患", "安全")):
        return "risk"
    if any(marker in lowered for marker in ("system", "平台", "框架", "引擎", "库", "runtime", "sdk")):
        return "system"
    if any(marker in lowered for marker in ("data", "数据集", "语料", "benchmark", "基准")):
        return "data"
    if any(marker in lowered for marker in ("workflow", "pipeline", "流程", "管线", "链路")):
        return "workflow"
    return "concept"


def _meaningful_term(term: str) -> bool:
    if len(term) < _MIN_TERM_LEN or len(term) > _MAX_TERM_LEN:
        return False
    if re.fullmatch(r"\d+([.,]\d+)?%?", term):
        return False
    lowered = term.lower()
    if lowered in _STOP_WORDS:
        return False
    # A token must contain at least one letter or CJK character.
    if not re.search(r"[a-z]|[\u4e00-\u9fff]", lowered):
        return False
    return True


def _collect_texts(briefing: DailyBriefing) -> list[str]:
    """Collect the free-text fields that should contribute candidate terms."""
    texts: list[str] = []
    synthesis = briefing.synthesis
    texts.extend(
        [
            synthesis.search_content_summary,
            synthesis.short_summary,
            synthesis.detailed_summary,
            synthesis.key_signal_interpretation,
            synthesis.analysis_judgment,
        ]
    )
    texts.extend(briefing.search_directions)
    texts.extend(briefing.keywords)
    for theme in synthesis.themes:
        texts.extend([theme.name, theme.analysis, theme.what_is_happening])
    for source in briefing.sources:
        texts.extend([source.title, source.snippet])
    return [text for text in texts if text]


def extract_entities(
    briefing: DailyBriefing,
    min_occurrences: int = _MIN_OCCURRENCES,
) -> list[DomainKnowledgeEntity]:
    """Extract candidate entities from a briefing trajectory by term frequency.

    Both single tokens and adjacent English bigrams are counted, so meaningful
    phrases such as "persistent memory" or "vector store" surface as entities
    instead of being split into generic words.  Only terms that appear in at
    least ``min_occurrences`` different fields are kept, filtering out one-off
    vocabulary while retaining recurring domain concepts.
    """
    texts = _collect_texts(briefing)
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = [token for token in tokenize(text) if _meaningful_term(token)]
        for term in set(tokens):
            counter[term] += 1
        for left, right in zip(tokens, tokens[1:]):
            bigram = f"{left} {right}"
            if _meaningful_term(bigram):
                counter[bigram] += 1

    now = datetime.now(UTC).isoformat()
    entities: list[DomainKnowledgeEntity] = []
    for term, count in counter.most_common():
        if count < min_occurrences:
            break  # counter is sorted by frequency; once below threshold, stop.
        entities.append(
            DomainKnowledgeEntity(
                id=f"auto_entity_{uuid.uuid4().hex[:12]}",
                name=term,
                entity_type=_entity_type(term),
                aliases=[],
                summary=(
                    f"Auto-extracted from the '{briefing.topic}' briefing trajectory "
                    f"(appeared in {count} fields); needs human review before trusted use."
                ),
                evidence_refs=[f"briefing:{briefing.id}"],
                metadata={"auto_extracted": True, "occurrences": count, "source": "trajectory"},
            )
        )
    return entities[:50]


def extract_relations(
    entities: list[DomainKnowledgeEntity],
    briefing: DailyBriefing,
) -> list[DomainKnowledgeRelation]:
    """Create co-occurrence relations between entities found in the same source."""
    entity_by_name = {entity.name.lower(): entity for entity in entities}
    co_occurrences: Counter[tuple[str, str]] = Counter()

    # Co-occurrence within a single source (title + snippet) signals relatedness.
    for source in briefing.sources:
        window_text = f"{source.title} {source.snippet}".lower()
        present = [name for name in entity_by_name if name in window_text]
        for index, left in enumerate(present):
            for right in present[index + 1 :]:
                key = (left, right) if left < right else (right, left)
                co_occurrences[key] += 1

    now = datetime.now(UTC).isoformat()
    relations: list[DomainKnowledgeRelation] = []
    for (left_name, right_name), count in co_occurrences.most_common():
        left = entity_by_name[left_name]
        right = entity_by_name[right_name]
        relations.append(
            DomainKnowledgeRelation(
                id=f"auto_rel_{uuid.uuid4().hex[:12]}",
                source_entity_id=left.id,
                relation_type="co_occurs_with",
                target_entity_id=right.id,
                description=(
                    f"Co-occurred in {count} source(s) of the '{briefing.topic}' briefing; "
                    "relation is auto-extracted and needs human review."
                ),
                evidence_refs=[f"briefing:{briefing.id}"],
                weight=float(count),
            )
        )
    return relations[:50]


def extend_graph_from_briefing(
    store: MemoryStore,
    briefing: DailyBriefing,
    graph_id: str | None = None,
    min_occurrences: int = _MIN_OCCURRENCES,
) -> dict[str, Any]:
    """Merge trajectory-extracted entities into a knowledge graph.

    A dedicated auto graph (``auto-<graph_id>``) is used so reviewed seed graphs
    stay untouched; auto entities carry ``metadata.auto_extracted=True`` and are
    meant to be reviewed before being treated as trusted knowledge.
    """
    entities = extract_entities(briefing, min_occurrences=min_occurrences)
    if not entities:
        return {"graph_id": None, "added_entities": 0, "added_relations": 0, "total_entities": 0}

    existing_graphs = store.list_domain_knowledge_graphs()
    auto_graph: DomainKnowledgeGraph | None = None
    for row in existing_graphs:
        graph = store.get_domain_knowledge_graph(str(row["id"]))
        if graph is not None and graph.id.startswith("auto-"):
            auto_graph = graph
            break

    relations = extract_relations(entities, briefing)
    if auto_graph is None:
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", briefing.topic).strip("-")[:40] or "misc"
        auto_graph = DomainKnowledgeGraph(
            id=f"auto-{slug}",
            name=f"自动提取图谱（{briefing.topic}）",
            description="Auto-extracted from briefing trajectories; review before trusted use.",
            overview=f"Entities and co-occurrence relations extracted from the '{briefing.topic}' briefing trajectory.",
            source="auto-extracted",
            version="1",
            entities=entities,
            relations=relations,
        )
        store.upsert_domain_knowledge_graph(auto_graph)
        return {
            "graph_id": auto_graph.id,
            "added_entities": len(entities),
            "added_relations": len(relations),
            "total_entities": len(entities),
        }

    # Merge into the existing auto graph, deduplicating by name (case-insensitive).
    existing_names = {entity.name.lower() for entity in auto_graph.entities}
    added_entities = [entity for entity in entities if entity.name.lower() not in existing_names]
    if added_entities:
        auto_graph.entities.extend(added_entities)

    # Relations referencing the same pair are deduplicated by (source,target).
    existing_pairs = {
        (relation.source_entity_id, relation.target_entity_id)
        for relation in auto_graph.relations
    }
    id_by_name = {entity.name.lower(): entity.id for entity in auto_graph.entities}
    added_relations: list[DomainKnowledgeRelation] = []
    for relation in relations:
        source_name = next(
            (name for name, entity_id in id_by_name.items() if entity_id == relation.source_entity_id),
            "",
        )
        target_name = next(
            (name for name, entity_id in id_by_name.items() if entity_id == relation.target_entity_id),
            "",
        )
        source_id = id_by_name.get(source_name)
        target_id = id_by_name.get(target_name)
        if source_id and target_id and (source_id, target_id) not in existing_pairs:
            relation.source_entity_id = source_id
            relation.target_entity_id = target_id
            added_relations.append(relation)
            existing_pairs.add((source_id, target_id))

    auto_graph.relations.extend(added_relations)
    store.upsert_domain_knowledge_graph(auto_graph)
    return {
        "graph_id": auto_graph.id,
        "added_entities": len(added_entities),
        "added_relations": len(added_relations),
        "total_entities": len(auto_graph.entities),
    }
