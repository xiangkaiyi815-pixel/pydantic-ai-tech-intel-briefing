from __future__ import annotations

import re

from search_assistant.contracts import (
    DomainKnowledgeEntity,
    DomainKnowledgeGraph,
    DomainKnowledgeRelation,
    DomainKnowledgeSearchHit,
)
from search_assistant.knowledge_graph.seeds import default_domain_graphs
from search_assistant.memory.store import MemoryStore


class DomainKnowledgeGraphService:
    """Manage reviewed domain graphs used as a lightweight GraphRAG index."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def seed_default_graphs(self, domain_ids: list[str] | None = None) -> dict[str, object]:
        requested = {domain_id.strip() for domain_id in domain_ids or [] if domain_id.strip()}
        graphs = default_domain_graphs(requested or None)
        if requested:
            found = {graph.id for graph in graphs}
            missing = sorted(requested - found)
            if missing:
                raise ValueError(f"unknown default domain graph: {', '.join(missing)}")
        results = [self.store.upsert_domain_knowledge_graph(graph) for graph in graphs]
        return {
            "seeded_graphs": len(results),
            "graphs": results,
        }

    def list_graphs(self) -> list[dict[str, object]]:
        return self.store.list_domain_knowledge_graphs()

    def query(self, query: str, domain_id: str | None = None, limit: int = 10) -> list[DomainKnowledgeSearchHit]:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            raise ValueError("query must not be empty")
        graphs = self._load_graphs(domain_id)
        hits: list[DomainKnowledgeSearchHit] = []
        for graph in graphs:
            entity_by_id = {entity.id: entity for entity in graph.entities}
            for entity in graph.entities:
                score, matched_aliases = self._score_entity(normalized, entity)
                relation_bonus = self._relation_bonus(normalized, entity, graph.relations, entity_by_id)
                score += relation_bonus
                if score <= 0:
                    continue
                hits.append(
                    DomainKnowledgeSearchHit(
                        graph_id=graph.id,
                        graph_name=graph.name,
                        entity_id=entity.id,
                        entity_name=entity.name,
                        entity_type=entity.entity_type,
                        score=round(score, 3),
                        summary=entity.summary,
                        matched_aliases=matched_aliases,
                        outgoing_relations=self._format_relations(
                            [relation for relation in graph.relations if relation.source_entity_id == entity.id],
                            entity_by_id,
                        ),
                        incoming_relations=self._format_relations(
                            [relation for relation in graph.relations if relation.target_entity_id == entity.id],
                            entity_by_id,
                        ),
                    )
                )
        hits.sort(key=lambda hit: (-hit.score, hit.graph_id, hit.entity_name))
        return hits[: max(1, limit)]

    def export_markdown(self, domain_id: str) -> str:
        graph = self.store.get_domain_knowledge_graph(domain_id)
        if graph is None:
            raise KeyError(f"domain knowledge graph not found: {domain_id}")
        entity_by_id = {entity.id: entity for entity in graph.entities}
        lines = [
            f"# {graph.name} 知识图谱",
            "",
            f"- 图谱 ID：`{graph.id}`",
            f"- 来源：`{graph.source}`",
            f"- 版本：`{graph.version}`",
            "",
            "## 概览",
            "",
            graph.overview,
            "",
            "## 实体",
            "",
        ]
        for entity in graph.entities:
            aliases = "、".join(entity.aliases) if entity.aliases else "无"
            lines.extend(
                [
                    f"### {entity.name}",
                    "",
                    f"- ID：`{entity.id}`",
                    f"- 类型：`{entity.entity_type}`",
                    f"- 别名：{aliases}",
                    f"- 证据引用：{', '.join(f'`{ref}`' for ref in entity.evidence_refs) or '无'}",
                    "",
                    entity.summary,
                    "",
                ]
            )
        lines.extend(["## 关系", ""])
        for relation in graph.relations:
            source = entity_by_id.get(relation.source_entity_id)
            target = entity_by_id.get(relation.target_entity_id)
            source_name = source.name if source else relation.source_entity_id
            target_name = target.name if target else relation.target_entity_id
            lines.extend(
                [
                    f"- `{relation.id}`：[{source_name}](#{_anchor(source_name)}) "
                    f"-- **{relation.relation_type}** --> "
                    f"[{target_name}](#{_anchor(target_name)})",
                    f"  - {relation.description}",
                ]
            )
        lines.extend(
            [
                "",
                "## 使用说明",
                "",
                "该图谱是经过审阅的领域语义骨架，适合作为日报检索、证据归类和后续增量知识沉淀的起点。",
                "新增知识时应保留原始自然语言证据，同时补充或更新实体-关系边，避免把孤立文本块直接平铺进知识库。",
                "",
            ]
        )
        return "\n".join(lines)

    def _load_graphs(self, domain_id: str | None) -> list[DomainKnowledgeGraph]:
        if domain_id is not None:
            graph = self.store.get_domain_knowledge_graph(domain_id)
            if graph is None:
                return []
            return [graph]
        return [
            graph
            for row in self.store.list_domain_knowledge_graphs()
            if (graph := self.store.get_domain_knowledge_graph(str(row["id"]))) is not None
        ]

    @staticmethod
    def _score_entity(query: str, entity: DomainKnowledgeEntity) -> tuple[float, list[str]]:
        query_lower = query.lower()
        terms = _terms(query_lower)
        name_lower = entity.name.lower()
        alias_lowers = [alias.lower() for alias in entity.aliases]
        summary_lower = entity.summary.lower()
        metadata_lower = " ".join(str(value).lower() for value in entity.metadata.values())

        score = 0.0
        matched_aliases: list[str] = []
        if query_lower == name_lower:
            score += 10.0
        elif query_lower in name_lower or name_lower in query_lower:
            score += 7.0
        for alias, alias_lower in zip(entity.aliases, alias_lowers, strict=False):
            if query_lower == alias_lower:
                score += 8.0
                matched_aliases.append(alias)
            elif query_lower in alias_lower or alias_lower in query_lower:
                score += 5.0
                matched_aliases.append(alias)
        for term in terms:
            if term in name_lower:
                score += 3.0
            if any(term in alias for alias in alias_lowers):
                score += 2.0
            if term in summary_lower:
                score += 1.0
            if term in metadata_lower:
                score += 0.5
        return score, matched_aliases

    @staticmethod
    def _relation_bonus(
        query: str,
        entity: DomainKnowledgeEntity,
        relations: list[DomainKnowledgeRelation],
        entity_by_id: dict[str, DomainKnowledgeEntity],
    ) -> float:
        query_lower = query.lower()
        terms = _terms(query_lower)
        bonus = 0.0
        for relation in relations:
            if relation.source_entity_id != entity.id and relation.target_entity_id != entity.id:
                continue
            other_id = relation.target_entity_id if relation.source_entity_id == entity.id else relation.source_entity_id
            other = entity_by_id.get(other_id)
            haystack = " ".join(
                (
                    relation.relation_type,
                    relation.description,
                    other.name if other else other_id,
                    " ".join(other.aliases) if other else "",
                )
            ).lower()
            if query_lower in haystack:
                bonus += 2.0
            bonus += sum(0.4 for term in terms if term in haystack)
        return bonus

    @staticmethod
    def _format_relations(
        relations: list[DomainKnowledgeRelation],
        entity_by_id: dict[str, DomainKnowledgeEntity],
    ) -> list[str]:
        formatted: list[str] = []
        for relation in relations:
            source = entity_by_id.get(relation.source_entity_id)
            target = entity_by_id.get(relation.target_entity_id)
            source_name = source.name if source else relation.source_entity_id
            target_name = target.name if target else relation.target_entity_id
            formatted.append(f"{source_name} --{relation.relation_type}--> {target_name}: {relation.description}")
        return formatted


def _terms(value: str) -> list[str]:
    return [term for term in re.split(r"[\s,，。；;:：/|()（）]+", value) if term]


def _anchor(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff -]+", "", text)
    return re.sub(r"\s+", "-", text)
