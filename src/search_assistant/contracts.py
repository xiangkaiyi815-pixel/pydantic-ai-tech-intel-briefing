from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


Classification = Literal["simple", "research", "hard", "high_stakes"]
Confidence = Literal["low", "medium", "high"]
ProviderEventStatus = Literal[
    "called",
    "success",
    "empty",
    "error",
    "timeout",
    "skipped",
    "fallback_used",
]
SourceCandidateStatus = Literal[
    "pending",
    "accepted",
    "duplicate",
    "feedback_seed",
    "rejected_invalid_url",
    "rejected_empty_content",
    "rejected_search_page_dump",
    "rejected_login_page",
    "rejected_generic_reference",
    "rejected_cad_medical",
    "rejected_cad_missing_anchor",
    "rejected_missing_industrial_anchor",
    "rejected_low_relevance",
]
MemoryLayer = Literal["event", "preference", "domain_knowledge", "run_experience"]


class IncomingMessage(BaseModel):
    message_id: str
    event_id: str | None = None
    user_id: str
    chat_id: str
    text: str
    source: str = "cli"
    raw_event: dict[str, Any] = Field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        return self.event_id or self.message_id


class VerifiedClaim(BaseModel):
    claim: str
    verdict: Literal["verified", "unverified", "contradicted"]
    source: str | None = None
    checked_at: str
    notes: str | None = None


class SourceEvidence(BaseModel):
    title: str
    url: str
    snippet: str = ""
    provider: str
    checked_at: str


class ProviderTraceEvent(BaseModel):
    """A compact, local-only record of one provider attempt.

    This does not store credentials or response bodies.  It exists so a user can
    tell whether a run really called a provider, skipped it, timed out, or fell
    back to another route.
    """

    provider: str
    query: str
    status: ProviderEventStatus
    result_count: int = 0
    reason: str | None = None
    error: str | None = None
    elapsed_ms: float | None = None
    tier: str | None = None
    budget_share: float | None = None
    checked_at: str


class SearchRecord(BaseModel):
    executed: bool
    queries: list[str] = Field(default_factory=list)
    sources: list[SourceEvidence] = Field(default_factory=list)
    engines: list[str] = Field(default_factory=list)
    skipped_reason: str | None = None


class SourceCandidate(BaseModel):
    """A normalized source candidate before and after briefing filters.

    Accepted candidates become ``CollectedSource`` rows.  Rejected candidates are
    kept as short audit records so overly strict filters can be diagnosed without
    leaking private data or forcing noisy items into the final report.
    """

    id: str
    topic_id: str
    user_id: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    platform: str = ""
    provider: str = ""
    query: str = ""
    status: SourceCandidateStatus = "pending"
    reason: str = ""
    relevance_score: float = 0.0
    importance_score: float = 0.0
    retrieved_at: str
    created_at: str


class LayeredMemoryItem(BaseModel):
    id: str
    layer: MemoryLayer
    kind: str
    content: str
    source_id: str
    confidence: Confidence = "medium"
    status: Literal["candidate", "active", "superseded", "rejected"] = "active"
    user_id: str = "system"
    chat_id: str = "system"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class TopicFeedbackSignal(BaseModel):
    id: str
    topic_id: str
    user_id: str
    chat_id: str
    signal_type: Literal["style", "evidence", "provider", "fact_correction", "case", "general"]
    scope: Literal["run", "source", "section", "topic", "provider"]
    body: str
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class CalibrationResult(BaseModel):
    ran: bool
    critique: str
    revision: str


class MemoryUpdate(BaseModel):
    kind: str
    content: str
    source_id: str | None = None


class AnswerPackage(BaseModel):
    question_id: str
    answer_text: str
    classification: Classification
    confidence: Confidence
    verified_claims: list[VerifiedClaim] = Field(default_factory=list)
    unverified_claims: list[str] = Field(default_factory=list)
    sources: list[SourceEvidence] = Field(default_factory=list)
    calibration: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    memory_updates: list[MemoryUpdate | dict[str, Any]] = Field(default_factory=list)
    search_record: SearchRecord | None = None
    trajectory_context: dict[str, Any] = Field(default_factory=dict, exclude=True)


class TopicSubscription(BaseModel):
    id: str
    user_id: str
    chat_id: str
    topic: str
    enabled: bool = True
    source_recipe: dict[str, float] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class CollectedSource(BaseModel):
    id: str
    topic_id: str
    user_id: str
    title: str
    url: str
    snippet: str = ""
    platform: str
    provider: str
    query: str
    relevance_score: float = 0.0
    importance_score: float = 0.0
    retrieved_at: str


class BriefingTheme(BaseModel):
    """A traceable evidence anchor for a briefing synthesis.

    The detailed report is intentionally free-form.  These fields retain the
    source-to-analysis relationship and keep legacy deterministic fallbacks
    compatible without forcing the model to use a fixed theme checklist.
    """

    name: str = Field(description="中文分析块名称，不是平台或文章标题。")
    analysis: str = Field(
        default="",
        description="该分析块的关键技术判断；模型输出时以具体实现、约束或工程取舍为中心。",
    )
    what_is_happening: str = Field(
        default="",
        description="说明业内正在构建、试点或部署的系统能力，概括多条来源而非罗列标题。"
    )
    core_technology: str = Field(
        default="",
        description="说明核心模型、算法、软件架构、系统接口、工程方法和关键约束。"
    )
    data_and_workflow: str = Field(
        default="",
        description="说明输入数据、处理/决策过程、结果写回的业务或工程系统，以及人工审核或接管点。"
    )
    why_it_matters: str = Field(default="", description="解释它影响的成本、效率、可靠性、能力边界或产业结构。")
    maturity: str = Field(default="", description="区分概念、试点、局部规模化或可复制能力，并说明剩余验证条件。")
    source_urls: list[str] = Field(default_factory=list, description="支撑该主题的原始来源 URL；有来源时必须来自输入来源。")


class BriefingSynthesis(BaseModel):
    search_content_summary: str = Field(description="说明本轮来源共同讨论的技术主线和领域变化。")
    short_summary: str = Field(description="用于速览，必须说明具体实现路径、关键技术和变化含义。")
    detailed_summary: str = Field(
        default="",
        description="详细总结正文，使用中文 Markdown 自由组织证据驱动的分析块；不得套用固定技术地图模板。",
    )
    themes: list[BriefingTheme] = Field(min_length=1, description="支撑详细总结的证据锚点，按实际证据组织，至少给出一个。")
    key_signal_interpretation: str = Field(description="解释跨来源重复出现的架构、数据闭环、指标或约束。")
    analysis_judgment: str = Field(description="可检查的跨来源归纳，不输出隐藏思维链。")
    next_search_directions: list[str] = Field(min_length=2, description="下一轮应验证的原始资料、指标或部署变化。")
    landing_suggestions: list[str] = Field(min_length=2, description="有边界、指标和人工接管点的实施建议。")


class DailyBriefing(BaseModel):
    id: str
    topic_id: str
    user_id: str
    chat_id: str
    topic: str
    run_date: date
    search_directions: list[str]
    keywords: list[str]
    sources: list[CollectedSource]
    source_candidates: list[SourceCandidate] = Field(default_factory=list)
    provider_events: list[ProviderTraceEvent] = Field(default_factory=list)
    synthesis: BriefingSynthesis
    markdown: str
    created_at: str


DomainKnowledgeEntityType = Literal[
    "concept",
    "system",
    "data",
    "workflow",
    "metric",
    "risk",
    "control",
    "source",
]


class DomainKnowledgeEntity(BaseModel):
    """A reviewed node in a domain-specific GraphRAG-style knowledge map."""

    id: str = Field(description="Stable graph-local entity identifier.")
    name: str
    entity_type: DomainKnowledgeEntityType = "concept"
    aliases: list[str] = Field(default_factory=list)
    summary: str
    evidence_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DomainKnowledgeRelation(BaseModel):
    """A typed edge between two reviewed domain entities."""

    id: str = Field(description="Stable graph-local relation identifier.")
    source_entity_id: str
    relation_type: str
    target_entity_id: str
    description: str
    evidence_refs: list[str] = Field(default_factory=list)
    weight: float = Field(default=1.0, ge=0.0)


class DomainKnowledgeGraph(BaseModel):
    """A small, auditable domain graph that keeps triples and prose together."""

    id: str
    name: str
    description: str
    overview: str
    source: str = "reviewed-seed"
    version: str = "1"
    entities: list[DomainKnowledgeEntity] = Field(min_length=1)
    relations: list[DomainKnowledgeRelation] = Field(default_factory=list)


class DomainKnowledgeSearchHit(BaseModel):
    graph_id: str
    graph_name: str
    entity_id: str
    entity_name: str
    entity_type: DomainKnowledgeEntityType
    score: float
    summary: str
    matched_aliases: list[str] = Field(default_factory=list)
    outgoing_relations: list[str] = Field(default_factory=list)
    incoming_relations: list[str] = Field(default_factory=list)


CandidateStatus = Literal["candidate", "validated", "deprecated"]


class DomainKnowledgeEvidence(BaseModel):
    title: str
    url: str
    provider: str
    retrieved_at: str


class DomainKnowledgeCandidate(BaseModel):
    id: str
    topic: str
    claim: str
    applies_when: str
    evidence: list[DomainKnowledgeEvidence] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    confidence: Confidence
    status: CandidateStatus = "candidate"
    source_ids: list[str] = Field(default_factory=list)
    fingerprint: str
    created_at: str
    updated_at: str
