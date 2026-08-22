from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal

from search_assistant.contracts import Classification


GroundingCategory = Literal["foundational", "evidence_required", "high_stakes"]


@dataclass(frozen=True)
class GroundingDecision:
    category: GroundingCategory
    allow_stable_model_knowledge: bool
    require_retrieval_sources: bool
    allow_precise_numbers: bool
    allow_vendor_version_claims: bool
    required_evidence_families: tuple[str, ...] = ()
    specific_entities: tuple[str, ...] = ()
    rationale: str = ""

    def to_context(self) -> dict[str, object]:
        return {
            "category": self.category,
            "allow_stable_model_knowledge": self.allow_stable_model_knowledge,
            "require_retrieval_sources": self.require_retrieval_sources,
            "allow_precise_numbers": self.allow_precise_numbers,
            "allow_vendor_version_claims": self.allow_vendor_version_claims,
            "required_evidence_families": list(self.required_evidence_families),
            "specific_entities": list(self.specific_entities),
            "rationale": self.rationale,
        }


def decide_grounding_policy(question: str, classification: Classification) -> GroundingDecision:
    specific_entities = tuple(_unique(_specific_entities(question)))
    if classification == "high_stakes":
        return GroundingDecision(
            category="high_stakes",
            allow_stable_model_knowledge=False,
            require_retrieval_sources=True,
            allow_precise_numbers=True,
            allow_vendor_version_claims=True,
            required_evidence_families=("authoritative_sources",),
            specific_entities=specific_entities,
            rationale="high-stakes question",
        )

    has_current_fact = _has_current_fact_signal(question)
    has_exact_value = _has_exact_value_signal(question)
    has_vendor_or_version = bool(specific_entities) or _has_vendor_or_version_signal(question)
    has_deployment_reasoning = _has_deployment_or_calculation_signal(question)
    foundational = _is_foundational_question(question)

    if foundational and not (has_current_fact or has_exact_value or has_vendor_or_version or has_deployment_reasoning):
        return GroundingDecision(
            category="foundational",
            allow_stable_model_knowledge=True,
            require_retrieval_sources=False,
            allow_precise_numbers=False,
            allow_vendor_version_claims=False,
            specific_entities=specific_entities,
            rationale="stable concept or understanding-check question",
        )

    evidence_families: list[str] = []
    if has_deployment_reasoning:
        evidence_families.extend(("deployment_feasibility", "model_parameters_or_benchmarks"))
    if has_current_fact:
        evidence_families.append("current_fact")
    if has_exact_value:
        evidence_families.append("exact_values")
    if has_vendor_or_version:
        evidence_families.append("vendor_or_version_claims")

    return GroundingDecision(
        category="evidence_required",
        allow_stable_model_knowledge=False,
        require_retrieval_sources=True,
        allow_precise_numbers=True,
        allow_vendor_version_claims=True,
        required_evidence_families=tuple(_unique(evidence_families or ["retrieval_sources"])),
        specific_entities=specific_entities,
        rationale="current, exact, vendor/version-specific, or deployment claim",
    )


def _is_foundational_question(question: str) -> bool:
    lowered = question.lower()
    markers = (
        "can i understand",
        "can be understood",
        "is it fair to say",
        "is it correct to think",
        "how should i understand",
        "what is",
        "explain",
        "concept",
        "principle",
        "mechanism",
        "是否可以理解为",
        "可以理解为",
        "能否理解为",
        "能不能理解为",
        "是不是可以理解",
        "如何理解",
        "怎么理解",
        "是什么",
        "什么是",
        "解释一下",
        "介绍一下",
        "基础",
        "概念",
        "原理",
        "机制",
    )
    return any(marker in lowered or marker in question for marker in markers)


def _has_current_fact_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered or marker in question
        for marker in (
            "latest",
            "current",
            "newest",
            "today",
            "as of",
            "recent",
            "release date",
            "schedule",
            "price",
            "ranking",
            "benchmark",
            "version",
            "2026",
            "最新",
            "当前",
            "今天",
            "截至",
            "目前",
            "现在",
            "发布",
            "发布日期",
            "版本",
            "排名",
            "价格",
            "基准",
            "实测",
        )
    )


def _has_exact_value_signal(question: str) -> bool:
    lowered = question.lower()
    if re.search(
        r"\b\d+(?:\.\d+)?\s*(?:gb|tb|mb|gb/s|tb/s|gt/s|tok/s|tokens/s|token/s|ms|%)\b",
        lowered,
    ):
        return True
    return any(
        marker in lowered or marker in question
        for marker in (
            "exact",
            "specific number",
            "specification",
            "specs",
            "parameter",
            "parameters",
            "how many",
            "capacity",
            "bandwidth",
            "latency",
            "throughput",
            "精确",
            "准确",
            "具体数字",
            "规格",
            "参数",
            "多少",
            "容量",
            "显存",
            "内存",
            "带宽",
            "延迟",
            "吞吐",
        )
    )


def _has_vendor_or_version_signal(question: str) -> bool:
    return bool(
        re.search(
            r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]+[-\s]?v\d+[A-Za-z0-9-]*(?![A-Za-z0-9])",
            question,
            flags=re.IGNORECASE,
        )
        or re.search(r"(?<![A-Za-z0-9])[A-Z]{2,}\s+[A-Z]?\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])", question)
        or re.search(r"(?<![A-Za-z0-9])[A-Za-z]+-\d+[A-Za-z0-9-]*(?![A-Za-z0-9])", question)
    )


def _has_deployment_or_calculation_signal(question: str) -> bool:
    lowered = question.lower()
    return any(
        marker in lowered or marker in question
        for marker in (
            "deploy",
            "deployment",
            "run",
            "fit",
            "calculate",
            "calculation",
            "estimate",
            "assumption",
            "assumptions",
            "parallel",
            "cluster",
            "feasibility",
            "can it",
            "can ",
            "could ",
            "部署",
            "运行",
            "装下",
            "计算",
            "估算",
            "假设",
            "并联",
            "集群",
            "可行",
            "能否",
        )
    )


def _specific_entities(question: str) -> list[str]:
    entities: list[str] = []
    patterns = (
        r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]+[-\s]?v\d+[A-Za-z0-9-]*(?![A-Za-z0-9])",
        r"(?<![A-Za-z0-9])[A-Z]{2,}\s+[A-Z]?\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])",
        r"(?<![A-Za-z0-9])[A-Z]{1,8}\d{2,}[A-Za-z0-9-]*(?![A-Za-z0-9])",
        r"(?<![A-Za-z0-9])[A-Za-z]+-\d+[A-Za-z0-9-]*(?![A-Za-z0-9])",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, question, flags=re.IGNORECASE):
            entities.append(re.sub(r"\s+", " ", match.group(0)).strip())
    return entities


def _unique(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique
