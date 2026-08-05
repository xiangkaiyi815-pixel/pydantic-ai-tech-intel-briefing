from __future__ import annotations

import argparse
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from search_assistant.config import Settings
from search_assistant.contracts import AnswerPackage, IncomingMessage
from search_assistant.memory.store import MemoryStore
from search_assistant.profile.service import ProfileService
from search_assistant.search.provider import search_client_from_settings
from search_assistant.workflow.runtime import runtime_from_settings
from search_assistant.workflow.service import SearchAssistantWorkflow


@dataclass(frozen=True)
class EvaluationQuestion:
    difficulty: str
    text: str
    expected_terms: tuple[str, ...] = ()
    expected_source_terms: tuple[str, ...] = ()


ROUND_QUESTIONS: dict[int, list[EvaluationQuestion]] = {
    1: [
        EvaluationQuestion("easy", "CXL是什么？请用三句话解释。", ("CXL", "Compute Express Link")),
        EvaluationQuestion("easy", "NVIDIA GB10 / DGX Spark 的统一内存是多少？请给来源。", ("128", "GB")),
        EvaluationQuestion("medium", "DeepSeek-V3 的总参数和激活参数分别是多少？请优先查模型卡或官方仓库。", ("671B", "37B"), ("ModelScope", "Hugging Face", "GitHub")),
        EvaluationQuestion("hard", "10台GB10并联能否部署满血DeepSeek-V4？请说明内存、量化、CX7互联和不确定性。", ("GB10", "CX7", "DeepSeek"), ("NVIDIA", "ModelScope")),
        EvaluationQuestion("hard", "截至2026年，AI物理大模型/world model/embodied physical AI发展到哪里了？只基于可搜索来源回答。", ("world model", "embodied"), ("NVIDIA", "DeepMind", "arXiv")),
    ],
    2: [
        EvaluationQuestion("easy", "什么是KV Cache？它为什么会影响大模型推理显存？", ("KV", "Cache")),
        EvaluationQuestion("easy", "ConnectX-7/CX7网卡是什么？它和AI集群有什么关系？", ("ConnectX-7", "CX7")),
        EvaluationQuestion("medium", "请比较CXL和NVLink在AI服务器内存/互联中的角色，必须区分已验证事实和推断。", ("CXL", "NVLink")),
        EvaluationQuestion("hard", "如果只能用浏览器搜索，怎么核验DeepSeek模型参数、魔搭社区模型卡、Hugging Face和GitHub之间的一致性？", ("魔搭", "Hugging Face", "GitHub")),
        EvaluationQuestion("hard", "请评估10台DGX Spark跑DeepSeek-V3或V4类MoE模型的可行性，要求列公式、来源和无法确认项。", ("DGX Spark", "DeepSeek", "MoE")),
    ],
    3: [
        EvaluationQuestion("easy", "什么是MoE模型？为什么总参数和激活参数不同？", ("MoE", "激活")),
        EvaluationQuestion("medium", "DeepSeek部署问题里为什么必须查魔搭/ModelScope、Hugging Face、GitHub和NVIDIA官方页？", ("ModelScope", "Hugging Face", "GitHub", "NVIDIA")),
        EvaluationQuestion("medium", "如果搜索结果大多是百科、词典或无关网页，你应该如何回答？", ("搜索结果相关性不足", "证据")),
        EvaluationQuestion("hard", "请制定一份GB10小集群部署大模型的验证清单，覆盖内存、量化、KV Cache、CX7网络、并行策略、tok/s估算风险。", ("KV Cache", "CX7", "tok/s")),
        EvaluationQuestion("hard", "现在AI物理大模型有哪些可靠进展？如果搜索不到可靠来源，请明确拒绝推测并给出补搜策略。", ("搜索", "来源")),
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True, choices=sorted(ROUND_QUESTIONS))
    parser.add_argument("--data-root", default=".local-data/quality-rounds-20260629")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"round-{args.round}"
    data_root = Path(args.data_root)
    run_dir = data_root / run_id
    store = MemoryStore(run_dir / "assistant.sqlite3")
    store.initialize()

    settings = Settings.from_env()
    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=runtime_from_settings(settings),
        search_client=search_client_from_settings(settings),
        search_budget_seconds=settings.workflow_search_budget_seconds,
    )
    profile_service = ProfileService(store)

    items: list[dict[str, Any]] = []
    for index, question in enumerate(ROUND_QUESTIONS[args.round], start=1):
        package = workflow.answer(
            IncomingMessage(
                message_id=f"{run_id}-{index}-{uuid.uuid4().hex}",
                event_id=None,
                user_id="quality-round-evaluator",
                chat_id=f"quality-round-{args.round}",
                text=question.text,
                source="quality-round",
            )
        )
        profile_service.update_from_answer(package)
        items.append(_item_from_package(index, question, package))

    summary = _summary(items)
    result = {
        "round": args.round,
        "run_id": run_id,
        "items": items,
        "summary": summary,
        "experience_items": store.list_experience_items(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "quality-evaluation.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = run_dir / "quality-evaluation.md"
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), "markdown_path": str(markdown_path), "summary": summary}, ensure_ascii=False, indent=2))
    return 0


def _item_from_package(index: int, question: EvaluationQuestion, package: AnswerPackage) -> dict[str, Any]:
    answer_text = package.answer_text
    review = package.review or {}
    search_queries = _extract_search_queries(answer_text)
    search_results = _extract_search_results(answer_text)
    score, reasons = _score_answer(question, package, search_queries, search_results)
    return {
        "index": index,
        "difficulty": question.difficulty,
        "question": question.text,
        "question_id": package.question_id,
        "classification": package.classification,
        "confidence": package.confidence,
        "score": score,
        "score_reasons": reasons,
        "source_count": len(package.sources),
        "source_titles": [source.title for source in package.sources],
        "source_urls": [source.url for source in package.sources],
        "search_queries": search_queries,
        "search_results_visible": search_results,
        "verified_claims": len(package.verified_claims),
        "unverified_claims": len(package.unverified_claims),
        "unverified_claim_text": package.unverified_claims,
        "calibration": package.calibration,
        "review": review,
        "memory_updates": [update.model_dump(mode="json") if hasattr(update, "model_dump") else update for update in package.memory_updates],
        "answer_text": answer_text,
    }


def _score_answer(
    question: EvaluationQuestion,
    package: AnswerPackage,
    search_queries: list[str],
    search_results: list[str],
) -> tuple[float, list[str]]:
    score = 10.0
    reasons: list[str] = []
    answer = package.answer_text
    review = package.review or {}

    if not search_queries:
        score -= 2.0
        reasons.append("未显示搜索词")
    if package.sources:
        score += min(1.0, len(package.sources) * 0.15)
    else:
        score -= 2.0
        reasons.append("无搜索来源")
    if review.get("ran") is not True:
        score -= 1.0
        reasons.append("审查agent未运行")
    if review.get("approved") is False:
        score -= 2.0
        reasons.append("审查未通过或被门禁阻断")
    if "最终审查未通过" in answer or "搜索结果相关性不足" in answer:
        score -= 1.0
        reasons.append("最终回答为阻断/降级答复")
    if package.confidence == "low":
        score -= 0.5
        reasons.append("低置信度")
    if package.unverified_claims:
        score -= min(1.5, 0.5 * len(package.unverified_claims))
        reasons.append("存在未验证声明")
    if "搜索记录:" not in answer:
        score -= 1.5
        reasons.append("缺少搜索记录")
    if re.search(r"校准后|修订后的回答|根据搜索结果中|draft|草稿", answer, flags=re.IGNORECASE):
        score -= 0.8
        reasons.append("含内部校准/草稿口吻")

    missing_terms = [term for term in question.expected_terms if term.lower() not in answer.lower()]
    if missing_terms:
        score -= min(2.0, 0.5 * len(missing_terms))
        reasons.append(f"缺少期望答案关键词: {', '.join(missing_terms)}")

    combined_sources = " ".join(package.model_dump(mode="json").get("source_titles", []) if False else [])
    combined_visible_sources = " ".join(search_results + [source.title for source in package.sources] + [source.url for source in package.sources])
    missing_source_terms = [
        term for term in question.expected_source_terms if term.lower() not in combined_visible_sources.lower()
    ]
    if missing_source_terms:
        score -= min(2.0, 0.5 * len(missing_source_terms))
        reasons.append(f"缺少期望来源族: {', '.join(missing_source_terms)}")

    return max(0.0, min(10.0, round(score, 2))), reasons or ["基本满足自动评分规则"]


def _extract_search_queries(answer_text: str) -> list[str]:
    return _extract_numbered_block(answer_text, "搜索词:")


def _extract_search_results(answer_text: str) -> list[str]:
    return _extract_numbered_block(answer_text, "搜索结果:")


def _extract_numbered_block(answer_text: str, marker: str) -> list[str]:
    lines = answer_text.splitlines()
    try:
        start = lines.index(f"- {marker}") + 1
    except ValueError:
        return []
    collected: list[str] = []
    for line in lines[start:]:
        if line.startswith("- ") and not re.match(r"\s+\d+\.", line):
            break
        match = re.match(r"\s+\d+\.\s*(.*)", line)
        if match:
            collected.append(match.group(1).strip())
    return collected


def _summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    average = round(sum(item["score"] for item in items) / max(1, len(items)), 2)
    return {
        "question_count": len(items),
        "average_score": average,
        "approved_count": sum(1 for item in items if (item.get("review") or {}).get("approved") is True),
        "blocked_count": sum(1 for item in items if "阻断" in item["answer_text"] or "最终审查未通过" in item["answer_text"]),
        "low_confidence_count": sum(1 for item in items if item["confidence"] == "low"),
        "scores_by_difficulty": {
            difficulty: round(
                sum(item["score"] for item in items if item["difficulty"] == difficulty)
                / max(1, sum(1 for item in items if item["difficulty"] == difficulty)),
                2,
            )
            for difficulty in sorted({item["difficulty"] for item in items})
        },
    }


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        f"# Quality Evaluation Round {result['round']}",
        "",
        f"- Run id: `{result['run_id']}`",
        f"- Average score: `{result['summary']['average_score']}`",
        f"- Questions: `{result['summary']['question_count']}`",
        f"- Approved: `{result['summary']['approved_count']}`",
        f"- Blocked: `{result['summary']['blocked_count']}`",
        "",
    ]
    for item in result["items"]:
        lines.extend(
            [
                f"## Q{item['index']} [{item['difficulty']}] Score {item['score']}",
                "",
                item["question"],
                "",
                f"- classification: `{item['classification']}`",
                f"- confidence: `{item['confidence']}`",
                f"- sources: `{item['source_count']}`",
                f"- review approved: `{(item.get('review') or {}).get('approved')}`",
                f"- reasons: {'; '.join(item['score_reasons'])}",
                "",
                "### Answer",
                "",
                item["answer_text"],
                "",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
