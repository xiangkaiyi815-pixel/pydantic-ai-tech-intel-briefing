from __future__ import annotations

import re
from datetime import UTC, datetime

from search_assistant.contracts import Classification, SourceEvidence, VerifiedClaim


_FRESHNESS_WORDS = re.compile(
    r"\b(latest|current|today|yesterday|tomorrow|recent|newest|now|202[0-9])\b",
    re.IGNORECASE,
)
_VERSION_OR_NUMBER = re.compile(r"\b\d+(?:\.\d+){1,}\b|\b\d{4}\b|[$¥€]\s?\d+|\b\d+(?:\.\d+)?%")
_HIGH_STAKES = re.compile(
    r"\b(medical|legal|financial|finance|security|safety|law|tax|investment|doctor|medicine)\b",
    re.IGNORECASE,
)
_API_OR_POLICY = re.compile(r"\b(api|sdk|policy|price|schedule|ranking|benchmark|version)\b", re.IGNORECASE)
_CHINESE_VERIFICATION_SIGNAL = re.compile(
    r"(确认|未确认|核实|验证|证据|官方|规格|参数|模型|内存|显存|网卡|带宽|推理|部署|并行)"
)
_NON_CLAIM_HEADINGS = {
    "证据核查",
    "证据检查",
    "证据检查了什么",
    "核查结果",
    "搜索记录",
    "搜索结果",
    "结论",
    "判断",
    "不确定性",
    "下一步",
}
_ASCII_TOKEN = re.compile(r"[a-z0-9][a-z0-9.+/-]*", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]{2,}")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def requires_verification(text: str, classification: Classification) -> bool:
    if classification in {"research", "hard", "high_stakes"}:
        return True
    return any(
        pattern.search(text)
        for pattern in (_FRESHNESS_WORDS, _VERSION_OR_NUMBER, _HIGH_STAKES, _API_OR_POLICY)
    )


def requires_calibration(classification: Classification, has_unverified_claims: bool) -> bool:
    return classification in {"research", "hard", "high_stakes"} or has_unverified_claims


def extract_key_claims(text: str) -> list[str]:
    claims: list[str] = []
    for sentence in _candidate_claim_sentences(text):
        cleaned = sentence.strip().rstrip(".!?。！？；;")
        if cleaned and _looks_like_key_claim(cleaned):
            claims.append(cleaned)
    return claims


def _candidate_claim_sentences(text: str) -> list[str]:
    candidates: list[str] = []
    in_search_record = False
    in_next_step_section = False
    table_headers: list[str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            in_search_record = False
            table_headers = None
            continue
        normalized_heading = line.strip("#:： ")
        if normalized_heading in {"搜索记录", "Search record", "search record"}:
            in_search_record = True
            continue
        if line.startswith("#"):
            in_next_step_section = _is_next_step_heading(normalized_heading)
            if in_next_step_section:
                continue
            if not _contains_concrete_technical_assertion(normalized_heading):
                continue
        elif in_next_step_section:
            continue
        if in_search_record:
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = _markdown_table_cells(line)
            if not cells:
                continue
            if _is_markdown_table_separator(cells):
                continue
            if table_headers is None:
                table_headers = cells
                continue
            candidates.extend(_markdown_table_claim_candidates(table_headers, cells))
            continue
        table_headers = None
        line = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", line)
        line = re.sub(r"^#+\s*", "", line).strip()
        if line in _NON_CLAIM_HEADINGS:
            continue
        if not line:
            continue
        candidates.extend(
            part.strip()
            for part in re.split(r"(?<=[。！？；;])\s*|(?<=[.!?])\s+", line)
            if part.strip()
        )
    return candidates


def _is_next_step_heading(heading: str) -> bool:
    lowered = heading.lower().strip()
    return lowered in {
        "下一步",
        "下一步验证",
        "下一步可验证",
        "下一步可核实",
        "next step",
        "next steps",
        "next verification",
    } or lowered.startswith(("下一步", "next step", "next verification"))


def _markdown_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_markdown_table_separator(cells: list[str]) -> bool:
    return all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells if cell.strip())


def _markdown_table_claim_candidates(headers: list[str], cells: list[str]) -> list[str]:
    if len(cells) < 2:
        return []
    normalized_headers = [header.lower() for header in headers]
    source_index = _first_matching_index(normalized_headers, ("来源", "source", "url", "页面", "文档"))
    confirmed_index = _first_matching_index(
        normalized_headers,
        ("确认", "confirmed", "supports", "verified", "what", "说明"),
    )
    if source_index is None or confirmed_index is None:
        source_index = 0
        confirmed_index = 1
    if source_index >= len(cells) or confirmed_index >= len(cells):
        return []
    subject = cells[source_index].strip()
    confirmed = cells[confirmed_index].strip()
    if not subject or not confirmed:
        return []
    if subject in _NON_CLAIM_HEADINGS or confirmed in _NON_CLAIM_HEADINGS:
        return []
    return [f"{subject} 确认 {confirmed}"]


def _first_matching_index(values: list[str], needles: tuple[str, ...]) -> int | None:
    for index, value in enumerate(values):
        if any(needle in value for needle in needles):
            return index
    return None


def _looks_like_key_claim(cleaned: str) -> bool:
    if _is_scaffolding_or_user_framing(cleaned):
        return False
    return bool(
            _FRESHNESS_WORDS.search(cleaned)
            or _VERSION_OR_NUMBER.search(cleaned)
            or _HIGH_STAKES.search(cleaned)
            or _API_OR_POLICY.search(cleaned)
            or _CHINESE_VERIFICATION_SIGNAL.search(cleaned)
    )


def _is_scaffolding_or_user_framing(cleaned: str) -> bool:
    stripped = cleaned.strip().strip("*_` ")
    lowered = stripped.lower()
    if _is_next_step_action_item(stripped, lowered):
        return True
    scaffolding_prefixes = (
        "下面",
        "接下来",
        "先说明",
        "我会",
        "我把",
        "本次搜索返回",
        "本轮搜索返回",
        "搜索结果显示",
        "根据搜索结果",
        "根据搜索结果中的",
        "根据现有搜索结果",
        "根据现有文档",
        "根据当前搜索",
        "根据当前搜索获得",
        "证据基础",
        "证据核查",
        "推理路径",
        "直接判断",
        "关键假设",
        "下一步",
        "不确定性",
        "你问的是",
        "你的问题是",
        "你的理解",
        "如果你",
        "we will",
        "i will",
        "based on the search results",
        "according to the search results",
        "the search results show",
        "next step",
        "uncertainty",
        "evidence basis",
        "reasoning path",
        "what i checked",
        "i checked",
        "i inspected",
        "i looked at",
        "what remains unknown",
        "我检查了什么",
        "搜索到的",
        "文档框架",
    )
    user_framing_prefixes = (
        "你的",
        "你关注",
        "你对",
        "你描述",
        "你说",
        "你的说法",
        "你的表述",
        "你理解",
        "your phrasing",
        "your description",
    )
    if any(lowered.startswith(prefix.lower()) or stripped.startswith(prefix) for prefix in user_framing_prefixes):
        return True
    if "相对独立" in stripped and (
        "说法" in stripped or "意味着" in stripped or "默认" in stripped or "程度取决于" in stripped
    ):
        return True
    if stripped.startswith(("我查阅", "这两份文档", "文档明确", "文档框架", "搜索到了", "搜索到的", "它们直接", "这个假设")):
        return True
    if "意味着你默认" in stripped:
        return True
    if ("未找到" in stripped or "没有找到" in stripped) and any(
        prefix in stripped for prefix in ("但同时", "同时", "搜索中", "搜索结果中")
    ):
        return True
    if any(lowered.startswith(prefix.lower()) or stripped.startswith(prefix) for prefix in scaffolding_prefixes):
        return not _contains_concrete_technical_assertion(stripped)
    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9 /＋+、（）()\-]+[:：]", stripped):
        return True
    return False


def _is_next_step_action_item(stripped: str, lowered: str) -> bool:
    action_prefixes = (
        "确认你",
        "确认硬件",
        "确认模型",
        "检查",
        "查看",
        "建议",
        "请确认",
        "请检查",
        "check ",
        "verify ",
        "confirm ",
        "inspect ",
        "search ",
        "look for ",
    )
    if not any(lowered.startswith(prefix.lower()) or stripped.startswith(prefix) for prefix in action_prefixes):
        return False
    return bool(
        "?" in stripped
        or "？" in stripped
        or "：" in stripped
        or ":" in stripped
        or lowered.startswith(("check ", "verify ", "confirm ", "inspect ", "search ", "look for "))
    )


def _contains_concrete_technical_assertion(text: str) -> bool:
    lowered = text.lower()
    technical_subjects = (
        "张量并行",
        "流水线并行",
        "专家并行",
        "数据并行",
        "tensor parallelism",
        "pipeline parallelism",
        "expert parallelism",
        "data parallelism",
        "cxl",
        "compute express link",
        "dgx",
        "gb10",
        "deepseek",
        "h100",
        "genie",
        "cosmos",
        "gr00t",
    )
    assertion_verbs = (
        "将",
        "把",
        "用于",
        "支持",
        "允许",
        "需要",
        "确认",
        "包含",
        "包括",
        "提供",
        "works",
        "supports",
        "uses",
        "requires",
        "includes",
        "provides",
    )
    return any(subject in lowered or subject in text for subject in technical_subjects) and any(
        verb in lowered or verb in text for verb in assertion_verbs
    )


def verify_claims_against_sources(
    claims: list[str],
    sources: list[SourceEvidence],
) -> tuple[list[VerifiedClaim], list[str]]:
    verified: list[VerifiedClaim] = []
    unverified: list[str] = []

    for claim in claims:
        support = _best_supporting_source(claim, sources)
        if support is None:
            unverified.append(claim)
            continue

        source, overlap_ratio, overlap_count = support
        verified.append(
            VerifiedClaim(
                claim=claim,
                verdict="verified",
                source=source.url,
                checked_at=datetime.now(UTC).isoformat(),
                notes=(
                    f"Conservative source match via {source.provider}; "
                    f"token_overlap={overlap_count}, overlap_ratio={overlap_ratio:.2f}"
                ),
            )
        )

    return verified, unverified


def _best_supporting_source(
    claim: str,
    sources: list[SourceEvidence],
) -> tuple[SourceEvidence, float, int] | None:
    claim_tokens = _significant_tokens(claim)
    if not claim_tokens:
        return None

    claim_token_set = set(claim_tokens)
    claim_aliases = set(_semantic_alias_tokens(claim))
    claim_numbers = _number_tokens(claim)
    best: tuple[SourceEvidence, float, int] | None = None

    for source in sources:
        source_text = f"{source.title} {source.snippet}"
        if _has_unbacked_comparison_subject(claim, source_text):
            continue
        source_tokens = set(_significant_tokens(source_text))
        source_aliases = set(_semantic_alias_tokens(source_text))
        if not source_tokens:
            continue
        if claim_numbers and not claim_numbers.issubset(set(_number_tokens(source_text))):
            continue

        alias_overlap = claim_aliases & source_aliases
        alias_overlap_count = len(alias_overlap)
        alias_overlap_ratio = alias_overlap_count / len(claim_aliases) if claim_aliases else 0.0
        if _is_semantically_supported(alias_overlap_count, alias_overlap_ratio, len(claim_aliases)):
            if best is None or alias_overlap_ratio > best[1] or (
                alias_overlap_ratio == best[1] and alias_overlap_count > best[2]
            ):
                best = (source, alias_overlap_ratio, alias_overlap_count)
            continue

        overlap = claim_token_set & source_tokens
        overlap_count = len(overlap)
        overlap_ratio = overlap_count / len(claim_token_set)
        if _is_supported(overlap_count, overlap_ratio, len(claim_token_set)):
            if best is None or overlap_ratio > best[1] or (
                overlap_ratio == best[1] and overlap_count > best[2]
            ):
                best = (source, overlap_ratio, overlap_count)

    return best


def _has_unbacked_comparison_subject(claim: str, source_text: str) -> bool:
    claim_lowered = claim.lower()
    source_lowered = source_text.lower()
    comparison_markers = (
        "本质区别",
        "不同于",
        "区别",
        "相比",
        "compared with",
        "compared to",
        "different from",
        "unlike",
    )
    if not any(marker in claim_lowered or marker in claim for marker in comparison_markers):
        return False
    comparison_subjects = (
        ("数据库", "database"),
        ("微服务", "microservice"),
        ("distributed database", "distributed database"),
        ("关系型数据库", "relational database"),
    )
    for chinese_term, english_term in comparison_subjects:
        claim_mentions_subject = chinese_term in claim or english_term in claim_lowered
        source_mentions_subject = chinese_term in source_text or english_term in source_lowered
        if claim_mentions_subject and not source_mentions_subject:
            return True
    return False


def _is_semantically_supported(overlap_count: int, overlap_ratio: float, alias_count: int) -> bool:
    if alias_count < 3:
        return False
    if alias_count == 3:
        return overlap_count == 3 and overlap_ratio == 1.0
    if alias_count <= 5:
        return overlap_count >= 4 and overlap_ratio >= 0.80
    return overlap_count >= 5 and overlap_ratio >= 0.70


def _is_supported(overlap_count: int, overlap_ratio: float, token_count: int) -> bool:
    if token_count <= 3:
        return overlap_count == token_count
    if token_count <= 5:
        return overlap_count >= 4 and overlap_ratio >= 0.80
    return overlap_count >= 5 and overlap_ratio >= 0.70


def _significant_tokens(text: str) -> list[str]:
    normalized = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text.lower())
    tokens = [
        token.strip(".+/ -")
        for token in _ASCII_TOKEN.findall(normalized)
        if _is_significant_ascii_token(token.strip(".+/ -"))
    ]
    tokens.extend(_cjk_ngrams(normalized))
    tokens.extend(_semantic_alias_tokens(normalized))
    return tokens


def _semantic_alias_tokens(text: str) -> list[str]:
    normalized = text.lower()
    alias_groups = (
        ("distributed_inference", ("distributed inference", "分布式推理", "分布式大模型推理")),
        ("inference_workload", ("inference", "serving", "推理")),
        ("large_model", ("large model", "llm", "大模型")),
        ("model_parallelism", ("model parallelism", "模型并行")),
        ("parallelism_strategy", ("parallelism", "parallelisms", "parallel strategy", "parallel strategies", "并行策略", "并行方式")),
        ("work_partition", ("partition", "partitioned", "split", "shard", "sharded", "分担", "分工", "负责一部分", "拆分", "切分")),
        ("tensor_parallelism", ("tensor parallelism", "tensor parallel", "张量并行")),
        ("pipeline_parallelism", ("pipeline parallelism", "pipeline parallel", "流水线并行")),
        ("expert_parallelism", ("expert parallelism", "expert parallel", "专家并行", "moe并行")),
        ("moe_model", ("moe model", "moe models", "mixture-of-experts", "mixture of experts", "moe 模型", "moe模型")),
        ("too_large_single_node", ("too large for a single node", "single node", "单节点放不下", "单节点装不下")),
        ("pipeline_across_nodes", ("across nodes", "cross node", "跨节点")),
        ("moe_token_dispatch", ("moe token dispatching", "token dispatch", "token dispatcher", "token 分发", "token分发", "token 路由")),
        ("all_to_all", ("all-to-all", "all to all", "全互联", "全到全")),
        ("communication", ("communication", "communicate", "interconnect", "通信", "互联")),
        ("node", ("node", "nodes", "节点")),
        ("kv_cache", ("kv cache", "kv-cache", "键值缓存", "kv缓存")),
        ("memory_bandwidth", ("memory bandwidth", "hbm bandwidth", "带宽", "内存带宽", "显存带宽")),
        ("unified_memory", ("unified memory", "统一内存")),
        ("network_adapter", ("network adapter", "nic", "网卡", "connectx", "connectx-7", "cx7")),
        ("cxl_protocol", ("cxl", "compute express link")),
        ("cache_coherent", ("cache-coherent", "cache coherent", "cache coherency", "缓存一致", "缓存一致性")),
        ("memory_expansion", ("memory expansion", "内存扩展")),
        ("memory_coherency", ("memory coherency", "memory coherent", "内存一致", "内存一致性")),
        ("bandwidth_doubling", ("doubles bandwidth", "increases the bandwidth", "带宽提升", "提升至")),
        ("bundled_ports", ("bundled ports", "捆绑端口")),
        ("memory_ras", ("memory ras", "内存 ras")),
        ("processor_cpu", ("processor", "processors", "cpu", "处理器")),
        ("accelerator_device", ("accelerator", "accelerators", "加速器")),
        ("cpu_complement", ("complement cpus", "complement cpu", "补充cpu", "补充 CPU")),
        ("resource_sharing", ("resource sharing", "资源共享", "共享")),
        ("open_standard", ("open standard", "开放标准")),
        ("high_speed_communication", ("high-speed communications", "high speed communications", "高速通信", "高速互连")),
        ("ai_ml_workload", ("artificial intelligence", "machine learning", "ai/ml", "ai workload", "ai workloads", "ai server", "ai servers", "ai服务器", "ai 工作负载", "ai工作负载")),
        ("world_model", ("world model", "world models", "世界模型")),
        ("foundation_world_model", ("foundation world model", "foundation world models", "基础世界模型", "世界基础模型")),
        ("large_scale_model", ("large-scale", "large scale", "大规模")),
        ("interactive_3d_world", ("interactive", "3d worlds", "3d world", "3d环境", "交互式")),
        ("single_image_prompt", ("single image", "单张图像", "单张图片")),
        ("research_framing", ("research", "研究系统", "研究")),
        ("open_reference_platform", ("open reference platform", "开放参考平台", "开源参考平台")),
        ("humanoid_robot", ("humanoid robot", "humanoid robots", "人形机器人")),
        ("robot_foundation_model", ("robot foundation model", "robot foundation models", "机器人基础模型")),
        ("downloadable_model", ("download", "downloadable", "可下载", "下载")),
    )
    aliases: list[str] = []
    for canonical, phrases in alias_groups:
        if any(phrase in normalized for phrase in phrases):
            aliases.append(canonical)
    return aliases


def _is_significant_ascii_token(token: str) -> bool:
    if not token or token in _STOPWORDS:
        return False
    return len(token) >= 3 or any(character.isdigit() for character in token)


def _cjk_ngrams(text: str) -> list[str]:
    grams: list[str] = []
    for run in _CJK_RUN.findall(text):
        if len(run) <= 3:
            grams.append(run)
            continue
        grams.extend(run[index : index + 3] for index in range(len(run) - 2))
    return grams


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"(?<!\d)\d+(?:\.\d+)*(?!\d)", text.lower()))
