from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
import re
import uuid
from typing import Protocol
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from search_assistant.contracts import (
    BriefingSynthesis,
    BriefingTheme,
    CollectedSource,
    DailyBriefing,
    ProviderTraceEvent,
    SourceCandidate,
    TopicSubscription,
)
from search_assistant.memory.store import MemoryStore
from search_assistant.search.provider import SearchClient, SearchOutcome, SearchResult, search_with_provider_events
from search_assistant.search.source_registry import normalize_source_recipe, source_recipe_summary


class BriefingRuntime(Protocol):
    def plan_briefing_queries(self, topic: str, context: dict[str, object]) -> list[str]:
        ...

    def synthesize_briefing(
        self,
        topic: str,
        sources: list[CollectedSource],
        context: dict[str, object],
    ) -> BriefingSynthesis | None:
        ...


@dataclass(frozen=True)
class BriefingCollectionTrace:
    sources: list[CollectedSource]
    source_candidates: list[SourceCandidate]
    provider_events: list[ProviderTraceEvent]


REPORT_CONTRACT = {
    "language": "zh-CN",
    "required_sections": [
        "搜索方向",
        "关键词",
        "搜索内容总结",
        "简短总结",
        "详细总结",
        "AI 分析判断",
        "下一步搜索方向",
        "落地建议",
        "原文链接",
    ],
    "detail_sections": "模型按本轮证据自行组织 2 至 5 个分析块，不使用固定技术地图模板。",
}

REPORT_SKILL_PATH = Path(__file__).resolve().parents[3] / "skills" / "content-collection-report" / "SKILL.md"


CHANNEL_QUERIES: tuple[tuple[str, str], ...] = (
    ("全球网页", "{topic} AI industry technology product release research"),
    ("中文网页", "{topic} 人工智能 技术 产业 落地 发布"),
    ("微信公众号（百度公开索引）", "site:mp.weixin.qq.com {topic}"),
    ("抖音", "site:douyin.com {topic}"),
    ("哔哩哔哩", "site:bilibili.com {topic}"),
    ("今日头条（百度公开索引）", "site:toutiao.com {topic}"),
    ("小红书（百度公开索引）", "site:xiaohongshu.com {topic}"),
    ("小红书分享链接（百度公开索引）", "site:xhslink.com {topic}"),
    ("知乎", "site:zhihu.com {topic}"),
    ("LinkedIn", "site:linkedin.com {topic} AI"),
    ("X", "site:x.com {topic} AI"),
    ("Reddit", "site:reddit.com {topic} AI"),
    ("YouTube", "site:youtube.com {topic} AI"),
    ("工程社区", "site:infoq.cn OR site:csdn.net OR site:36kr.com {topic}"),
)


THEMES: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "工业智能体与生产协同",
        ("agent", "智能体", "mes", "erp", "scada", "plc", "调度", "工单"),
        "把自然语言任务、生产约束和企业系统接口连接起来的 Agent 编排层。",
        "关键不在聊天，而在受控工具调用、权限、工单闭环和人工接管。",
        "已出现明确场景，但跨系统权限、审计和可靠执行仍决定能否规模化。",
    ),
    (
        "数字孪生、仿真与虚拟调试",
        ("digital twin", "数字孪生", "simulation", "仿真", "virtual commissioning", "虚拟调试"),
        "用设备、产线或工厂的可计算模型在物理部署前验证工艺和控制策略。",
        "核心是模型输入、约束求解、what-if 推演和与现场数据同步，不是可视化大屏。",
        "工程软件和平台侧较成熟，和实时生产数据、控制系统的闭环仍需验证。",
    ),
    (
        "机器视觉质检与质量追溯",
        ("aoi", "machine vision", "机器视觉", "视觉", "质检", "缺陷", "质量"),
        "从缺陷识别延伸到原因定位、批次追溯和工艺参数回写。",
        "核心技术包括多模态视觉、异常检测、缺陷样本闭环和质量数据关联。",
        "视觉识别已较常见，跨批次泛化和与工艺调参闭环仍是难点。",
    ),
    (
        "预测性维护与设备优化",
        ("predictive maintenance", "预测性维护", "设备维护", "停机", "oee", "参数调优"),
        "将传感器时序、故障模式和维修工单串成提前预警与执行闭环。",
        "价值来自时间序列异常检测、剩余寿命估计、因果排查和维修调度联动。",
        "局部设备预测已可落地，数据质量和跨设备迁移决定覆盖范围。",
    ),
    (
        "工业软件 Copilot 与工程自动化",
        ("copilot", "hmi", "cad", "cae", "ansys", "工业软件", "工程"),
        "把需求、规范和工程资产转为辅助建模、代码生成、仿真配置或控制界面。",
        "关键是领域知识检索、结构化工程资产、可验证生成和版本控制。",
        "目前多为工程师增强工具，进入安全关键控制环节前需要严格验证。",
    ),
)


CAD_THEMES: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "生成式 CAD 与参数化建模",
        (
            "text-to-cad", "text to cad", "cad generation", "cad copilot", "parametric cad",
            "参数化建模", "cad生成", "文本到cad", "特征建模", "草图约束",
        ),
        "把自然语言、草图或参考图转换为可编辑的草图、特征树和参数，而不是只生成不可修改的三维网格。",
        "工程可用性的分界线在于生成结果能否保留尺寸、约束、特征依赖和版本历史，从而进入后续修改与制造流程。",
        "文本和图像到概念几何已较活跃；稳定生成可编辑的参数化实体仍需要约束求解、特征有效性校验和人工复核。",
    ),
    (
        "二维工程图理解、矢量化与图纸生成",
        (
            "engineering drawing", "2d drawing", "technical drawing", "drawing generation", "drawing understanding",
            "dwg", "dxf", "vectorization", "图纸生成", "工程图", "二维图", "矢量化", "标注识别",
        ),
        "将扫描图、PDF 或二维草图解析为几何、尺寸、标注和图层语义，再生成可编辑的二维 CAD 图纸。",
        "它决定存量图纸能否进入结构化工程数据链路，也决定自动出图是否能通过尺寸、公差和标准图框校核。",
        "OCR 与线条矢量化已有成熟组件；尺寸语义、符号标准和复杂装配图的拓扑还原仍需领域规则参与。",
    ),
    (
        "三维模型到二维工程图的自动出图",
        (
            "3d to 2d", "3d-to-2d", "drawing extraction", "orthographic projection", "section view",
            "投影视图", "三维转二维", "自动出图", "剖视图", "三视图", "尺寸标注",
        ),
        "从 B-Rep 或参数化三维模型中选择视角，生成主视图、剖视图、隐藏线、尺寸和 BOM 关联的二维交付物。",
        "该路线把三维设计数据直接复用于加工、检验和供应链沟通，减少人工制图时的视图漏选与版本漂移。",
        "常规投影和出图能力已存在于 CAD 软件；AI 的价值主要在视图选择、标注建议和规则冲突提示，必须由制图标准校验。",
    ),
    (
        "几何表示、约束校验与制造可用性",
        (
            "b-rep", "brep", "boundary representation", "geometric constraint", "constraint solving",
            "gdt", "gd&t", "tolerance", "几何约束", "公差", "边界表示", "可制造性",
        ),
        "围绕 B-Rep、草图约束、拓扑一致性和公差规则建立生成后的可验证层，而不是把模型输出直接当成工程交付。",
        "这层校验决定 AI 图纸能否被下游 CAD、CAM、CAE 和质量体系消费，也是防止貌似正确但不可制造几何的关键。",
        "几何内核和规则校验可复用，但把学习模型输出稳定映射到可求解、可制造的实体仍处于工程化整合阶段。",
    ),
)

_CAD_TOPIC_MARKERS = (
    "cad", "工程图", "二维", "三维", "2d", "3d", "dwg", "dxf", "b-rep", "brep",
    "参数化", "建模", "制图", "投影", "图纸", "草图", "公差",
)
_CAD_EVIDENCE_ANCHORS = (
    "cad", "工程图", "dwg", "dxf", "b-rep", "brep", "参数化", "草图约束", "几何约束",
    "尺寸标注", "公差", "投影视图", "剖视图", "三视图", "机械制图", "实体建模",
)


_PLATFORM_TERMS = {
    "抖音", "哔哩哔哩", "b站", "微信公众号", "微信", "今日头条", "头条", "小红书", "知乎",
    "linkedin", "youtube", "reddit", "发布在", "原文链接",
}
_NOISE_TERMS = {"已经收获", "先问问", "你的手和脑", "请关注", "点击查看", "完整视频"}


_GENERIC_REFERENCE_DOMAINS = {
    "amap.com",
    "baike.baidu.com",
    "hanyuguoxue.com",
    "iciba.com",
    "jingyan.baidu.com",
    "zhidao.baidu.com",
    "mayoclinic.org",
    "wikipedia.org",
}
_SEARCH_DUMP_MARKERS = (
    "not_struct",
    "renderflags",
    "cosmicpkgs",
    "pagestyleupgrade",
    "\"hastop\"",
    "\"isencoding\"",
    "-->",
)
_CAD_MEDICAL_MARKERS = (
    "coronary artery disease",
    "cardiovascular disease",
    "heart disease",
    "冠状动脉疾病",
    "心血管疾病",
)
_LOGIN_PATH_MARKERS = ("/login", "/signin", "/sign-in", "/auth/login", "/account/login")
_QUERY_STOPWORDS = {
    "and",
    "for",
    "from",
    "global",
    "industry",
    "product",
    "release",
    "research",
    "site",
    "technology",
    "the",
    "with",
}
_TECHNICAL_SIGNAL_MARKERS = (
    "industrial ai",
    "manufacturing ai",
    "\u5de5\u4e1aai",
    "\u5de5\u4e1a\u4eba\u5de5\u667a\u80fd",
    "\u667a\u80fd\u5236\u9020",
    "\u4e91\u8fb9\u534f\u540c",
    "\u8fb9\u7f18\u63a8\u7406",
    "\u65f6\u5e8f\u6570\u636e",
    "\u5f02\u5e38\u68c0\u6d4b",
    "\u4f20\u611f\u5668",
    "robotics",
    "\u673a\u5668\u4eba",
)
_INDUSTRIAL_ANCHORS = (
    "industrial",
    "manufactur",
    "factory",
    "maintenance",
    "\u5de5\u4e1a",
    "\u5236\u9020",
    "\u5de5\u5382",
    "\u8bbe\u5907",
    "\u8bbe\u5907\u7ef4\u62a4",
    "\u9884\u6d4b\u6027\u7ef4\u62a4",
    "\u8d28\u68c0",
    "\u7f3a\u9677",
)
_INDUSTRIAL_ACRONYM_ANCHORS = ("mes", "erp", "scada", "plc")


def _clean_planned_queries(queries: object) -> list[str]:
    if not isinstance(queries, list):
        return []
    cleaned: list[str] = []
    for raw_query in queries:
        query = " ".join(str(raw_query).replace("site:", "").split())
        if not _is_meaningful_query(query):
            continue
        if query.lower() in {item.lower() for item in cleaned}:
            continue
        cleaned.append(query[:180])
        if len(cleaned) >= 8:
            break
    return cleaned


def _is_meaningful_feedback(value: str) -> bool:
    return _is_meaningful_query(value) and len(value) >= 6


def _is_meaningful_query(value: str) -> bool:
    normalized = " ".join(value.split()).strip()
    if len(normalized) < 3:
        return False
    lowered = normalized.lower()
    if any(term in lowered for term in _PLATFORM_TERMS | _NOISE_TERMS):
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", normalized))


def _load_report_skill() -> str:
    try:
        return REPORT_SKILL_PATH.read_text(encoding="utf-8")
    except OSError:
        return "Follow the fixed Chinese content collection report contract and synthesize technology themes, not search logs."


def _is_substantive_synthesis(synthesis: BriefingSynthesis) -> bool:
    if not synthesis.themes or len(synthesis.analysis_judgment.strip()) < 60:
        return False

    # New model route: the detailed narrative owns the shape of the analysis.
    # Themes are evidence anchors, not a mandatory per-theme form.
    if synthesis.detailed_summary.strip():
        return (
            len(synthesis.short_summary.strip()) >= 75
            and len(synthesis.detailed_summary.strip()) >= 240
            and all(
                len(theme.analysis.strip()) >= 30 and bool(theme.source_urls)
                for theme in synthesis.themes
            )
        )

    # Accept older model responses while the deterministic fallback remains
    # available, but do not let this legacy shape dictate new generations.
    if len(synthesis.short_summary.strip()) < 60:
        return False
    if len(synthesis.key_signal_interpretation.strip()) < 80:
        return False
    return all(
        len(theme.what_is_happening.strip()) >= 45
        and len(theme.core_technology.strip()) >= 20
        and len(theme.data_and_workflow.strip()) >= 45
        and len(theme.why_it_matters.strip()) >= 30
        and len(theme.maturity.strip()) >= 10
        and bool(theme.source_urls)
        for theme in synthesis.themes
    )


def _is_cad_topic(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _CAD_TOPIC_MARKERS)


def _has_cad_anchor(title: str, snippet: str) -> bool:
    text = f"{title} {snippet}".lower()
    for marker in _CAD_EVIDENCE_ANCHORS:
        if marker.isascii() and re.fullmatch(r"[a-z0-9-]+", marker):
            if re.search(rf"\b{re.escape(marker)}\b", text):
                return True
        elif marker in text:
            return True
    return False


def _deterministic_technical_queries(topic: str) -> list[str]:
    if _is_cad_topic(topic):
        return [
            "Text-to-CAD parametric B-Rep generation open source evaluation",
            "AI engineering drawing generation 3D to 2D projection dimensioning CAD",
            "2D engineering drawing vectorization DWG DXF OCR CAD workflow",
            "CAD copilot sketch constraint solving feature modeling architecture",
            "B-Rep topology validation geometric constraints manufacturability generated CAD",
        ]
    return [
        f"{topic} technical architecture implementation evaluation",
        f"{topic} open source repository paper dataset benchmark",
        f"{topic} deployment workflow data interface case study",
    ]


def _order_plan_by_source_recipe(
    plans: list[tuple[str, str]],
    source_recipe: dict[str, float] | None,
) -> list[tuple[str, str]]:
    if not plans:
        return []
    if not source_recipe:
        return plans
    recipe = normalize_source_recipe(source_recipe)
    indexed = list(enumerate(plans))
    return [
        plan
        for _, plan in sorted(
            indexed,
            key=lambda item: (
                -recipe.get(_source_slug_for_plan(*item[1]), recipe.get("general-web", 1.0)),
                item[0],
            ),
        )
    ]


def _source_slug_for_plan(platform: str, query: str) -> str:
    lowered = f"{platform} {query}".lower()
    if "mp.weixin.qq.com" in lowered or "weixin.qq.com" in lowered:
        return "wechat-public-index"
    if "bilibili.com" in lowered or "b站" in lowered or "bilibili" in lowered:
        return "bilibili"
    if "zhihu.com" in lowered or "知乎" in lowered:
        return "zhihu"
    if "toutiao.com" in lowered or "今日头条" in lowered:
        return "toutiao-public-index"
    if "xiaohongshu.com" in lowered or "xhslink.com" in lowered or "小红书" in lowered:
        return "xiaohongshu-public-index"
    if "youtube.com" in lowered:
        return "youtube"
    if "reddit.com" in lowered:
        return "reddit"
    if "linkedin.com" in lowered:
        return "linkedin-public-index"
    if "site:x.com" in lowered or "twitter.com" in lowered:
        return "x-public-index"
    if "github.com" in lowered or "arxiv.org" in lowered:
        return "mcp-public"
    return "general-web"


class DailyBriefingService:
    def __init__(
        self,
        store: MemoryStore,
        search_client: SearchClient,
        runtime: BriefingRuntime | None = None,
        max_queries: int = 24,
        results_per_query: int = 10,
        max_sources: int = 50,
        model_max_sources: int = 12,
        search_budget_seconds: float = 90.0,
        timezone_name: str = "Asia/Shanghai",
    ):
        self.store = store
        self.search_client = search_client
        self.runtime = runtime
        self.max_queries = max_queries
        self.results_per_query = results_per_query
        self.max_sources = max_sources
        self.model_max_sources = model_max_sources
        self.search_budget_seconds = search_budget_seconds
        self.timezone_name = timezone_name

    def run(
        self,
        topic: str,
        user_id: str,
        chat_id: str,
        run_date: date | None = None,
    ) -> DailyBriefing:
        subscription = self.store.upsert_topic(user_id, chat_id, topic)
        feedback = self.store.list_topic_feedback(subscription.id)
        search_plan = self.build_search_plan(subscription, feedback)
        briefing_id = f"brief_{uuid.uuid4().hex}"
        collection = self._collect_sources_with_trace(subscription, search_plan, feedback, run_id=briefing_id)
        sources = collection.sources
        ranked_sources = sorted(sources, key=lambda source: source.importance_score, reverse=True)[: self.max_sources]
        synthesis = self._synthesize(
            subscription.topic,
            ranked_sources[: self.model_max_sources],
            search_plan,
        )
        resolved_date = run_date or self._briefing_date()
        briefing = DailyBriefing(
            id=briefing_id,
            topic_id=subscription.id,
            user_id=user_id,
            chat_id=chat_id,
            topic=subscription.topic,
            run_date=resolved_date,
            search_directions=[f"{platform}: {query}" for platform, query in search_plan],
            keywords=self._keywords(subscription.topic, feedback, search_plan),
            sources=ranked_sources,
            source_candidates=collection.source_candidates,
            provider_events=collection.provider_events,
            synthesis=synthesis,
            markdown="",
            created_at=datetime.now(UTC).isoformat(),
        )
        briefing.markdown = self.render_markdown(briefing)
        self.store.record_daily_briefing(briefing)
        return briefing

    def build_search_plan(
        self,
        subscription: TopicSubscription,
        feedback: list[dict[str, object]],
    ) -> list[tuple[str, str]]:
        plans: list[tuple[str, str]] = []
        if self.runtime is not None:
            try:
                generated = self.runtime.plan_briefing_queries(
                    subscription.topic,
                    {
                        "feedback": feedback[:5],
                        "channels": [platform for platform, _ in CHANNEL_QUERIES],
                        "report_skill": _load_report_skill(),
                        "source_recipe": source_recipe_summary(subscription.source_recipe),
                    },
                )
            except Exception:
                generated = []
            plans.extend(("技术路线", query) for query in _clean_planned_queries(generated))
        plans.extend(("技术路线（确定性保障）", query) for query in _deterministic_technical_queries(subscription.topic))

        for item in feedback[:3]:
            note = " ".join(str(item.get("body") or "").split())
            if note and _is_meaningful_feedback(note):
                plans.append(("案例反馈", f"{subscription.topic} {note[:160]}"))
        plans.extend((platform, template.format(topic=subscription.topic)) for platform, template in CHANNEL_QUERIES)
        unique: list[tuple[str, str]] = []
        seen: set[str] = set()
        for platform, query in plans:
            normalized = query.lower().strip()
            if normalized and normalized not in seen:
                unique.append((platform, query))
                seen.add(normalized)
        return _order_plan_by_source_recipe(unique, subscription.source_recipe)[: self.max_queries]

    def add_feedback(
        self,
        topic: str,
        user_id: str,
        chat_id: str,
        body: str,
        source_url: str | None = None,
    ) -> str:
        subscription = self.store.upsert_topic(user_id, chat_id, topic)
        enriched_body = body.strip()
        case_context = self._inspect_case_source(source_url)
        if case_context:
            enriched_body = "\n".join(part for part in (enriched_body, case_context) if part)
        feedback_id = self.store.add_topic_feedback(subscription.id, user_id, chat_id, enriched_body, source_url)
        signal_type, scope = self._feedback_signal_type(enriched_body, source_url)
        self.store.add_topic_feedback_signal(
            subscription.id,
            user_id,
            chat_id,
            signal_type=signal_type,
            scope=scope,
            body=enriched_body,
            source_url=source_url,
            metadata={"feedback_id": feedback_id},
        )
        memory_layer = "preference" if signal_type == "style" else "run_experience"
        self.store.add_layered_memory_item(
            memory_layer,
            f"briefing_{signal_type}_feedback",
            enriched_body,
            source_id=feedback_id,
            confidence="medium",
            status="active",
            user_id=user_id,
            chat_id=chat_id,
            metadata={"topic_id": subscription.id, "scope": scope, "source_url": source_url},
        )
        return feedback_id

    @staticmethod
    def _feedback_signal_type(body: str, source_url: str | None) -> tuple[str, str]:
        lowered = body.lower()
        if any(term in lowered for term in ("too long", "too verbose", "readability", "style", "简洁", "太长", "难读")):
            return "style", "topic"
        if any(term in lowered for term in ("provider", "agent reach", "search source", "没走", "来源", "搜索源")):
            return "provider", "provider"
        if any(term in lowered for term in ("wrong", "incorrect", "事实", "错误", "纠正")):
            return "fact_correction", "source" if source_url else "topic"
        if source_url:
            return "case", "source"
        if any(term in lowered for term in ("evidence", "source", "证据", "原文", "引用")):
            return "evidence", "source"
        return "general", "topic"

    def _inspect_case_source(self, source_url: str | None) -> str:
        if not source_url:
            return ""
        normalized_url = self._normalize_url(source_url)
        if not normalized_url:
            return ""
        try:
            results = self.search_client.search(normalized_url, limit=3)
        except Exception:
            return ""
        if not results:
            return ""
        matched = next(
            (result for result in results if self._normalize_url(result.url) == normalized_url),
            results[0],
        )
        title = " ".join(matched.title.split())[:240]
        snippet = " ".join(matched.snippet.split())[:800]
        if not title and not snippet:
            return ""
        return f"案例公开索引：{title}\n{snippet}".strip()

    def _collect_sources(
        self,
        subscription: TopicSubscription,
        search_plan: list[tuple[str, str]],
        feedback: list[dict[str, object]],
    ) -> list[CollectedSource]:
        return self._collect_sources_with_trace(subscription, search_plan, feedback).sources

    def _collect_sources_with_trace(
        self,
        subscription: TopicSubscription,
        search_plan: list[tuple[str, str]],
        feedback: list[dict[str, object]],
        run_id: str | None = None,
    ) -> BriefingCollectionTrace:
        by_url: dict[str, CollectedSource] = {}
        query_results: list[tuple[str, str, SearchOutcome]] = []
        source_candidates: list[SourceCandidate] = []
        provider_events: list[ProviderTraceEvent] = []
        workers = min(6, len(search_plan))
        executor = ThreadPoolExecutor(max_workers=max(1, workers))
        futures = [
            (requested_platform, query, executor.submit(self._search_with_trace, query))
            for requested_platform, query in search_plan
        ]
        try:
            completed, pending = wait(
                [future for _, _, future in futures],
                timeout=self.search_budget_seconds,
            )
            for requested_platform, query, future in futures:
                if future not in completed:
                    continue
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = SearchOutcome(
                        results=[],
                        provider_events=[
                            ProviderTraceEvent(
                                provider="briefing-search",
                                query=query,
                                status="error",
                                result_count=0,
                                error=f"{type(exc).__name__}: {exc}"[:240],
                                checked_at=datetime.now(UTC).isoformat(),
                            )
                        ],
                    )
                query_results.append((requested_platform, query, outcome))
                for event in outcome.provider_events:
                    provider_events.append(event)
                    self.store.record_provider_trace_event(event, run_id=run_id, topic_id=subscription.id)
            for future in pending:
                future.cancel()
        finally:
            # Do not block the daily briefing on slow public endpoints. Completed
            # results are retained; queued calls are cancelled when possible.
            executor.shutdown(wait=False, cancel_futures=True)

        for requested_platform, query, outcome in query_results:
            for result in outcome.results:
                normalized_url = self._normalize_url(result.url)
                title = self._compact_source_text(result.title, 240)
                snippet = self._compact_source_text(result.snippet, 900)
                relevance_score = max(
                    self._relevance_score(subscription.topic, title, snippet),
                    self._query_relevance_score(query, title, snippet),
                )
                importance_score = self._importance_score(
                    subscription.topic,
                    query,
                    title,
                    snippet,
                    result.provider,
                )
                rejection = self._source_rejection_reason(subscription.topic, query, normalized_url, title, snippet)
                if rejection is not None:
                    rejected = self._source_candidate(
                        subscription=subscription,
                        requested_platform=requested_platform,
                        query=query,
                        result=result,
                        normalized_url=normalized_url,
                        title=title,
                        snippet=snippet,
                        status=rejection[0],
                        reason=rejection[1],
                        relevance_score=relevance_score,
                        importance_score=importance_score,
                    )
                    source_candidates.append(rejected)
                    self.store.record_source_candidate(rejected)
                    continue
                candidate = CollectedSource(
                    id=f"src_{uuid.uuid4().hex}",
                    topic_id=subscription.id,
                    user_id=subscription.user_id,
                    title=title,
                    url=normalized_url,
                    snippet=snippet,
                    platform=self._platform_from_url(normalized_url, requested_platform),
                    provider=result.provider,
                    query=query,
                    relevance_score=relevance_score,
                    importance_score=importance_score,
                    retrieved_at=result.checked_at,
                )
                prior = by_url.get(normalized_url)
                if prior is not None and candidate.importance_score <= prior.importance_score:
                    duplicate = self._source_candidate(
                        subscription=subscription,
                        requested_platform=requested_platform,
                        query=query,
                        result=result,
                        normalized_url=normalized_url,
                        title=title,
                        snippet=snippet,
                        status="duplicate",
                        reason="same normalized URL already has an equal or stronger candidate",
                        relevance_score=relevance_score,
                        importance_score=importance_score,
                    )
                    source_candidates.append(duplicate)
                    self.store.record_source_candidate(duplicate)
                    continue
                if prior is None or candidate.importance_score > prior.importance_score:
                    by_url[normalized_url] = candidate

        for item in feedback:
            source_url = str(item.get("source_url") or "").strip()
            if not source_url or source_url in by_url:
                continue
            body = str(item.get("body") or "")
            by_url[source_url] = CollectedSource(
                id=f"src_{uuid.uuid4().hex}",
                topic_id=subscription.id,
                user_id=subscription.user_id,
                title="用户提供案例",
                url=source_url,
                snippet=body,
                platform="用户案例",
                provider="user-feedback",
                query=subscription.topic,
                relevance_score=10.0,
                importance_score=12.0,
                retrieved_at=str(item.get("created_at") or datetime.now(UTC).isoformat()),
            )
            feedback_candidate = SourceCandidate(
                id=f"cand_{uuid.uuid4().hex}",
                topic_id=subscription.id,
                user_id=subscription.user_id,
                title=by_url[source_url].title,
                url=source_url,
                snippet=self._compact_source_text(body, 900),
                platform=by_url[source_url].platform,
                provider="user-feedback",
                query=subscription.topic,
                status="feedback_seed",
                reason="user supplied a public case URL",
                relevance_score=10.0,
                importance_score=12.0,
                retrieved_at=by_url[source_url].retrieved_at,
                created_at=datetime.now(UTC).isoformat(),
            )
            source_candidates.append(feedback_candidate)
            self.store.record_source_candidate(feedback_candidate)

        persisted: list[CollectedSource] = []
        for source in by_url.values():
            stored_source = self.store.upsert_collected_source(source)
            persisted.append(stored_source)
            accepted = SourceCandidate(
                id=f"cand_{uuid.uuid4().hex}",
                topic_id=subscription.id,
                user_id=subscription.user_id,
                title=stored_source.title,
                url=stored_source.url,
                snippet=stored_source.snippet,
                platform=stored_source.platform,
                provider=stored_source.provider,
                query=stored_source.query,
                status="accepted",
                reason="accepted for briefing ranking",
                relevance_score=stored_source.relevance_score,
                importance_score=stored_source.importance_score,
                retrieved_at=stored_source.retrieved_at,
                created_at=datetime.now(UTC).isoformat(),
            )
            source_candidates.append(accepted)
            self.store.record_source_candidate(accepted)
        return BriefingCollectionTrace(
            sources=persisted,
            source_candidates=source_candidates,
            provider_events=provider_events,
        )

    def _search(self, query: str) -> list[SearchResult]:
        return self.search_client.search(query, limit=self.results_per_query)

    def _search_with_trace(self, query: str) -> SearchOutcome:
        return search_with_provider_events(self.search_client, query, limit=self.results_per_query)

    def _synthesize(
        self,
        topic: str,
        sources: list[CollectedSource],
        search_plan: list[tuple[str, str]],
    ) -> BriefingSynthesis:
        if self.runtime is not None:
            try:
                generated = self.runtime.synthesize_briefing(
                    topic,
                    sources,
                    {
                        "report_contract": REPORT_CONTRACT,
                        "report_skill": _load_report_skill(),
                        "search_plan": search_plan,
                    },
                )
            except Exception:
                generated = None
            if generated is not None and _is_substantive_synthesis(generated):
                return self._preserve_generated_detail(generated, topic, sources)
        return self._fallback_synthesis(topic, sources)

    def _preserve_generated_detail(
        self,
        generated: BriefingSynthesis,
        topic: str,
        sources: list[CollectedSource],
    ) -> BriefingSynthesis:
        # A generated detailed summary has already selected its own evidence
        # structure.  Do not inject catalogue themes merely to reach a count.
        if generated.detailed_summary.strip() or len(generated.themes) >= 3 or len(sources) < 3:
            return generated
        fallback = self._fallback_synthesis(topic, sources)
        if len(fallback.themes) < 3:
            return generated
        return generated.model_copy(update={"themes": fallback.themes})

    def _fallback_synthesis(self, topic: str, sources: list[CollectedSource]) -> BriefingSynthesis:
        theme_catalog = CAD_THEMES + THEMES if _is_cad_topic(topic) else THEMES
        themed_sources: dict[str, list[CollectedSource]] = {}
        for source in sources:
            haystack = f"{source.title} {source.snippet}".lower()
            matched = False
            for name, markers, _, _, _ in theme_catalog:
                if any(self._matches_technical_marker(haystack, marker) for marker in markers):
                    themed_sources.setdefault(name, []).append(source)
                    matched = True
                    break
            if not matched:
                themed_sources.setdefault("产业落地与产品动态", []).append(source)

        themes: list[BriefingTheme] = []
        for name, markers, technology, importance, maturity in theme_catalog:
            evidence = themed_sources.get(name, [])
            if not evidence:
                continue
            themes.append(
                BriefingTheme(
                    name=name,
                    analysis=self._fallback_theme_analysis(evidence, technology, importance, maturity),
                    what_is_happening=self._evidence_summary(evidence),
                    core_technology=technology,
                    why_it_matters=importance,
                    maturity=maturity,
                    data_and_workflow=self._evidence_workflow(evidence),
                    source_urls=[source.url for source in evidence[:4]],
                )
            )
        residual = themed_sources.get("产业落地与产品动态", [])
        residual_name = "工程图与 CAD 工具链线索" if _is_cad_topic(topic) else "产业落地与产品动态"
        if residual:
            themes.append(
                BriefingTheme(
                    name=residual_name,
                    analysis=self._fallback_theme_analysis(
                        residual,
                        "从产品发布、部署经验和行业讨论中识别可重复的技术路线与约束。",
                        "它提供了需求侧和供给侧的信号，但需要继续回到技术实现和量化指标核验。",
                        "信息线索阶段，需补充原始技术材料和现场指标。",
                    ),
                    what_is_happening=self._evidence_summary(residual),
                    core_technology="从产品发布、部署经验和行业讨论中识别可重复的技术路线与约束。",
                    why_it_matters="它提供了需求侧和供给侧的信号，但需要继续回到技术实现和量化指标核验。",
                    maturity="信息线索阶段，需补充原始技术材料和现场指标。",
                    data_and_workflow=self._evidence_workflow(residual),
                    source_urls=[source.url for source in residual[:4]],
                )
            )
        if not themes:
            themes.append(
                BriefingTheme(
                    name="待补充技术主题",
                    analysis="本轮没有足够的原始技术材料，不能据此还原具体实现路径或工程边界。",
                    what_is_happening="本轮公开检索没有返回可用于技术归纳的有效线索。",
                    core_technology="需要从原始论文、产品文档、代码仓库或公开演讲补齐。",
                    why_it_matters="没有足够原始材料时，不应把搜索空白误写成技术结论。",
                    maturity="未判定。",
                    data_and_workflow="尚无可用于还原输入、处理和回写流程的公开材料。",
                )
            )

        source_count = len(sources)
        theme_names = "、".join(theme.name for theme in themes[:4])
        if _is_cad_topic(topic):
            return BriefingSynthesis(
                search_content_summary=(
                    f"本轮围绕“{topic}”保留了 {source_count} 条高相关公开线索，内容可归为{theme_names}。"
                    "讨论的共同目标不是把文本或图片直接变成展示模型，而是把输入转成可编辑的 CAD 几何、二维图纸和可校验的工程约束。"
                ),
                short_summary=(
                    "这批线索指向同一条工程化路径：前端用语言、草图、扫描图或三维模型提供设计意图，"
                    "中间层生成参数化特征、投影视图与标注，末端以几何约束、公差和制图标准校验输出。"
                    "真正的技术门槛不在图像生成，而在生成结果能否继续编辑、版本化并进入 CAD/CAM/CAE 工具链。"
                ),
                detailed_summary=self._fallback_detailed_summary(themes, key_signal_interpretation=(
                    "本轮线索共同指向从视觉化概念模型走向可编辑、可校验工程实体的迁移；"
                    "后续需要核验每个方案是否输出 B-Rep 或原生特征树，以及约束、公差和制图规则是否由确定性工具执行。"
                )),
                themes=themes,
                key_signal_interpretation=(
                    "从来源中反复出现的 CAD、二维/三维互转、参数化和工程图关键词可以看出，"
                    "产品路线正在从“视觉上像零件”的网格生成，转向“拓扑、尺寸和约束可被工程软件读取”的结构化模型。"
                    "比较方案时应逐项核验：输入是否保留设计意图、输出是否为 B-Rep/特征树而非网格、"
                    "二维图的尺寸与公差是否可编辑、以及是否有规则校验阻止不可制造结果。"
                ),
                analysis_judgment=(
                    "跨来源的结论是，生成式模型适合缩短草图到初版几何、三维到二维出图和图纸数字化的准备时间，"
                    "但不应取代几何内核、约束求解器和制图规范。可落地方案需要把 LLM 或视觉模型限定在候选生成与语义提取，"
                    "再由 CAD API、规则引擎和工程师审批完成确定性校验与发布。"
                ),
                next_search_directions=[
                    "追踪 Text-to-CAD、B-Rep 生成、草图约束求解和工程图理解的原始论文、数据集与开源实现。",
                    "核验各方案导出的 STEP、DXF、DWG、B-Rep 或原生特征树是否能在目标 CAD 软件中继续编辑。",
                    "收集自动标注、投影视图和 GD&T 校验的评测集与人工复核指标，区分演示图与可发布工程图。",
                ],
                landing_suggestions=[
                    "先选取规则明确的零件族，建立草图、参数表、三维模型和二维工程图的成对基准集。",
                    "将 AI 输出限定为候选特征、视图和标注建议，由 CAD 几何内核执行约束求解、干涉检查和导出校验。",
                    "将图纸版本、尺寸变更、公差规则和审批结果写入可追溯数据模型，避免二维图与三维源模型脱节。",
                ],
            )
        return BriefingSynthesis(
            search_content_summary=(
                f"本轮围绕“{topic}”保留了 {source_count} 条公开线索，技术讨论主要聚集在{theme_names}。"
                "这些材料不是孤立新闻：它们共同描述了 AI 从单点识别或助手功能，走向接入工程数据、业务系统和现场闭环的过程。"
            ),
            short_summary=(
                f"这一批内容的共同主题是：{topic} 正从概念展示转向具体工作流。"
                "最值得关注的是系统如何连接领域数据、约束、工具调用和人工复核，而不是单一模型名称。"
            ),
            detailed_summary=self._fallback_detailed_summary(themes, key_signal_interpretation=(
                "公开材料反复出现的不是单一模型名称，而是领域数据进入模型、模型调用受控工具、"
                "结果写回业务或工程系统并保留人工接管的闭环；没有这些环节的内容只能视为早期线索。"
            )),
            themes=themes,
            key_signal_interpretation=(
                "多条线索同时出现时，应优先看它们是否共享同一数据闭环：输入来自何处、模型或规则如何做判断、"
                "结果如何写回业务/工程系统、以及失败后由谁接管。能回答这四个问题的项目，才可能从演示走向可运营能力。"
            ),
            analysis_judgment=(
                "跨来源归纳看，下一阶段的竞争不只是模型能力，而是行业语义、系统接口、评估指标和责任边界的产品化。"
                "因此应把“是否有 Agent/大模型”降为筛选条件，把“是否形成可审计的闭环与量化收益”升为排序条件。"
            ),
            next_search_directions=[
                "继续追踪每个主题的原始论文、产品技术文档、代码仓库和公开架构图。",
                "补齐数据输入、系统接口、评估指标和人工接管机制，验证是否形成可运营闭环。",
                "对重复出现的项目追踪后续版本、客户案例和部署限制，区分演示与规模化落地。",
            ],
            landing_suggestions=[
                "先选一个边界清晰、可量化的流程建立数据和评估基线。",
                "将模型输出限定为建议、工单或仿真方案，保留审批、回滚和完整审计记录。",
                "优先建设来源、实体、指标和项目之间的关联库，让后续日报可做连续对比。",
            ],
        )

    def _fallback_theme_analysis(
        self,
        evidence: list[CollectedSource],
        technology: str,
        importance: str,
        maturity: str,
    ) -> str:
        evidence_summary = self._evidence_summary(evidence)
        workflow = self._evidence_workflow(evidence)
        return (
            f"已有材料显示：{evidence_summary}。可从中还原的实现路径是{workflow}；"
            f"其中真正承担工程能力的部分是{technology}。这会影响{importance}。"
            f"当前应按“{maturity}”理解，不把公开线索直接当作已验证的规模化能力。"
        )

    @staticmethod
    def _fallback_detailed_summary(
        themes: list[BriefingTheme],
        *,
        key_signal_interpretation: str,
    ) -> str:
        """Render a readable fallback without resurrecting the old field checklist."""
        blocks: list[str] = []
        for theme in themes:
            body = theme.analysis.strip()
            if not body:
                details = [
                    theme.what_is_happening.strip(),
                    theme.core_technology.strip(),
                    theme.data_and_workflow.strip(),
                    theme.why_it_matters.strip(),
                    theme.maturity.strip(),
                ]
                body = " ".join(part for part in details if part)
            if body:
                blocks.extend([f"### {theme.name}", body, ""])
        if key_signal_interpretation.strip():
            blocks.extend(["### 仍需核验的技术分界线", key_signal_interpretation.strip()])
        return "\n".join(blocks).strip()

    def render_markdown(self, briefing: DailyBriefing) -> str:
        synthesis = briefing.synthesis
        lines = [
            "# 内容搜集报告",
            "",
            f"- 选题：{briefing.topic}",
            f"- 日期：{briefing.run_date.isoformat()}",
            f"- 原始线索：{len(briefing.sources)} 条",
            "",
            "## 搜索方向",
            *[f"- {direction}" for direction in briefing.search_directions],
            "",
            "## 关键词",
            ", ".join(briefing.keywords),
            "",
            "## 搜索内容总结",
            synthesis.search_content_summary,
            "",
            "## 简短总结",
            synthesis.short_summary,
            "",
            "## 详细总结",
            "",
        ]
        detailed_summary = synthesis.detailed_summary.strip() or self._fallback_detailed_summary(
            synthesis.themes,
            key_signal_interpretation=synthesis.key_signal_interpretation,
        )
        lines.extend(detailed_summary.splitlines())
        lines.extend(
            [
                "",
                "## AI 分析判断",
                synthesis.analysis_judgment,
                "",
                "## 下一步搜索方向",
                *[f"- {direction}" for direction in synthesis.next_search_directions],
                "",
                "## 落地建议",
                *[f"- {suggestion}" for suggestion in synthesis.landing_suggestions],
                "",
                "## 原文链接",
            ]
        )
        for index, source in enumerate(briefing.sources, start=1):
            lines.append(f"{index}. [{source.title}]({source.url}) | {source.platform} | 重要度 {source.importance_score:.1f}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _normalize_url(url: str) -> str:
        value = url.strip()
        if not value.startswith(("https://", "http://")):
            return ""
        return value.rstrip("/")

    def _briefing_date(self) -> date:
        try:
            return datetime.now(ZoneInfo(self.timezone_name)).date()
        except ZoneInfoNotFoundError:
            return datetime.now(UTC).date()

    @staticmethod
    def _compact_source_text(value: str, limit: int) -> str:
        return " ".join(str(value or "").split())[:limit]

    @staticmethod
    def _is_search_page_dump(title: str, snippet: str) -> bool:
        text = f"{title} {snippet}".lower()
        return any(marker in text for marker in _SEARCH_DUMP_MARKERS)

    @staticmethod
    def _platform_from_url(url: str, fallback: str) -> str:
        host = urlparse(url).netloc.lower()
        mappings = {
            "bilibili.com": "哔哩哔哩",
            "douyin.com": "抖音",
            "mp.weixin.qq.com": "微信公众号",
            "toutiao.com": "今日头条",
            "xiaohongshu.com": "小红书",
            "linkedin.com": "LinkedIn",
            "reddit.com": "Reddit",
            "youtube.com": "YouTube",
            "x.com": "X",
            "zhihu.com": "知乎",
        }
        return next((name for domain, name in mappings.items() if domain in host), fallback)

    @staticmethod
    def _keywords(
        topic: str,
        feedback: list[dict[str, object]],
        search_plan: list[tuple[str, str]],
    ) -> list[str]:
        candidates = [topic]
        candidates.extend(
            query
            for platform, query in search_plan
            if platform == "技术路线" and _is_meaningful_query(query)
        )
        for item in feedback[:3]:
            note = " ".join(str(item.get("body") or "").split())
            if _is_meaningful_feedback(note):
                candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9+._/-]{1,}|[\u4e00-\u9fff]{2,}", note))
        keywords: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip()
            if normalized and _is_meaningful_query(normalized) and normalized not in keywords:
                keywords.append(normalized)
            if len(keywords) >= 12:
                break
        return keywords or [topic]

    @staticmethod
    def _relevance_score(topic: str, title: str, snippet: str) -> float:
        terms = [term.lower() for term in re.findall(r"[A-Za-z0-9+._/-]{2,}|[\u4e00-\u9fff]{2,}", topic)]
        haystack = f"{title} {snippet}".lower()
        return float(sum(1 for term in terms if term in haystack))

    @staticmethod
    def _query_relevance_score(query: str, title: str, snippet: str) -> float:
        terms = [
            term.lower()
            for term in re.findall(r"[A-Za-z][A-Za-z0-9+._/-]{1,}|[\u4e00-\u9fff]{2,}", query)
            if term.lower() not in _QUERY_STOPWORDS
        ]
        haystack = f"{title} {snippet}".lower()
        return float(sum(1 for term in dict.fromkeys(terms) if term in haystack))

    @staticmethod
    def _matches_technical_marker(text: str, marker: str) -> bool:
        if marker.isascii() and re.fullmatch(r"[a-z0-9]+", marker):
            return bool(re.search(rf"\b{re.escape(marker)}\b", text))
        return marker in text

    @staticmethod
    def _is_generic_reference_domain(url: str) -> bool:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        return any(host == domain or host.endswith(f".{domain}") for domain in _GENERIC_REFERENCE_DOMAINS)

    @classmethod
    def _technical_signal_count(cls, title: str, snippet: str) -> int:
        text = f"{title} {snippet}".lower()
        theme_hits = sum(
            1
            for _, markers, *_ in CAD_THEMES + THEMES
            if any(cls._matches_technical_marker(text, marker) for marker in markers if marker != "\u5de5\u7a0b")
        )
        explicit_hits = sum(cls._matches_technical_marker(text, marker) for marker in _TECHNICAL_SIGNAL_MARKERS)
        return theme_hits + explicit_hits

    @staticmethod
    def _requires_industrial_anchor(topic: str, query: str) -> bool:
        text = f"{topic} {query}".lower()
        return any(marker in text for marker in ("industrial", "manufactur", "\u5de5\u4e1a", "\u5236\u9020"))

    @staticmethod
    def _has_industrial_anchor(title: str, snippet: str) -> bool:
        text = f"{title} {snippet}".lower()
        return any(marker in text for marker in _INDUSTRIAL_ANCHORS) or any(
            re.search(rf"\b{re.escape(marker)}\b", text) for marker in _INDUSTRIAL_ACRONYM_ANCHORS
        )

    @classmethod
    def _is_report_source_candidate(
        cls,
        topic: str,
        query: str,
        url: str,
        title: str,
        snippet: str,
    ) -> bool:
        if cls._is_generic_reference_domain(url):
            return False
        host = urlparse(url).netloc.lower().removeprefix("www.")
        path = urlparse(url).path.lower()
        if host in {"baidu.com", "m.baidu.com"} or any(marker in path for marker in _LOGIN_PATH_MARKERS):
            return False
        text = f"{title} {snippet}".lower()
        if cls._is_search_page_dump(title, snippet):
            return False
        if _is_cad_topic(f"{topic} {query}") and any(marker in text for marker in _CAD_MEDICAL_MARKERS):
            return False
        if _is_cad_topic(f"{topic} {query}") and not _has_cad_anchor(title, snippet):
            return False
        if cls._requires_industrial_anchor(topic, query) and not cls._has_industrial_anchor(title, snippet):
            return False
        topic_score = cls._relevance_score(topic, title, snippet)
        query_score = cls._query_relevance_score(query, title, snippet)
        if _is_cad_topic(f"{topic} {query}"):
            return topic_score >= 2 or query_score >= 2 or (
                cls._technical_signal_count(title, snippet) > 0 and max(topic_score, query_score) >= 1
            )
        return topic_score >= 2 or query_score >= 2 or cls._technical_signal_count(title, snippet) > 0

    @classmethod
    def _source_rejection_reason(
        cls,
        topic: str,
        query: str,
        url: str,
        title: str,
        snippet: str,
    ) -> tuple[str, str] | None:
        if not url:
            return "rejected_invalid_url", "missing or non-http URL"
        if not title and not snippet:
            return "rejected_empty_content", "title and snippet are both empty"
        if cls._is_search_page_dump(title, snippet):
            return "rejected_search_page_dump", "result looked like a search-result page dump"
        if cls._is_generic_reference_domain(url):
            return "rejected_generic_reference", "generic reference domain is not strong briefing evidence"
        parsed = urlparse(url)
        host = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.lower()
        if host in {"baidu.com", "m.baidu.com"}:
            return "rejected_search_page_dump", "Baidu wrapper URL is not an original public source"
        if any(marker in path for marker in _LOGIN_PATH_MARKERS):
            return "rejected_login_page", "login-gated URL path is outside public-source boundary"
        text = f"{title} {snippet}".lower()
        if _is_cad_topic(f"{topic} {query}") and any(marker in text for marker in _CAD_MEDICAL_MARKERS):
            return "rejected_cad_medical", "CAD query matched medical CAD acronym content"
        if _is_cad_topic(f"{topic} {query}") and not _has_cad_anchor(title, snippet):
            return "rejected_cad_missing_anchor", "CAD query lacked engineering CAD anchors"
        if cls._requires_industrial_anchor(topic, query) and not cls._has_industrial_anchor(title, snippet):
            return "rejected_missing_industrial_anchor", "industrial query lacked manufacturing or industrial anchors"
        if not cls._is_report_source_candidate(topic, query, url, title, snippet):
            return "rejected_low_relevance", "insufficient topic, query, or technical signal"
        return None

    def _source_candidate(
        self,
        *,
        subscription: TopicSubscription,
        requested_platform: str,
        query: str,
        result: SearchResult,
        normalized_url: str,
        title: str,
        snippet: str,
        status: str,
        reason: str,
        relevance_score: float,
        importance_score: float,
    ) -> SourceCandidate:
        return SourceCandidate(
            id=f"cand_{uuid.uuid4().hex}",
            topic_id=subscription.id,
            user_id=subscription.user_id,
            title=title,
            url=normalized_url,
            snippet=snippet,
            platform=self._platform_from_url(normalized_url, requested_platform) if normalized_url else requested_platform,
            provider=result.provider,
            query=query,
            status=status,  # type: ignore[arg-type]
            reason=reason,
            relevance_score=relevance_score,
            importance_score=importance_score,
            retrieved_at=result.checked_at,
            created_at=datetime.now(UTC).isoformat(),
        )

    def _importance_score(self, topic: str, query: str, title: str, snippet: str, provider: str) -> float:
        text = f"{title} {snippet}".lower()
        technical_hits = self._technical_signal_count(title, snippet)
        cad_hits = sum(self._matches_technical_marker(text, marker) for marker in _CAD_EVIDENCE_ANCHORS)
        evidence_hits = sum(marker in text for marker in ("paper", "论文", "official", "官方", "release", "发布", "benchmark", "案例", "架构"))
        provider_bonus = 2.0 if provider.startswith("direct-") else 1.0 if provider.startswith("mcp:") else 0.0
        return (
            self._relevance_score(topic, title, snippet) * 2
            + self._query_relevance_score(query, title, snippet) * 2
            + technical_hits * 3
            + cad_hits * 2
            + evidence_hits
            + provider_bonus
        )

    @staticmethod
    def _evidence_workflow(sources: list[CollectedSource]) -> str:
        excerpts = [" ".join(source.snippet.split()) for source in sources if source.snippet]
        if not excerpts:
            return "来源未披露完整数据闭环；下一步应补查输入数据、系统接口、执行动作和人工接管点。"
        return (
            "从公开描述可见的流程线索是："
            + "；".join(excerpts[:2])[:900]
            + "。仍应核对数据来源、决策/规则层、写回系统与异常接管是否完整闭环。"
        )

    @staticmethod
    def _evidence_summary(sources: list[CollectedSource]) -> str:
        titles = "；".join(source.title for source in sources[:3])
        return f"相关线索集中讨论：{titles}。"
