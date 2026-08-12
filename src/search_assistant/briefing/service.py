from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from datetime import UTC, date, datetime
from pathlib import Path
import re
import time
import uuid
from typing import Protocol
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from search_assistant.contracts import BriefingSynthesis, BriefingTheme, CollectedSource, DailyBriefing, TopicSubscription
from search_assistant.evolution.service import DomainKnowledgeCandidateService
from search_assistant.memory.store import MemoryStore
from search_assistant.search.provider import SearchClient, SearchResult


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


GENERIC_EVIDENCE_THEMES: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    (
        "政策、规模与产业链信号",
        (
            "政策", "规划", "行动计划", "产业链", "规模", "报告", "白皮书", "全景分析",
            "万亿", "亿元", "市场", "产业", "发布", "大会", "waic", "naai",
        ),
        "把政策文件、产业规模、产业链梳理和大会/报告类材料视为环境信号，用来判断资金、监管、供给链和需求侧正在如何给技术落地设定边界。",
        "它能说明行业为什么加速投入、哪些环节被反复强调，但不能直接证明某个系统已经形成可复制的工程闭环。",
        "宏观信号阶段；需要继续核验原始政策文本、统计口径、企业案例和技术指标之间是否相互支撑。",
    ),
    (
        "技术底座、数据与开源生态",
        (
            "dataset", "benchmark", "github", "hugging face", "modelscope", "open source", "paper",
            "repository", "数据集", "评测", "基准", "开源", "论文", "模型", "框架", "算力",
            "api", "数据", "架构",
        ),
        "把数据集、评测基准、模型/框架、开源仓库和工程接口视为技术底座，用来判断产业发展是否具备可复现、可评估和可集成的基础。",
        "这类来源决定日报能不能从口号进入实现层：是否有可获得的数据、可比较的指标、可复用的软件组件和清晰的系统接口。",
        "技术线索阶段；需要补查原始论文、模型卡、仓库 README、benchmark 定义和真实部署约束。",
    ),
    (
        "应用落地与业务转型案例",
        (
            "应用", "落地", "实践", "案例", "场景", "解决方案", "客户", "部署", "转型",
            "升级", "生产", "制造", "医疗", "教育", "金融", "政务", "工业", "工作流",
        ),
        "把行业案例和业务流程描述还原为输入数据、模型或规则处理、系统接口、人工复核和结果写回的工作流假设。",
        "它能帮助识别哪些环节已经从演示进入业务流程，也能暴露公开材料尚未说明的审批、回滚、异常接管和量化收益。",
        "案例线索阶段；没有现场指标、接口说明和失败处理机制前，不应把宣传材料直接视为规模化能力。",
    ),
    (
        "教育传播、公众讨论与弱证据线索",
        (
            "教程", "课程", "纪录片", "科普", "入门", "讲完", "学习", "培训", "视频",
            "论坛", "人山人海", "观点", "讨论", "评论", "bilibili", "youtube",
        ),
        "把课程、纪录片、公开视频和公众讨论作为关注度与知识扩散信号，而不是直接作为技术实现证据。",
        "这类来源能说明话题热度、人才供给和概念传播，但通常缺少系统架构、数据接口、评测指标和部署边界。",
        "弱证据阶段；适合引出下一轮检索方向，不适合单独支撑技术结论。",
    ),
    (
        "综合产业动态与待核验证据",
        (),
        "把无法归入明确技术或政策类别的公开材料先作为综合线索保留，再等待原始技术文档、案例指标或代码材料校验。",
        "它避免遗漏潜在线索，但在证据等级上低于论文、产品文档、开源仓库和带指标的落地案例。",
        "待核验线索阶段；下一步应回到原始来源、技术材料和可量化结果。",
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
_MARKETING_CONTENT_DOMAINS = {
    "mp.weixin.qq.com",
    "weixin.qq.com",
    "toutiao.com",
    "xiaohongshu.com",
    "xhslink.com",
    "baijiahao.baidu.com",
    "sohu.com",
}
_MARKETING_STRONG_PHRASES = (
    "广告合作",
    "本文为广告",
    "商业推广",
    "商务合作",
    "商业合作",
    "软文",
    "赞助商",
    "招商加盟",
    "代理加盟",
    "限时优惠",
    "限时福利",
    "免费领取",
    "扫码领取",
    "扫码加",
    "加微信",
    "联系微信",
    "私信领取",
    "私信进群",
    "领取资料",
    "点击购买",
    "购买链接",
    "优惠券",
    "训练营",
    "报名入口",
    "课程报名",
    "带货",
    "引流",
    "裂变",
    "涨粉",
    "变现",
    "advertorial",
    "sponsored",
    "business cooperation",
    "commercial cooperation",
    "scan qr",
    "add wechat",
    "limited offer",
    "limited discount",
    "coupon",
    "buy now",
    "training camp",
    "course enrollment",
)
_MARKETING_WEAK_PHRASES = (
    "重磅福利",
    "速看",
    "震惊",
    "必看",
    "看完就懂",
    "一文看懂",
    "保姆级",
    "干货满满",
    "建议收藏",
    "点赞关注",
    "关注不迷路",
    "转发收藏",
    "评论区",
    "错过再等一年",
    "全网最全",
    "你还不知道",
    "must read",
    "save this",
    "follow us",
)
_SOURCE_EVIDENCE_PHRASES = (
    "paper",
    "论文",
    "arxiv",
    "github",
    "benchmark",
    "基准",
    "dataset",
    "数据集",
    "white paper",
    "official",
    "官方",
    "release",
    "源码",
    "architecture",
    "架构",
    "evaluation",
    "评测",
    "case study",
    "案例",
    "接口",
    "模型",
    "系统",
    "算法",
    "部署",
    "开源",
)


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


def _has_substantive_detailed_summary(value: str) -> bool:
    text = value.strip()
    if len(text) < 280:
        return False
    heading_count = len(re.findall(r"^###\s+\S", text, flags=re.MULTILINE))
    if heading_count >= 2:
        return True
    return heading_count == 1 and len(text) >= 500


def _is_substantive_synthesis(synthesis: BriefingSynthesis) -> bool:
    if not synthesis.themes or len(synthesis.analysis_judgment.strip()) < 60:
        return False

    # New model route: the detailed narrative owns the shape of the analysis.
    # Themes are evidence anchors, not a mandatory per-theme form.
    if synthesis.detailed_summary.strip():
        return (
            len(synthesis.short_summary.strip()) >= 75
            and _has_substantive_detailed_summary(synthesis.detailed_summary)
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


def _duration_ms(started: float) -> float:
    return round((time.monotonic() - started) * 1000, 3)


def _count_knowledge_layers(validation_results: list[dict[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in validation_results:
        layer = str(item.get("knowledge_layer") or "unreviewed_candidate")
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def _safe_trace_error(exc: Exception) -> str:
    text = str(exc)
    for marker in ("api_key", "API key", "Authorization", "token", "secret", "password"):
        text = text.replace(marker, "[redacted]")
    return text[:500]


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
        run_id = f"brief_run_{uuid.uuid4().hex}"
        run_started = time.monotonic()
        try:
            phase_started = time.monotonic()
            subscription = self.store.upsert_topic(user_id, chat_id, topic)
            self._ensure_project_state()
            feedback = self.store.list_topic_feedback(subscription.id)
            search_plan = self.build_search_plan(subscription, feedback)
            self._record_trace_event(
                run_id,
                "briefing",
                "plan_queries",
                phase_started,
                metadata={
                    "topic": subscription.topic,
                    "query_count": len(search_plan),
                    "feedback_count": len(feedback),
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "planned",
                payload={
                    "topic_id": subscription.id,
                    "query_count": len(search_plan),
                    "feedback_count": len(feedback),
                    "queries": [query for _, query in search_plan[:8]],
                },
            )

            phase_started = time.monotonic()
            sources = self._collect_sources(subscription, search_plan, feedback, run_id=run_id)
            ranked_sources = sorted(sources, key=lambda source: source.importance_score, reverse=True)[
                : self.max_sources
            ]
            self._record_trace_event(
                run_id,
                "briefing",
                "collect_sources",
                phase_started,
                metadata={
                    "topic": subscription.topic,
                    "raw_source_count": len(sources),
                    "ranked_source_count": len(ranked_sources),
                    "max_sources": self.max_sources,
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "sources_collected",
                payload={
                    "raw_source_count": len(sources),
                    "ranked_source_count": len(ranked_sources),
                    "source_urls": [source.url for source in ranked_sources[:10]],
                },
            )

            phase_started = time.monotonic()
            synthesis = self._synthesize(
                subscription.topic,
                ranked_sources[: self.model_max_sources],
                search_plan,
                fallback_sources=ranked_sources,
            )
            self._record_trace_event(
                run_id,
                "briefing",
                "synthesize_report",
                phase_started,
                metadata={
                    "topic": subscription.topic,
                    "model_source_count": min(len(ranked_sources), self.model_max_sources),
                    "theme_count": len(synthesis.themes),
                    "used_runtime": self.runtime is not None,
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "synthesized",
                payload={
                    "theme_count": len(synthesis.themes),
                    "model_source_count": min(len(ranked_sources), self.model_max_sources),
                    "used_runtime": self.runtime is not None,
                },
            )

            resolved_date = run_date or self._briefing_date()
            briefing = DailyBriefing(
                id=f"brief_{uuid.uuid4().hex}",
                topic_id=subscription.id,
                user_id=user_id,
                chat_id=chat_id,
                topic=subscription.topic,
                run_date=resolved_date,
                search_directions=[f"{platform}: {query}" for platform, query in search_plan],
                keywords=self._keywords(subscription.topic, feedback, search_plan),
                sources=ranked_sources,
                synthesis=synthesis,
                markdown="",
                created_at=datetime.now(UTC).isoformat(),
            )
            briefing.markdown = self.render_markdown(briefing)
            self.store.record_daily_briefing(briefing)

            phase_started = time.monotonic()
            candidate_service = DomainKnowledgeCandidateService(self.store)
            candidate_ids = candidate_service.capture_briefing(briefing)
            validation_results = [candidate_service.record_validation_gate(candidate_id) for candidate_id in candidate_ids]
            knowledge_layers = _count_knowledge_layers(validation_results)
            self._record_trace_event(
                run_id,
                "evolution",
                "capture_domain_knowledge_candidates",
                phase_started,
                metadata={
                    "topic": subscription.topic,
                    "candidate_count": len(candidate_ids),
                    "validation_gate_passed": sum(1 for item in validation_results if item["validated"]),
                    "validation_gate_failed": sum(1 for item in validation_results if not item["validated"]),
                    "knowledge_layers": knowledge_layers,
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "candidates_captured",
                payload={
                    "briefing_id": briefing.id,
                    "candidate_ids": candidate_ids,
                    "validation_gate_passed": sum(1 for item in validation_results if item["validated"]),
                    "validation_gate_failed": sum(1 for item in validation_results if not item["validated"]),
                    "knowledge_layers": knowledge_layers,
                },
            )
            self.store.add_project_ledger_entry(
                entry_type="briefing_run",
                subject=subscription.topic,
                status="completed",
                summary=f"Generated daily briefing with {len(ranked_sources)} retained sources.",
                evidence_refs=[
                    f"briefing:{briefing.id}",
                    *[source.url for source in ranked_sources[:8]],
                    *[f"candidate:{candidate_id}" for candidate_id in candidate_ids[:8]],
                ],
                risk="Public search coverage can be incomplete or blocked; the report should be read with source URLs.",
                rollback="Use an isolated --data-dir for tests, or delete the generated briefing/data directory.",
                metadata={
                    "run_id": run_id,
                    "topic_id": subscription.id,
                    "query_count": len(search_plan),
                    "source_count": len(ranked_sources),
                    "candidate_count": len(candidate_ids),
                    "validation_gate_passed": sum(1 for item in validation_results if item["validated"]),
                    "validation_gate_failed": sum(1 for item in validation_results if not item["validated"]),
                    "knowledge_layers": knowledge_layers,
                    "run_date": briefing.run_date.isoformat(),
                },
            )
            self._record_trace_event(
                run_id,
                "briefing",
                "run",
                run_started,
                metadata={
                    "topic": subscription.topic,
                    "status": "completed",
                    "query_count": len(search_plan),
                    "source_count": len(ranked_sources),
                },
            )
            self._record_checkpoint(
                run_id,
                subscription.topic,
                "completed",
                payload={
                    "briefing_id": briefing.id,
                    "source_count": len(ranked_sources),
                    "candidate_count": len(candidate_ids),
                },
            )
            return briefing
        except Exception as exc:
            self._record_trace_event(
                run_id,
                "briefing",
                "run",
                run_started,
                status="failed",
                metadata={"topic": topic},
                error=_safe_trace_error(exc),
            )
            self._record_checkpoint(
                run_id,
                topic,
                "failed",
                status="failed",
                payload={"error": _safe_trace_error(exc)},
            )
            raise

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
            if len(unique) >= self.max_queries:
                break
        return unique

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
        return self.store.add_topic_feedback(subscription.id, user_id, chat_id, enriched_body, source_url)

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
        run_id: str | None = None,
    ) -> list[CollectedSource]:
        by_url: dict[str, CollectedSource] = {}
        query_results: list[tuple[str, str, list[SearchResult]]] = []
        workers = min(6, len(search_plan))
        executor = ThreadPoolExecutor(max_workers=max(1, workers))
        futures = [
            (requested_platform, query, time.monotonic(), executor.submit(self._search, query))
            for requested_platform, query in search_plan
        ]
        try:
            completed, pending = wait(
                [future for _, _, _, future in futures],
                timeout=self.search_budget_seconds,
            )
            for requested_platform, query, started, future in futures:
                if future not in completed:
                    self._record_provider_health(
                        run_id,
                        requested_platform,
                        query,
                        [],
                        _duration_ms(started),
                        ok=False,
                        error="search budget expired before this query completed",
                    )
                    continue
                try:
                    results = future.result()
                except Exception as exc:
                    results = []
                    self._record_provider_health(
                        run_id,
                        requested_platform,
                        query,
                        results,
                        _duration_ms(started),
                        ok=False,
                        error=_safe_trace_error(exc),
                    )
                else:
                    self._record_provider_health(
                        run_id,
                        requested_platform,
                        query,
                        results,
                        _duration_ms(started),
                        ok=True,
                    )
                query_results.append((requested_platform, query, results))
            for future in pending:
                future.cancel()
        finally:
            # Do not block the daily briefing on slow public endpoints. Completed
            # results are retained; queued calls are cancelled when possible.
            executor.shutdown(wait=False, cancel_futures=True)

        for requested_platform, query, results in query_results:
            for result in results:
                normalized_url = self._normalize_url(result.url)
                if self._is_search_page_dump(result.title, result.snippet):
                    continue
                title = self._compact_source_text(result.title, 240)
                snippet = self._compact_source_text(result.snippet, 900)
                if not normalized_url or not self._is_report_source_candidate(
                    subscription.topic,
                    query,
                    normalized_url,
                    title,
                    snippet,
                ):
                    continue
                if not title and not snippet:
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
                    relevance_score=max(
                        self._relevance_score(subscription.topic, title, snippet),
                        self._query_relevance_score(query, title, snippet),
                    ),
                    importance_score=self._importance_score(
                        subscription.topic,
                        query,
                        title,
                        snippet,
                        result.provider,
                    ),
                    retrieved_at=result.checked_at,
                )
                prior = by_url.get(normalized_url)
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

        persisted = [self.store.upsert_collected_source(source) for source in by_url.values()]
        return persisted

    def _search(self, query: str) -> list[SearchResult]:
        return self.search_client.search(query, limit=self.results_per_query)

    def _record_trace_event(
        self,
        run_id: str,
        event_type: str,
        name: str,
        started: float,
        status: str = "completed",
        metadata: dict[str, object] | None = None,
        error: str | None = None,
    ) -> None:
        try:
            self.store.add_trace_event(
                run_id=run_id,
                event_type=event_type,
                name=name,
                status=status,
                duration_ms=_duration_ms(started),
                metadata=metadata or {},
                error=error,
            )
        except Exception:
            # Observability should never make a briefing fail.
            return

    def _record_checkpoint(
        self,
        run_id: str,
        subject: str,
        step: str,
        status: str = "completed",
        payload: dict[str, object] | None = None,
    ) -> None:
        try:
            self.store.add_run_checkpoint(
                run_id=run_id,
                workflow="daily_briefing",
                subject=subject,
                step=step,
                status=status,
                payload=payload or {},
            )
        except Exception:
            # Checkpoint metadata should never make a briefing fail.
            return

    def _ensure_project_state(self) -> None:
        try:
            self.store.ensure_project_ledger_snapshot(
                project_id="pydantic-ai-tech-intel-briefing",
                objective="Generate source-backed technology intelligence briefings with reviewable self-evolution.",
                phase="agentops-readiness",
                status="active",
                next_decision="Run eval replay, then release domain knowledge only after eval and human review gates pass.",
                open_blockers=[],
                constraints={
                    "knowledge_release": "candidate-only until eval gate and human review pass",
                    "search": "public source coverage can be incomplete or blocked",
                    "privacy": "local configuration and secrets must not be persisted in audit logs",
                },
                decisions=[
                    {
                        "id": "agentops-p1-gates",
                        "decision": "Record automatic evidence gates, checkpoints, provider health, and release ledger.",
                    }
                ],
                evidence_refs=["README.md", "docs/development.md"],
                version="1",
            )
        except Exception:
            return

    def _record_provider_health(
        self,
        run_id: str | None,
        requested_platform: str,
        query: str,
        results: list[SearchResult],
        duration_ms: float,
        ok: bool,
        error: str | None = None,
    ) -> None:
        if run_id is None:
            return
        providers = sorted({result.provider for result in results if result.provider})
        provider_label = ", ".join(providers[:3]) if providers else self.search_client.__class__.__name__
        if len(providers) > 3:
            provider_label = f"{provider_label}, +{len(providers) - 3} more"
        try:
            self.store.record_search_provider_health(
                run_id=run_id,
                requested_platform=requested_platform,
                provider=provider_label,
                query=query,
                ok=ok,
                result_count=len(results),
                duration_ms=duration_ms,
                error=error,
            )
        except Exception:
            return

    def _synthesize(
        self,
        topic: str,
        sources: list[CollectedSource],
        search_plan: list[tuple[str, str]],
        fallback_sources: list[CollectedSource] | None = None,
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
        return self._fallback_synthesis(topic, fallback_sources or sources)

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
        is_cad = _is_cad_topic(topic)
        for source in sources:
            haystack = f"{source.title} {source.snippet}".lower()
            matched = False
            for name, markers, _, _, _ in theme_catalog:
                if any(self._matches_technical_marker(haystack, marker) for marker in markers):
                    themed_sources.setdefault(name, []).append(source)
                    matched = True
                    break
            if not matched:
                if is_cad:
                    themed_sources.setdefault("工程图与 CAD 工具链线索", []).append(source)
                else:
                    themed_sources.setdefault(self._generic_evidence_theme_name(source), []).append(source)

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
        if is_cad:
            residual = themed_sources.get("工程图与 CAD 工具链线索", [])
            if residual:
                themes.append(
                    BriefingTheme(
                        name="工程图与 CAD 工具链线索",
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
        else:
            for name, _, technology, importance, maturity in GENERIC_EVIDENCE_THEMES:
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
                f"本轮围绕“{topic}”保留了 {source_count} 条公开线索，可分为{theme_names}。"
                "这些材料需要分层阅读：政策、规模和活动类内容提供产业环境信号；数据、开源、评测和案例材料才更接近实现证据。"
            ),
            short_summary=(
                f"本轮关于“{topic}”的有效阅读方式，是先把来源拆成宏观产业信号、技术底座、应用场景和弱证据四层。"
                "只有能说明数据输入、模型或工具链、系统接口、评估指标与人工接管机制的材料，才足以支撑技术落地判断。"
            ),
            detailed_summary=self._fallback_detailed_summary(themes, key_signal_interpretation=(
                "公开材料的证据强度并不相同：报告、政策和视频能说明关注度与投入方向，"
                "但是否真正形成产业能力，还要回到原始技术资料、数据集/评测、接口说明、客户案例和可复现指标。"
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

    def _generic_evidence_theme_name(self, source: CollectedSource) -> str:
        text = f"{source.title} {source.snippet} {source.platform} {source.provider}".lower()
        best_name = "综合产业动态与待核验证据"
        best_score = 0
        for name, markers, *_ in GENERIC_EVIDENCE_THEMES:
            if not markers:
                continue
            score = sum(1 for marker in markers if self._matches_technical_marker(text, marker.lower()))
            if score > best_score:
                best_name = name
                best_score = score
        return best_name

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
            f"{evidence_summary}\n\n"
            f"能够支撑的工作假设是：{technology}{workflow}\n\n"
            f"工程含义是：{importance}证据边界是：{maturity} "
            "因此不应把公开线索直接写成已验证的规模化能力。"
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
    def _has_exact_cjk_topic_phrase(topic: str, title: str, snippet: str) -> bool:
        phrases = re.findall(r"[\u4e00-\u9fff]{4,}", topic)
        if not phrases:
            return False
        haystack = f"{title} {snippet}"
        return any(phrase in haystack for phrase in phrases)

    @staticmethod
    def _cjk_topic_match_score(topic: str, title: str, snippet: str) -> int:
        haystack = f"{title} {snippet}"
        best_score = 0
        for phrase in re.findall(r"[\u4e00-\u9fff]{3,}", topic):
            if phrase in haystack:
                best_score = max(best_score, 4)
                continue
            chunks = {phrase[index : index + 2] for index in range(0, len(phrase) - 1)}
            matched_chunks = {chunk for chunk in chunks if chunk in haystack}
            best_score = max(best_score, len(matched_chunks))
        return best_score

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

    @staticmethod
    def _is_marketing_domain(url: str) -> bool:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        return any(host == domain or host.endswith(f".{domain}") for domain in _MARKETING_CONTENT_DOMAINS)

    @classmethod
    def _is_marketing_account_content(cls, url: str, title: str, snippet: str) -> bool:
        text = f"{title} {snippet}".lower()
        compact_text = re.sub(r"\s+", "", text)
        strong_hits = sum(
            1 for phrase in _MARKETING_STRONG_PHRASES if phrase in text or phrase in compact_text
        )
        weak_hits = sum(1 for phrase in _MARKETING_WEAK_PHRASES if phrase in text or phrase in compact_text)
        evidence_hits = sum(1 for phrase in _SOURCE_EVIDENCE_PHRASES if phrase in text or phrase in compact_text)
        technical_hits = cls._technical_signal_count(title, snippet)
        on_marketing_prone_domain = cls._is_marketing_domain(url)

        if strong_hits >= 2:
            return True
        if on_marketing_prone_domain and strong_hits >= 1 and weak_hits >= 1:
            return True
        if on_marketing_prone_domain and strong_hits >= 1 and evidence_hits == 0 and technical_hits == 0:
            return True
        if weak_hits >= 3 and evidence_hits == 0 and technical_hits == 0:
            return True
        return False

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
        if cls._is_marketing_account_content(url, title, snippet):
            return False
        if _is_cad_topic(f"{topic} {query}") and any(marker in text for marker in _CAD_MEDICAL_MARKERS):
            return False
        if _is_cad_topic(f"{topic} {query}") and not _has_cad_anchor(title, snippet):
            return False
        if cls._requires_industrial_anchor(topic, query) and not cls._has_industrial_anchor(title, snippet):
            return False
        topic_score = cls._relevance_score(topic, title, snippet)
        query_score = cls._query_relevance_score(query, title, snippet)
        if cls._has_exact_cjk_topic_phrase(topic, title, snippet):
            return True
        if cls._cjk_topic_match_score(topic, title, snippet) >= 2:
            return True
        if _is_cad_topic(f"{topic} {query}"):
            return topic_score >= 2 or query_score >= 2 or (
                cls._technical_signal_count(title, snippet) > 0 and max(topic_score, query_score) >= 1
            )
        return topic_score >= 2 or query_score >= 2 or cls._technical_signal_count(title, snippet) > 0

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
        excerpts = [
            DailyBriefingService._short_evidence_excerpt(source.snippet)
            for source in sources
            if source.snippet
        ]
        excerpts = [excerpt for excerpt in excerpts if excerpt]
        if not excerpts:
            return " 来源未披露完整数据闭环；下一步应补查输入数据、系统接口、执行动作和人工接管点。"
        return (
            " 公开片段提供的可用线索包括："
            + "；".join(excerpts[:3])
            + "。但这些片段仍不足以完整证明输入、处理、接口写回和异常接管机制。"
        )

    @staticmethod
    def _evidence_summary(sources: list[CollectedSource]) -> str:
        titles: list[str] = []
        seen: set[str] = set()
        for source in sources:
            title = DailyBriefingService._trim_source_title(source.title)
            key = title.lower()
            if not title or key in seen:
                continue
            titles.append(title)
            seen.add(key)
            if len(titles) >= 4:
                break
        if not titles:
            return "本组暂无可读标题"
        suffix = "等" if len(sources) > len(titles) else ""
        return "主要证据包括：" + "；".join(f"《{title}》" for title in titles) + suffix + "。"

    @staticmethod
    def _short_evidence_excerpt(value: str, max_chars: int = 150) -> str:
        text = " ".join(value.split()).strip()
        if not text:
            return ""
        text = re.sub(r"作者：.*$", "", text).strip()
        text = re.sub(r"[-—]{2,}.*$", "", text).strip()
        candidates = [part.strip() for part in re.split(r"[。！？；\n]+", text) if part.strip()]
        if candidates:
            text = max(candidates[:3], key=len)
        if len(text) < 8 or not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", text):
            return ""
        if len(text) > max_chars:
            text = text[: max_chars - 1].rstrip("，,；;：:、 ") + "…"
        return text

    @staticmethod
    def _trim_source_title(value: str, max_chars: int = 46) -> str:
        title = " ".join(value.split()).strip()
        title = re.sub(r"[_\-—|].*$", "", title).strip() or title
        if len(title) > max_chars:
            return title[: max_chars - 1].rstrip("，,；;：:、 ") + "…"
        return title
