from datetime import date
import json
import time
from search_assistant.briefing.service import DailyBriefingService, REPORT_SKILL_PATH, _is_substantive_synthesis
from search_assistant.contracts import BriefingSynthesis, BriefingTheme, CollectedSource, IncomingMessage
from search_assistant.memory.store import MemoryStore
from search_assistant.search.provider import SearchResult
from search_assistant.workflow.runtime import DeepSeekChatRuntime, FakeAgentRuntime
from search_assistant.workflow.runtime import GLMPydanticAIRuntime, MicrosoftAgentRuntime, runtime_from_settings
from search_assistant.workflow.service import SearchAssistantWorkflow
from search_assistant.config import Settings


class RecordingSearchClient:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        self.queries.append(query)
        results = [
            SearchResult(
                title="制造业 AI Agent 连接 MES 与工单闭环",
                url="https://www.example.com/agent-mes/",
                snippet="Agent 调用 MES、ERP 与工单系统，保留人工审批和生产追溯。",
                provider="direct-official",
                checked_at="2026-07-23T00:00:00+00:00",
            ),
            SearchResult(
                title="工业数字孪生用于虚拟调试与产线仿真",
                url="https://www.bilibili.com/video/BV1test",
                snippet="数字孪生结合仿真、工艺约束和现场数据，用于部署前验证。",
                provider="browser-google",
                checked_at="2026-07-23T00:00:00+00:00",
            ),
            SearchResult(
                title="AOI 质检与缺陷追溯的生产数据闭环",
                url="https://mp.weixin.qq.com/s/example",
                snippet="机器视觉识别缺陷，并将批次数据回写质量追溯系统。",
                provider="browser-bing",
                checked_at="2026-07-23T00:00:00+00:00",
            ),
        ]
        return results[:limit]


class BudgetedSearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if query == "slow":
            time.sleep(0.3)
            return [
                SearchResult(
                    title="slow result",
                    url="https://example.com/slow",
                    snippet="This result must not be retained after the search budget expires.",
                    provider="test",
                    checked_at="2026-07-25T00:00:00+00:00",
                )
            ]
        return [
            SearchResult(
                title="Industrial AI agent fast result",
                url="https://example.com/fast",
                snippet="An agent connects machine events to MES work orders with approval controls.",
                provider="test",
                checked_at="2026-07-25T00:00:00+00:00",
            )
        ]


class NoisySearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title="Industrial definition",
                url="https://baike.baidu.com/item/industrial",
                snippet="A generic reference page.",
                provider="browser-bing",
                checked_at="2026-07-25T00:00:00+00:00",
            ),
            SearchResult(
                title="General AI cloud page",
                url="https://example.com/general-ai",
                snippet="AI cloud productivity platform without domain evidence.",
                provider="browser-bing",
                checked_at="2026-07-25T00:00:00+00:00",
            ),
            SearchResult(
                title="Industrial AI agent connects MES and maintenance workflows",
                url="https://example.com/industrial-agent",
                snippet="The agent reads machine events, invokes MES work orders, and keeps an approval trail.",
                provider="mcp:public:test",
                checked_at="2026-07-25T00:00:00+00:00",
            ),
        ][:limit]


class GenericIndustrySearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title="NAAI 发布全球人工智能产业发展报告",
                url="https://mp.weixin.qq.com/s/industry-report",
                snippet="报告覆盖人工智能产业发展规模、产业链、政策环境和投资方向，但没有直接说明系统接口。",
                provider="browser-baidu",
                checked_at="2026-08-08T00:00:00+00:00",
            ),
            SearchResult(
                title="人工智能产业发展中的数据集与基准建设",
                url="https://example.com/ai-dataset-benchmark",
                snippet="文章讨论人工智能产业发展所需的数据集、基准、开源模型和评估指标。",
                provider="browser-bing",
                checked_at="2026-08-08T00:00:00+00:00",
            ),
            SearchResult(
                title="人工智能产业发展进入制造业与政务工作流",
                url="https://example.com/ai-application-case",
                snippet="案例描述人工智能进入业务流程后仍需要数据输入、系统接口、人工审批和 ROI 验证。",
                provider="browser-google",
                checked_at="2026-08-08T00:00:00+00:00",
            ),
            SearchResult(
                title="公开视频解读人工智能产业发展趋势",
                url="https://www.bilibili.com/video/av996452521",
                snippet="公开视频从科普角度说明人工智能产业发展趋势，属于传播信号而非系统实现证据。",
                provider="bilibili-public-api",
                checked_at="2026-08-08T00:00:00+00:00",
            ),
            SearchResult(
                title="人工智能产业发展大会展示产品与生态合作",
                url="https://example.com/ai-conference",
                snippet="大会展示人工智能产品更新、生态合作和产业讨论，仍需补充原始技术文档和部署指标。",
                provider="browser-baidu",
                checked_at="2026-08-08T00:00:00+00:00",
            ),
        ][:limit]


class MarketingSearchClient:
    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                title="Industrial AI MES agent training camp limited offer",
                url="https://mp.weixin.qq.com/s/marketing-camp",
                snippet=(
                    "Industrial AI MES agent case with scan QR, add WeChat, coupon, "
                    "course enrollment, and business cooperation."
                ),
                provider="browser-baidu",
                checked_at="2026-07-25T00:00:00+00:00",
            ),
            SearchResult(
                title="Industrial AI MES agent architecture and edge deployment",
                url="https://example.com/industrial-agent-architecture",
                snippet=(
                    "The system describes machine events, MES work order interfaces, "
                    "edge inference deployment, approval trails, and benchmark evaluation."
                ),
                provider="mcp:public:test",
                checked_at="2026-07-25T00:00:00+00:00",
            ),
        ][:limit]


class PlanningRuntime(FakeAgentRuntime):
    def __init__(self):
        super().__init__()
        self.briefing_context: dict[str, object] | None = None

    def plan_briefing_queries(self, topic: str, context: dict[str, object]) -> list[str]:
        self.briefing_context = context
        return [
            "industrial AI agent MES ERP workflow architecture",
            "LinkedIn industrial AI discussion",
            "predictive maintenance time-series anomaly detection evaluation",
        ]


class SourceLimitRuntime(FakeAgentRuntime):
    def __init__(self):
        super().__init__()
        self.synthesis_source_count = 0

    def synthesize_briefing(self, topic, sources, context):
        self.synthesis_source_count = len(sources)
        return None


class SingleThemeRuntime(FakeAgentRuntime):
    def synthesize_briefing(self, topic, sources, context):
        return BriefingSynthesis(
            search_content_summary="本轮材料讨论的是把生产事件接入受控工具链，而不是单独部署一个对话模型。",
            short_summary=(
                "这组方案把设备事件作为输入，用领域检索和规则把事件转换为候选工单，再通过 MES 适配器写入执行系统；"
                "审批与审计日志位于模型调用之后，因而核心差异不在语言模型本身，而在能否把建议约束成可回滚、可追溯的生产动作。"
            ),
            detailed_summary=(
                "### 模型不直接控制设备，而是生成受控的工单候选\n"
                "来源描述的链路从设备事件和生产上下文开始，模型先结合领域检索解释异常，再调用受权限限制的 MES 接口生成候选工单。"
                "工单没有直接下发到现场，而是先经过人工审批并记录输入、调用参数和最终写回结果；这把生成式模型放在语义归纳层，"
                "把确定性执行和责任归属留给现有生产系统。\n\n"
                "### 真正需要验证的是接口契约和失败路径\n"
                "公开描述能支持事件到工单的流程，但没有给出异常分类准确率、接口失败重试、权限越界阻断或跨班组回滚的量化数据。"
                "因此这类系统的评估不应只看问答效果，而应复现同一事件在不同权限、数据缺失和人工驳回条件下是否仍能产生可审计的结果。"
            ),
            themes=[
                BriefingTheme(
                    name="模型给出的受控工单链路",
                    analysis=(
                        "模型承担事件语义归纳和候选工单生成，MES 负责确定性写回，审批和审计记录构成控制边界；"
                        "缺少接口失败与权限越界的数据前，不能把该链路视为可直接自治的生产控制方案。"
                    ),
                    source_urls=[sources[0].url],
                )
            ],
            key_signal_interpretation="反复出现的共同结构是事件、模型、工具调用、审批和工单写回，而不是孤立的大模型问答。",
            analysis_judgment=(
                "这类方案的护城河不是把通用模型部署到工厂，而是把权限模型、MES 接口、审计记录与异常回滚编成可验证的执行契约。"
                "没有这些契约，模型即使能解释事件，也无法承担生产流程中的责任边界。"
            ),
            next_search_directions=["核验 MES 接口与权限模型", "寻找失败重试和回滚的公开指标"],
            landing_suggestions=["先限制在候选工单场景", "保留审批和完整审计记录"],
        )


def test_daily_briefing_searches_public_social_channels_and_renders_required_contract(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    search = RecordingSearchClient()
    service = DailyBriefingService(
        store=store,
        search_client=search,
        runtime=FakeAgentRuntime(),
        max_queries=13,
        results_per_query=3,
        max_sources=10,
    )

    briefing = service.run("AI+工业界发展", "u-1", "c-1", run_date=date(2026, 7, 23))

    assert any("site:mp.weixin.qq.com" in query for query in search.queries)
    assert any("site:douyin.com" in query for query in search.queries)
    assert any("site:bilibili.com" in query for query in search.queries)
    assert any("site:linkedin.com" in query for query in search.queries)
    assert len(briefing.sources) == 3
    assert briefing.sources[0].url == "https://www.example.com/agent-mes"
    for section in (
        "## 搜索方向",
        "## 关键词",
        "## 搜索内容总结",
        "## 简短总结",
        "## 详细总结",
        "## AI 分析判断",
        "## 下一步搜索方向",
        "## 落地建议",
        "## 原文链接",
    ):
        assert section in briefing.markdown
    assert "### 本轮技术主题地图" not in briefing.markdown
    assert "### 核心技术提炼" not in briefing.markdown
    assert "### 重点线索解读" not in briefing.markdown
    assert "https://www.example.com/agent-mes" in briefing.markdown
    assert "https://www.bilibili.com/video/BV1test" in briefing.markdown
    assert store.latest_daily_briefing(briefing.topic_id).id == briefing.id
    candidates = store.list_domain_knowledge_candidates()
    assert candidates
    assert all(candidate["status"] == "candidate" for candidate in candidates)
    assert all(briefing.id in candidate["source_ids"] for candidate in candidates)
    assert all(candidate["evidence"] for candidate in candidates)
    validation_gates = store.list_gate_records(gate_type="domain_knowledge_candidate_validation")
    assert len(validation_gates) == len(candidates)
    validation_passed = sum(1 for gate in validation_gates if gate["result"] == "passed")
    validation_failed = sum(1 for gate in validation_gates if gate["result"] == "failed")
    assert validation_passed + validation_failed == len(candidates)
    ledger_entries = store.list_project_ledger_entries(entry_type="briefing_run")
    assert len(ledger_entries) == 1
    assert ledger_entries[0]["metadata"]["source_count"] == 3
    assert ledger_entries[0]["metadata"]["validation_gate_passed"] == validation_passed
    assert ledger_entries[0]["metadata"]["validation_gate_failed"] == validation_failed
    assert sum(ledger_entries[0]["metadata"]["knowledge_layers"].values()) == len(candidates)
    assert store.latest_project_ledger_snapshot("pydantic-ai-tech-intel-briefing")["status"] == "active"
    trace_names = {event["name"] for event in store.list_trace_events()}
    assert {
        "plan_queries",
        "collect_sources",
        "synthesize_report",
        "capture_domain_knowledge_candidates",
        "run",
    }.issubset(trace_names)
    checkpoint_steps = {checkpoint["step"] for checkpoint in store.list_run_checkpoints()}
    assert {
        "planned",
        "sources_collected",
        "synthesized",
        "candidates_captured",
        "completed",
    }.issubset(checkpoint_steps)
    provider_health = store.list_search_provider_health()
    assert len(provider_health) == len(search.queries)
    assert all(row["ok"] for row in provider_health)


def test_daily_briefing_keeps_completed_sources_when_the_search_budget_expires(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(
        store,
        BudgetedSearchClient(),
        search_budget_seconds=0.03,
    )
    subscription = store.upsert_topic("u-1", "c-1", "industrial AI")

    started = time.monotonic()
    sources = service._collect_sources(
        subscription,
        [("test", "industrial AI agent fast"), ("test", "slow")],
        [],
    )

    assert time.monotonic() - started < 0.15
    assert [source.url for source in sources] == ["https://example.com/fast"]


def test_daily_briefing_filters_generic_reference_pages_before_ranking(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(store, NoisySearchClient())
    subscription = store.upsert_topic("u-1", "c-1", "industrial AI")

    sources = service._collect_sources(
        subscription,
        [("technical", "industrial AI agent MES manufacturing")],
        [],
    )

    assert [source.url for source in sources] == ["https://example.com/industrial-agent"]


def test_daily_briefing_filters_marketing_account_content_before_ranking(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(store, MarketingSearchClient())
    subscription = store.upsert_topic("u-1", "c-1", "industrial AI")

    sources = service._collect_sources(
        subscription,
        [("public-social", "industrial AI MES agent architecture deployment")],
        [],
    )

    assert [source.url for source in sources] == ["https://example.com/industrial-agent-architecture"]


def test_marketing_filter_keeps_technical_public_account_posts():
    assert not DailyBriefingService._is_report_source_candidate(
        "industrial AI",
        "industrial AI MES agent architecture deployment",
        "https://mp.weixin.qq.com/s/marketing-camp",
        "Industrial AI MES agent training camp limited offer",
        "Scan QR, add WeChat, coupon, course enrollment, and business cooperation.",
    )
    assert DailyBriefingService._is_report_source_candidate(
        "industrial AI",
        "industrial AI MES agent architecture deployment",
        "https://mp.weixin.qq.com/s/technical-architecture",
        "Industrial AI MES agent architecture and edge deployment",
        "Machine events, MES work order interfaces, edge inference, approval trails, and benchmark evaluation.",
    )


def test_case_feedback_changes_follow_up_briefing_direction_and_keeps_original_url(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(store, RecordingSearchClient(), runtime=FakeAgentRuntime(), max_queries=16)

    feedback_id = service.add_feedback(
        "AI+工业界发展",
        "u-1",
        "c-1",
        "重点观察半导体良率优化、工艺参数闭环和设备协同。",
        "https://www.toutiao.com/article/example",
    )
    briefing = service.run("AI+工业界发展", "u-1", "c-1", run_date=date(2026, 7, 23))

    assert feedback_id.startswith("feedback_")
    assert any("半导体良率优化" in direction for direction in briefing.search_directions)
    assert any(source.url == "https://www.toutiao.com/article/example" for source in briefing.sources)
    assert "https://www.toutiao.com/article/example" in briefing.markdown


def test_daily_briefing_uses_model_planned_technical_queries_and_static_skill(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    search = RecordingSearchClient()
    runtime = PlanningRuntime()
    service = DailyBriefingService(store, search, runtime=runtime, max_queries=18)

    briefing = service.run("industrial AI", "u-1", "c-1", run_date=date(2026, 7, 23))

    assert "industrial AI agent MES ERP workflow architecture" in search.queries
    assert "LinkedIn industrial AI discussion" not in search.queries
    assert "predictive maintenance time-series anomaly detection evaluation" in search.queries
    assert "LinkedIn industrial AI discussion" not in briefing.keywords
    assert runtime.briefing_context is not None
    assert "内容搜集报告" in str(runtime.briefing_context["report_skill"])
    assert REPORT_SKILL_PATH.exists()


def test_case_feedback_uses_public_index_context_before_the_next_planning_step(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    search = RecordingSearchClient()
    runtime = PlanningRuntime()
    service = DailyBriefingService(store, search, runtime=runtime, max_queries=18)

    service.add_feedback("industrial AI", "u-1", "c-1", "", "https://www.toutiao.com/article/example")
    service.run("industrial AI", "u-1", "c-1", run_date=date(2026, 7, 23))

    assert "https://www.toutiao.com/article/example" in search.queries
    feedback = runtime.briefing_context["feedback"]
    assert "案例公开索引" in str(feedback)


def test_daily_briefing_archives_more_sources_than_it_sends_to_the_model(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = SourceLimitRuntime()
    service = DailyBriefingService(
        store,
        RecordingSearchClient(),
        runtime=runtime,
        max_queries=13,
        results_per_query=3,
        max_sources=10,
        model_max_sources=2,
    )

    briefing = service.run("industrial AI", "u-1", "c-1", run_date=date(2026, 7, 23))

    assert len(briefing.sources) == 3
    assert runtime.synthesis_source_count == 2


def test_generic_topic_fallback_groups_sources_into_readable_evidence_blocks(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime = SourceLimitRuntime()
    service = DailyBriefingService(
        store,
        GenericIndustrySearchClient(),
        runtime=runtime,
        max_queries=3,
        results_per_query=5,
        max_sources=8,
        model_max_sources=2,
    )

    briefing = service.run("人工智能产业发展", "u-1", "c-1", run_date=date(2026, 8, 8))

    assert runtime.synthesis_source_count == 2
    assert len(briefing.sources) == 5
    assert "保留了 5 条公开线索" in briefing.synthesis.search_content_summary
    assert "### 政策、规模与产业链信号" in briefing.synthesis.detailed_summary
    assert "### 技术底座、数据与开源生态" in briefing.synthesis.detailed_summary
    assert "### 应用落地与业务转型案例" in briefing.synthesis.detailed_summary
    assert "### 教育传播、公众讨论与弱证据线索" in briefing.synthesis.detailed_summary
    assert "相关线索集中讨论" not in briefing.synthesis.detailed_summary
    assert _is_substantive_synthesis(briefing.synthesis) is True


def test_short_single_block_generated_detail_is_not_substantive():
    synthesis = BriefingSynthesis(
        search_content_summary="本轮来源共同讨论人工智能产业发展，但材料需要继续分层核验。",
        short_summary=(
            "本轮材料需要先区分宏观产业信号、技术底座和业务落地案例，再核验数据输入、模型或工具链、"
            "系统接口、评估指标和人工接管机制，不能只按标题判断技术成熟度。"
        ),
        detailed_summary=(
            "### 产业动态\n"
            "本轮来源包含报告、教程和公开视频，说明人工智能产业发展受到关注，但这段总结过短，"
            "没有把证据拆成政策、技术底座、应用场景和弱证据层次。"
        ),
        themes=[
            BriefingTheme(
                name="产业动态",
                analysis="公开材料说明话题热度上升，但没有给出足够系统接口、评测指标和部署边界。",
                source_urls=["https://example.com/source"],
            )
        ],
        key_signal_interpretation="这些来源需要先按证据强度分层，再判断是否能支持技术实现和落地能力。",
        analysis_judgment="报告和视频可以说明关注度，但不能直接证明工程闭环已经成熟，需要继续核验原始技术材料和可量化指标。",
        next_search_directions=["补查原始技术文档", "核验公开评测指标"],
        landing_suggestions=["先建立证据分层", "只把有接口和指标的来源用于落地判断"],
    )

    assert _is_substantive_synthesis(synthesis) is False


def test_daily_briefing_keeps_model_selected_detail_structure_without_injecting_static_themes(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(
        store,
        RecordingSearchClient(),
        runtime=SingleThemeRuntime(),
        max_queries=13,
        results_per_query=3,
        max_sources=10,
    )

    briefing = service.run("industrial AI", "u-1", "c-1", run_date=date(2026, 7, 23))

    assert briefing.synthesis.analysis_judgment.startswith("这类方案的护城河")
    assert len(briefing.synthesis.themes) == 1
    assert briefing.synthesis.themes[0].name == "模型给出的受控工单链路"
    assert "### 模型不直接控制设备，而是生成受控的工单候选" in briefing.markdown


def test_cad_topic_filter_rejects_generic_ai_content_and_keeps_engineering_evidence():
    topic = "AI 3D CAD engineering drawing"
    query = "text-to-CAD parametric modeling B-Rep evaluation"

    assert not DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://example.com/opencode",
        "OpenCode AI coding agent",
        "An AI agent for repositories and shell commands.",
    )
    assert DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://example.com/text-to-cad",
        "Text-to-CAD parametric B-Rep generation",
        "The system produces editable CAD features and validates geometric constraints.",
    )


def test_report_source_filter_keeps_exact_chinese_topic_phrase_without_overmatching_fragments():
    topic = "人工智能产业发展"
    query = "site:bilibili.com 人工智能产业发展"

    assert DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://www.bilibili.com/video/av116578230279494",
        "【政策研究】中国 人工智能产业发展 调查",
        "围绕人工智能产业发展讨论政策、产业链与应用落地。",
    )
    assert DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://www.bilibili.com/video/av114070036480806",
        "人工智能创新加速我国产业转型升级",
        "公开视频讨论人工智能技术快速发展、产业链和应用落地。",
    )
    assert not DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://example.com/ren-gong",
        "人工 的意思",
        "人工是一个汉语词汇解释页面。",
    )


def test_cad_topic_filter_rejects_search_dumps_login_pages_and_medical_cad():
    topic = "AI 3D CAD engineering drawing"
    query = "text-to-CAD parametric modeling B-Rep evaluation"

    assert not DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://m.baidu.com/search-result",
        "CAD result",
        "A CAD result",
    )
    assert not DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://web.autocad.com/login",
        "AutoCAD Web App sign in",
        "CAD editor sign in",
    )
    assert not DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://example.com/medical",
        "Coronary artery disease",
        "Coronary artery disease symptoms and treatment",
    )
    assert not DailyBriefingService._is_report_source_candidate(
        topic,
        query,
        "https://example.com/search-dump",
        "CAD search result",
        '{"not_struct":true,"renderFlags":{"fp":1},"cosmicPkgs":{}}',
    )


def test_cad_topic_uses_deterministic_technical_queries_when_model_planning_is_empty(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(store, RecordingSearchClient(), runtime=FakeAgentRuntime(), max_queries=24)
    subscription = store.upsert_topic("u-1", "c-1", "AI 3D CAD engineering drawing")

    plan = service.build_search_plan(subscription, [])
    technical_queries = [query for platform, query in plan if "确定性保障" in platform]

    assert len(technical_queries) == 5
    assert any("Text-to-CAD" in query for query in technical_queries)
    assert any("DWG DXF" in query for query in technical_queries)


def test_cad_fallback_uses_cad_technical_themes_instead_of_industrial_templates(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(store, RecordingSearchClient())
    source = CollectedSource(
        id="src-cad",
        topic_id="topic-cad",
        user_id="u-1",
        title="Text-to-CAD creates editable parametric B-Rep features",
        url="https://example.com/text-to-cad",
        snippet="Natural language is converted into sketches, dimensions, and constrained CAD feature history.",
        platform="technical route",
        provider="mcp:public:github-api",
        query="text-to-CAD parametric B-Rep evaluation",
        importance_score=12.0,
        retrieved_at="2026-07-26T00:00:00+00:00",
    )

    synthesis = service._fallback_synthesis("AI 3D CAD engineering drawing", [source])

    assert any("CAD" in theme.name for theme in synthesis.themes)
    assert "geometry" in synthesis.short_summary.lower() or "CAD" in synthesis.short_summary
    assert "MES" not in synthesis.search_content_summary


def test_scoped_memory_and_skill_promotion_permission_do_not_cross_user_boundaries(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.add_memory_item("topic", "User A private topic", "q-a", user_id="u-a", chat_id="c-a")
    store.add_experience_item("User A feedback", "private rule", ["q-a"], user_id="u-a", chat_id="c-a")

    assert store.list_memory_items("u-b", "c-b") == []
    assert store.list_experience_items("u-b", "c-b") == []

    workflow = SearchAssistantWorkflow(
        store=store,
        runtime=FakeAgentRuntime(),
        admin_user_ids=set(),
        skill_drafts_dir=tmp_path / "skills" / "drafts",
    )
    package = workflow.answer(
        IncomingMessage(
            message_id="m-1",
            user_id="u-b",
            chat_id="c-b",
            text="启用skill：anything",
        )
    )

    assert "没有启用 Skill 的权限" in package.answer_text
    assert package.review["approved"] is False


def test_glm_briefing_synthesis_validates_text_json_against_collected_urls():
    source = CollectedSource(
        id="src-1",
        topic_id="topic-1",
        user_id="u-1",
        title="Industrial AI agent",
        url="https://example.com/industrial-agent",
        snippet="Agent reads machine events and writes approved work orders to MES.",
        platform="technical",
        provider="mcp:public:test",
        query="industrial AI agent MES",
        importance_score=9.0,
        retrieved_at="2026-07-25T00:00:00+00:00",
    )
    output = {
        "search_content_summary": "来源描述了工业 AI 从孤立模型走向受控生产工作流的趋势。",
        "short_summary": (
            "这类工业智能体把设备事件输入结合检索和规则转为候选工单，再由 MES 适配器写回业务系统；"
            "人工审批、权限校验和审计日志不属于模型能力，却决定生成结果是否能成为可回滚、可追溯的生产动作。"
        ),
        "detailed_summary": (
            "### 事件到工单的实现边界\n"
            "来源中的实现不是让模型直接控制设备，而是将机器事件和生产上下文送入检索、规则与语言模型组成的解释层，"
            "由该层提出工单候选，再通过受权限控制的 MES 接口写入既有系统。模型负责把非结构化事件归纳成可操作语义，"
            "MES 则继续负责工单状态、人员权限和流程一致性，因此它更像受控编排器而不是自动控制器。\n\n"
            "### 不能被标题掩盖的验证缺口\n"
            "材料支持审批和追溯存在，却没有提供异常分类误差、接口失败后的补偿事务、越权调用阻断率或人工驳回后的回滚指标。"
            "在这些数据出现前，评估应覆盖相同事件在数据缺失、权限变化和人工拒绝条件下的端到端结果，"
            "而不是只测模型是否能给出貌似合理的文字解释。"
        ),
        "themes": [
            {
                "name": "受控工单编排",
                "analysis": "模型通过检索和规则解释设备事件并生成候选工单，MES 适配器负责确定性写回；审批、权限和审计是该系统能否进入生产流程的关键边界。",
                "source_urls": [source.url],
            }
        ],
        "key_signal_interpretation": "关键结构是事件、语义归纳、候选工单、审批和系统写回形成的闭环，而不是单一模型标签。",
        "analysis_judgment": "工业 AI 的真实价值取决于系统接口、权限边界和可追溯执行，而不是通用模型的文字生成能力。只有在失败补偿和人工接管被验证后，候选工单才能成为生产流程的一部分。",
        "next_search_directions": ["核验 MES 适配器的事务设计", "寻找失败补偿与人工接管指标"],
        "landing_suggestions": ["从候选工单开始试点", "记录每次调用与回滚路径"],
    }
    calls = []

    def runner(*args):
        calls.append(args)
        return json.dumps(output)

    runtime = GLMPydanticAIRuntime(api_key="test-key", agent_runner=runner)
    synthesis = runtime.synthesize_briefing("industrial AI", [source], {"report_contract": {}, "report_skill": "contract"})

    assert synthesis.themes[0].source_urls == [source.url]
    assert _is_substantive_synthesis(synthesis) is True
    assert calls[0][6] == 2400
    assert "Return ONLY one valid JSON object" in calls[0][3]
    assert "detailed_summary" in calls[0][3]
    assert "input form" in calls[0][3]
    assert "industry-wide claim" in calls[0][3]

    retry_calls = []

    def retry_runner(*args):
        retry_calls.append(args)
        if len(retry_calls) == 1:
            raise TimeoutError("simulated GLM timeout")
        return json.dumps(output)

    retry_runtime = GLMPydanticAIRuntime(api_key="test-key", agent_runner=retry_runner)
    retry_runtime.synthesize_briefing("industrial AI", [source], {"report_contract": {}, "report_skill": "contract"})
    assert len(retry_calls) == 2
    assert retry_calls[1][6] == 1800
    assert len(json.loads(retry_calls[1][4])["sources"]) == 1


def test_deepseek_briefing_planning_and_synthesis_call_the_model_runner():
    source = CollectedSource(
        id="src-1",
        topic_id="topic-1",
        user_id="u-1",
        title="Industrial AI agent",
        url="https://example.com/industrial-agent",
        snippet="Agent reads machine events and writes approved work orders to MES.",
        platform="technical",
        provider="mcp:public:test",
        query="industrial AI agent MES",
        importance_score=9.0,
        retrieved_at="2026-07-25T00:00:00+00:00",
    )
    output = {
        "search_content_summary": "本轮来源显示工业 AI 正把设备事件、审批流程和 MES 写回连接成受控工作流。",
        "short_summary": (
            "这类工业智能体以设备事件和生产上下文为输入，先由检索、规则和语言模型生成候选工单，"
            "再通过受权限控制的 MES 接口写回；人工审批、审计日志和失败回滚决定它能否进入生产流程。"
        ),
        "detailed_summary": (
            "### 事件到工单的受控编排\n"
            "来源支持的路径不是让模型直接控制设备，而是把机器事件、生产上下文和历史规则送入解释层。"
            "解释层生成候选工单后，仍由 MES 适配器、权限模型和审批流程完成确定性写回，这让模型停留在语义归纳和建议生成层，"
            "把生产执行责任保留在既有系统和人工审核链路中。\n\n"
            "### 仍需补齐的验证指标\n"
            "现有材料没有给出异常分类误差、接口失败补偿、越权阻断率或人工驳回后的回滚指标。"
            "因此评估这类方案时，应把同一事件在数据缺失、权限变化和人工拒绝条件下的端到端结果纳入测试，"
            "而不是只检查模型能否生成看似合理的工单说明。"
            "这些指标会决定系统是可审计的生产辅助，还是只能停留在演示层的文本自动化。"
        ),
        "themes": [
            {
                "name": "受控工单编排",
                "analysis": "模型负责解释设备事件并生成候选工单，MES 和审批流程负责确定性写回、权限边界与审计记录。",
                "source_urls": [source.url],
            }
        ],
        "key_signal_interpretation": "关键结构是事件、解释层、候选工单、审批和系统写回形成的闭环。",
        "analysis_judgment": (
            "本轮材料支持把工业 AI 视为受控编排层，而不是自主控制层；下一步应核验接口、权限和回滚指标。"
            "只有这些边界被验证后，候选工单才适合进入真实生产流程。"
        ),
        "next_search_directions": ["核验 MES 适配器事务设计", "寻找失败补偿和人工接管指标"],
        "landing_suggestions": ["从候选工单试点开始", "记录每次调用、审批和回滚路径"],
    }
    calls: list[dict[str, object]] = []

    def runner(model, api_key, base_url, instructions, prompt, temperature, max_tokens, timeout_seconds):
        calls.append(
            {
                "model": model,
                "base_url": base_url,
                "instructions": instructions,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
            }
        )
        if max_tokens == 700:
            return json.dumps(["industrial AI MES workflow architecture"])
        return json.dumps(output, ensure_ascii=False)

    runtime = DeepSeekChatRuntime(
        api_key="test-key",
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        timeout_seconds=180.0,
        briefing_planning_timeout_seconds=40.0,
        agent_runner=runner,
    )

    queries = runtime.plan_briefing_queries("industrial AI", {"feedback": []})
    synthesis = runtime.synthesize_briefing("industrial AI", [source], {"report_contract": {}, "report_skill": "contract"})

    assert queries == ["industrial AI MES workflow architecture"]
    assert synthesis.themes[0].source_urls == [source.url]
    assert _is_substantive_synthesis(synthesis) is True
    assert [call["max_tokens"] for call in calls[:2]] == [700, 2400]
    assert calls[0]["timeout_seconds"] == 40.0
    assert calls[1]["timeout_seconds"] == 180.0
    assert calls[1]["model"] == "deepseek-v4-flash"
    assert calls[1]["base_url"] == "https://api.deepseek.com"
    assert "Return ONLY one valid JSON object" in str(calls[1]["instructions"])
    assert json.loads(str(calls[1]["prompt"]))["sources"][0]["url"] == source.url


def test_glm_provider_selects_the_pydantic_ai_runtime():
    runtime = runtime_from_settings(
        Settings.from_env(
            {
                "SEARCH_ASSISTANT_MODEL_PROVIDER": "glm",
                "GLM_API_KEY": "test-key",
                "GLM_MODEL": "glm-4.7",
            }
        )
    )

    assert isinstance(runtime, GLMPydanticAIRuntime)
    assert runtime.model == "glm-4.7"
    assert runtime.timeout_seconds == 120.0


def test_deepseek_provider_selects_runtime_with_briefing_planning_timeout():
    runtime = runtime_from_settings(
        Settings.from_env(
            {
                "SEARCH_ASSISTANT_MODEL_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test-key",
                "DEEPSEEK_MODEL": "deepseek-v4-flash",
                "DEEPSEEK_TIMEOUT_SECONDS": "180",
                "BRIEFING_PLANNING_TIMEOUT_SECONDS": "40",
            }
        )
    )

    assert isinstance(runtime, MicrosoftAgentRuntime)
    assert runtime.model == "deepseek-v4-flash"
    assert runtime.timeout_seconds == 180.0
    assert runtime.briefing_planning_timeout_seconds == 40.0


def test_glm_scope_review_detects_unsupported_industry_wide_claims():
    synthesis = BriefingSynthesis(
        search_content_summary="本轮有少数 CAD 项目采用参数化代码生成。",
        short_summary="本轮材料中的项目将自然语言转换为参数化代码，再由 CAD 内核生成可编辑实体；这说明该路线值得进一步核验，但不足以代表整个行业。",
        detailed_summary=(
            "### 证据范围\n"
            "严肃的机械 CAD 生成已放弃纯网格输出，转向参数化代码和可验证几何内核；"
            "这一判断需要被改写为本轮项目的选择，而不是行业事实。"
        ),
        themes=[
            BriefingTheme(
                name="参数化代码",
                analysis="少数项目把自然语言转换为 CAD 脚本并交由几何内核执行，这提供了一条可编辑的实现路径，但尚不构成行业普遍结论。",
                source_urls=["https://example.com/cad"],
            )
        ],
        key_signal_interpretation="这些来源共同强调了参数化表示和几何校验的价值。",
        analysis_judgment="应以项目材料的直接描述界定技术结论，不能把少数案例中的产品路线写成整个行业已经完成的迁移。",
        next_search_directions=["查找工程基准", "核验几何内核"],
        landing_suggestions=["保留规则校验", "限制在候选生成"],
    )

    assert GLMPydanticAIRuntime._needs_scope_revision(synthesis) is True

    scoped = synthesis.model_copy(
        update={
            "detailed_summary": (
                "### 证据范围\n"
                "本轮项目呈现了放弃纯网格输出、选择参数化代码路线的倾向，但没有材料能够支持它代表整个行业的结论。"
            )
        }
    )
    assert GLMPydanticAIRuntime._needs_scope_revision(scoped) is False
