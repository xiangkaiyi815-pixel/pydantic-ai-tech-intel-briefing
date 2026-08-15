from datetime import date
import json
import time
from search_assistant.briefing.service import DailyBriefingService, REPORT_SKILL_PATH, _is_substantive_synthesis
from search_assistant.contracts import BriefingSynthesis, BriefingTheme, CollectedSource, IncomingMessage
from search_assistant.memory.store import MemoryStore
from search_assistant.search.provider import SearchResult
from search_assistant.workflow.runtime import FakeAgentRuntime
from search_assistant.workflow.runtime import GLMPydanticAIRuntime, runtime_from_settings
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


def test_daily_briefing_records_candidate_lifecycle_and_provider_trace(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    service = DailyBriefingService(store, NoisySearchClient())
    subscription = store.upsert_topic("u-1", "c-1", "industrial AI")

    collection = service._collect_sources_with_trace(
        subscription,
        [("technical", "industrial AI agent MES manufacturing")],
        [],
        run_id="brief-test",
    )

    assert [source.url for source in collection.sources] == ["https://example.com/industrial-agent"]
    statuses = {candidate.status for candidate in collection.source_candidates}
    assert "accepted" in statuses
    assert "rejected_generic_reference" in statuses
    assert "rejected_missing_industrial_anchor" in statuses
    assert collection.provider_events[0].status == "success"
    assert store.list_provider_trace_events(topic_id=subscription.id)[0].provider == "NoisySearchClient"
    persisted_statuses = {candidate.status for candidate in store.list_source_candidates(topic_id=subscription.id)}
    assert statuses.issubset(persisted_statuses)


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
