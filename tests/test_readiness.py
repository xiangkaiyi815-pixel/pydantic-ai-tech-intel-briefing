import json

from search_assistant.config import Settings
from search_assistant.contracts import AnswerPackage, SourceEvidence, VerifiedClaim
from search_assistant.diagnostics.readiness import ReadinessService
from search_assistant.memory.store import MemoryStore


def test_readiness_accepts_glm_and_checked_in_hybrid_mcp_configuration(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    settings = Settings(
        data_dir=tmp_path,
        model_provider="glm",
        glm_api_key="glm-secret",
        glm_model="glm-4.7",
        search_provider="hybrid",
    )

    report = ReadinessService(store=store, settings=settings, data_dir=tmp_path).run()

    model_check = next(check for check in report["checks"] if check["name"] == "model_runtime")
    search_check = next(check for check in report["checks"] if check["name"] == "browser_search")
    assert model_check["ok"] is True
    assert model_check["data"] == {"provider": "glm", "model": "glm-4.7"}
    assert search_check["ok"] is True
    assert search_check["data"]["provider"] == "hybrid"
    assert "glm-secret" not in json.dumps(report)


def test_readiness_report_passes_with_production_config_and_local_evidence(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_answer(
        AnswerPackage(
            question_id="q-ready",
            answer_text="Feishu message reply API supports replying to received messages in 2026.",
            classification="research",
            confidence="medium",
            verified_claims=[
                VerifiedClaim(
                    claim="Feishu message reply API supports replying to received messages in 2026",
                    verdict="verified",
                    source="https://open.feishu.cn/document/server-docs/im-v1/message/reply",
                    checked_at="2026-07-03T00:00:00Z",
                )
            ],
            sources=[
                SourceEvidence(
                    title="Feishu message reply API",
                    url="https://open.feishu.cn/document/server-docs/im-v1/message/reply",
                    snippet="Feishu message reply API supports replying to received messages in 2026.",
                    provider="unit",
                    checked_at="2026-07-03T00:00:00Z",
                )
            ],
        )
    )
    store.add_profile_snapshot({"recurring_topics": ["Feishu integration"]})
    store.add_experience_item("Verification habit", "Search and calibrate API answers.", ["q-ready"])
    store.add_learning_report("# report", str(tmp_path / "reports" / "learning-report.md"))
    store.add_skill_draft("Feishu Reply Practice", str(tmp_path / "skills" / "drafts" / "skill"), ["q-ready"])
    store.record_runtime_session(
        "feishu-long-connection",
        process_id=1234,
        status="started",
        metadata={"model_provider": "deepseek"},
    )
    store.record_reply_attempt("q-ready", "om-ready", "feishu-long-connection", "success")
    evaluations_dir = tmp_path / "evaluations"
    evaluations_dir.mkdir()
    _write_quality_evaluation_report(evaluations_dir / "evaluation-report.json")
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: True,
        process_command_resolver=lambda pid: "python -m search_assistant.cli feishu-long-connection --data-dir live",
    ).run()

    assert report["ok"] is True
    assert {check["name"] for check in report["checks"]} >= {
        "model_runtime",
        "browser_search",
        "feishu_credentials",
        "runtime_session",
        "reply_attempt",
        "learning_evidence",
        "verification_evidence",
        "skill_evidence",
        "evaluation_report",
    }
    payload = json.dumps(report, ensure_ascii=False)
    assert "secret-key" not in payload
    assert "cli_secret" not in payload


def test_readiness_skill_evidence_reports_draft_and_promoted_counts_with_promotion_hint(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    draft_path = tmp_path / "skills" / "drafts" / "feishu-reply-practice" / "SKILL.md"
    draft_path.parent.mkdir(parents=True)
    draft_path.write_text("# Feishu Reply Practice", encoding="utf-8")
    store.add_skill_draft("Feishu Reply Practice", str(draft_path), ["q-ready"])
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: True,
        process_command_resolver=lambda pid: "python -m search_assistant.cli feishu-long-connection --data-dir live",
    ).run()

    skill_check = next(check for check in report["checks"] if check["name"] == "skill_evidence")
    assert skill_check["ok"] is True
    assert skill_check["data"] == {
        "skill_drafts": 1,
        "draft_skills": 1,
        "promoted_skills": 0,
        "active_skill_files": 0,
        "missing_active_skill_files": 0,
    }
    assert any("skill-promote" in hint for hint in report["hints"])


def test_readiness_flags_missing_verified_evidence_when_source_answers_exist(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_answer(
        AnswerPackage(
            question_id="q-source-no-evidence",
            answer_text="NVIDIA DGX Spark answer with sources but no verified claims.",
            classification="research",
            confidence="medium",
            sources=[
                SourceEvidence(
                    title="NVIDIA DGX Spark",
                    url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                    snippet="NVIDIA DGX Spark official specifications.",
                    provider="unit",
                    checked_at="2026-07-03T00:00:00Z",
                )
            ],
        )
    )
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(store=store, settings=settings, data_dir=tmp_path).run()

    verification_check = next(check for check in report["checks"] if check["name"] == "verification_evidence")
    assert verification_check["ok"] is False
    assert verification_check["data"]["source_backed_answers"] == 1
    assert verification_check["data"]["evidence"] == 0
    assert any("evidence-backfill" in hint for hint in report["hints"])


def test_readiness_skill_evidence_fails_when_promoted_skill_file_is_missing(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    missing_active_path = tmp_path / "skills" / "active" / "missing" / "SKILL.md"
    draft_id = store.add_skill_draft("Missing Active Skill", str(missing_active_path), ["q-ready"])
    store.update_skill_draft_status(draft_id, "promoted", path=str(missing_active_path))
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: True,
        process_command_resolver=lambda pid: "python -m search_assistant.cli feishu-long-connection --data-dir live",
    ).run()

    skill_check = next(check for check in report["checks"] if check["name"] == "skill_evidence")
    assert skill_check["ok"] is False
    assert skill_check["data"]["promoted_skills"] == 1
    assert skill_check["data"]["missing_active_skill_files"] == 1
    assert any("active skill file" in hint.lower() for hint in report["hints"])


def test_readiness_uses_latest_runtime_metadata_when_current_shell_lacks_secrets(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_answer(
        AnswerPackage(
            question_id="q-ready",
            answer_text="Feishu message reply API supports replying to received messages in 2026.",
            classification="research",
            confidence="medium",
            verified_claims=[
                VerifiedClaim(
                    claim="Feishu message reply API supports replying to received messages in 2026",
                    verdict="verified",
                    source="https://open.feishu.cn/document/server-docs/im-v1/message/reply",
                    checked_at="2026-07-03T00:00:00Z",
                )
            ],
            sources=[
                SourceEvidence(
                    title="Feishu message reply API",
                    url="https://open.feishu.cn/document/server-docs/im-v1/message/reply",
                    snippet="Feishu message reply API supports replying to received messages in 2026.",
                    provider="unit",
                    checked_at="2026-07-03T00:00:00Z",
                )
            ],
        )
    )
    store.add_profile_snapshot({"recurring_topics": ["Feishu integration"]})
    store.add_experience_item("Verification habit", "Search and calibrate API answers.", ["q-ready"])
    store.add_learning_report("# report", str(tmp_path / "reports" / "learning-report.md"))
    store.add_skill_draft("Feishu Reply Practice", str(tmp_path / "skills" / "drafts" / "skill"), ["q-ready"])
    store.record_runtime_session(
        "feishu-long-connection",
        process_id=1234,
        status="started",
        metadata={
            "model_provider": "deepseek",
            "deepseek_api_key_configured": True,
            "allow_fake_runtime": False,
            "search_provider": "browser",
            "browser_search_engines": ["bing", "baidu", "google"],
            "feishu_app_id_configured": True,
            "feishu_app_secret_configured": True,
        },
    )
    store.record_reply_attempt("q-ready", "om-ready", "feishu-long-connection", "success")
    evaluations_dir = tmp_path / "evaluations"
    evaluations_dir.mkdir()
    _write_quality_evaluation_report(evaluations_dir / "evaluation-report.json")
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: True,
        process_command_resolver=lambda pid: "python -m search_assistant.cli feishu-long-connection --data-dir live",
    ).run()

    assert report["ok"] is True
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["model_runtime"]["ok"] is True
    assert checks["model_runtime"]["data"]["source"] == "latest_runtime_session"
    assert checks["feishu_credentials"]["ok"] is True
    assert checks["feishu_credentials"]["data"]["source"] == "latest_runtime_session"
    assert checks["runtime_session"]["data"]["process_identity_ok"] is True
    assert any("current shell is missing" in hint.lower() for hint in report["hints"])
    payload = json.dumps(report, ensure_ascii=False)
    assert "secret" not in payload


def test_readiness_rejects_runtime_metadata_when_pid_belongs_to_unrelated_process(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_runtime_session(
        "feishu-long-connection",
        process_id=2468,
        status="started",
        metadata={
            "model_provider": "deepseek",
            "deepseek_api_key_configured": True,
            "allow_fake_runtime": False,
            "feishu_app_id_configured": True,
            "feishu_app_secret_configured": True,
        },
    )
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: True,
        process_command_resolver=lambda pid: "python -m unrelated.worker",
    ).run()

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runtime_session"]["ok"] is False
    assert checks["runtime_session"]["data"]["process_alive"] is True
    assert checks["runtime_session"]["data"]["process_identity_ok"] is False
    assert "does not match feishu-long-connection" in checks["runtime_session"]["detail"]
    assert checks["model_runtime"]["ok"] is False
    assert checks["model_runtime"]["detail"] == "DEEPSEEK_API_KEY is missing"
    assert checks["feishu_credentials"]["ok"] is False
    assert checks["feishu_credentials"]["detail"] == "FEISHU_APP_ID or FEISHU_APP_SECRET is missing"


def test_readiness_rejects_stale_runtime_metadata_when_process_is_dead(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_runtime_session(
        "feishu-long-connection",
        process_id=987654,
        status="started",
        metadata={
            "model_provider": "deepseek",
            "deepseek_api_key_configured": True,
            "allow_fake_runtime": False,
            "feishu_app_id_configured": True,
            "feishu_app_secret_configured": True,
        },
    )
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: False,
    ).run()

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runtime_session"]["ok"] is False
    assert checks["runtime_session"]["data"]["process_alive"] is False
    assert "not running" in checks["runtime_session"]["detail"]
    assert checks["model_runtime"]["ok"] is False
    assert checks["model_runtime"]["detail"] == "DEEPSEEK_API_KEY is missing"
    assert checks["feishu_credentials"]["ok"] is False
    assert checks["feishu_credentials"]["detail"] == "FEISHU_APP_ID or FEISHU_APP_SECRET is missing"
    assert not any("current shell is missing" in hint.lower() for hint in report["hints"])


def test_readiness_uses_latest_feishu_runtime_when_other_runtime_is_newer(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    feishu_runtime_id = store.record_runtime_session(
        "feishu-long-connection",
        process_id=1234,
        status="started",
        metadata={
            "model_provider": "deepseek",
            "deepseek_api_key_configured": True,
            "allow_fake_runtime": False,
            "feishu_app_id_configured": True,
            "feishu_app_secret_configured": True,
        },
    )
    reply_id = store.record_reply_attempt("q-ready", "om-ready", "feishu-long-connection", "success")
    evolve_runtime_id = store.record_runtime_session(
        "evolve-loop",
        process_id=5678,
        status="started",
        metadata={"interval_seconds": 86400.0},
    )
    _set_created_at(store, "runtime_sessions", feishu_runtime_id, "2026-07-03T10:00:00+00:00")
    _set_created_at(store, "reply_attempts", reply_id, "2026-07-03T10:05:00+00:00")
    _set_created_at(store, "runtime_sessions", evolve_runtime_id, "2026-07-03T11:00:00+00:00")
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: True,
        process_command_resolver=lambda pid: "python -m search_assistant.cli feishu-long-connection --data-dir live"
        if pid == 1234
        else "python -m search_assistant.cli evolve-loop --data-dir live",
    ).run()

    checks = {check["name"]: check for check in report["checks"]}
    assert checks["runtime_session"]["data"]["kind"] == "feishu-long-connection"
    assert checks["runtime_session"]["data"]["process_id"] == 1234
    assert checks["model_runtime"]["data"]["process_id"] == 1234
    assert checks["feishu_credentials"]["data"]["process_id"] == 1234
    assert checks["reply_attempt"]["data"]["latest_runtime_created_at"] == "2026-07-03T10:00:00+00:00"


def test_readiness_report_flags_fake_runtime_and_missing_deepseek_key(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    settings = Settings(
        data_dir=tmp_path,
        model_provider="fake",
        allow_fake_runtime=True,
        search_provider="browser",
        browser_search_engines=["bing"],
    )

    report = ReadinessService(store=store, settings=settings, data_dir=tmp_path).run()

    assert report["ok"] is False
    failed = {check["name"]: check for check in report["checks"] if not check["ok"]}
    assert failed["model_runtime"]["detail"] == "production requires SEARCH_ASSISTANT_MODEL_PROVIDER=glm or deepseek"
    assert failed["browser_search"]["detail"] == "browser search should include bing, baidu, and google"
    assert failed["learning_evidence"]["data"]["experience_items"] == 0
    assert report["hints"]


def test_readiness_reply_attempt_requires_long_connection_channel(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_reply_attempt("q-probe", "om-probe", "feishu-probe", "success")
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(store=store, settings=settings, data_dir=tmp_path).run()

    reply_check = next(check for check in report["checks"] if check["name"] == "reply_attempt")
    assert reply_check["ok"] is False
    assert "automatic Feishu reply attempt" in reply_check["detail"]


def test_readiness_reply_attempt_allows_polling_channel(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    store.record_reply_attempt("q-poll", "om-poll", "feishu-polling", "success")
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(store=store, settings=settings, data_dir=tmp_path).run()

    reply_check = next(check for check in report["checks"] if check["name"] == "reply_attempt")
    assert reply_check["ok"] is True
    assert reply_check["data"]["channel"] == "feishu-polling"


def test_readiness_reply_attempt_must_be_after_latest_verified_runtime_session(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    attempt_id = store.record_reply_attempt("q-old", "om-old", "feishu-long-connection", "success")
    runtime_id = store.record_runtime_session(
        "feishu-long-connection",
        process_id=1234,
        status="started",
        metadata={"model_provider": "deepseek"},
    )
    _set_created_at(store, "reply_attempts", attempt_id, "2026-07-03T10:00:00+00:00")
    _set_created_at(store, "runtime_sessions", runtime_id, "2026-07-03T11:00:00+00:00")
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: True,
        process_command_resolver=lambda pid: "python -m search_assistant.cli feishu-long-connection --data-dir live",
    ).run()

    reply_check = next(check for check in report["checks"] if check["name"] == "reply_attempt")
    assert reply_check["ok"] is False
    assert "after latest runtime session" in reply_check["detail"]
    assert reply_check["data"]["latest_runtime_created_at"] == "2026-07-03T11:00:00+00:00"
    assert reply_check["data"]["latest_attempt_created_at"] == "2026-07-03T10:00:00+00:00"
    assert any("new Feishu bot message after restart" in hint for hint in report["hints"])


def test_readiness_reply_attempt_passes_when_success_is_after_latest_verified_runtime_session(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    runtime_id = store.record_runtime_session(
        "feishu-long-connection",
        process_id=1234,
        status="started",
        metadata={"model_provider": "deepseek"},
    )
    attempt_id = store.record_reply_attempt("q-new", "om-new", "feishu-long-connection", "success")
    _set_created_at(store, "runtime_sessions", runtime_id, "2026-07-03T11:00:00+00:00")
    _set_created_at(store, "reply_attempts", attempt_id, "2026-07-03T11:05:00+00:00")
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(
        store=store,
        settings=settings,
        data_dir=tmp_path,
        process_checker=lambda pid: True,
        process_command_resolver=lambda pid: "python -m search_assistant.cli feishu-long-connection --data-dir live",
    ).run()

    reply_check = next(check for check in report["checks"] if check["name"] == "reply_attempt")
    assert reply_check["ok"] is True
    assert reply_check["data"]["message_id"] == "om-new"
    assert reply_check["data"]["latest_runtime_created_at"] == "2026-07-03T11:00:00+00:00"


def test_readiness_evaluation_report_requires_quality_audit_schema(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    evaluations_dir = tmp_path / "evaluations"
    evaluations_dir.mkdir()
    (evaluations_dir / "evaluation-report.json").write_text(
        json.dumps(
            {
                "total_questions": 1,
                "items": [
                    {
                        "question": "old report",
                        "source_count": 1,
                        "calibration_ran": True,
                    }
                ],
                "summary": {
                    "evaluation_report_path": str(evaluations_dir / "evaluation-report.json"),
                },
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(store=store, settings=settings, data_dir=tmp_path).run()

    evaluation_check = next(check for check in report["checks"] if check["name"] == "evaluation_report")
    assert evaluation_check["ok"] is False
    assert "quality audit fields" in evaluation_check["detail"]
    assert any("eval-suite" in hint for hint in report["hints"])


def test_readiness_evaluation_report_requires_uncertainty_assessment_field(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    evaluations_dir = tmp_path / "evaluations"
    evaluations_dir.mkdir()
    _write_quality_evaluation_report(evaluations_dir / "evaluation-report.json", include_uncertainty_assessment=False)
    settings = Settings(
        data_dir=tmp_path,
        model_provider="deepseek",
        deepseek_api_key="secret-key",
        search_provider="browser",
        browser_search_engines=["bing", "baidu", "google"],
        feishu_app_id="cli_secret",
        feishu_app_secret="secret",
    )

    report = ReadinessService(store=store, settings=settings, data_dir=tmp_path).run()

    evaluation_check = next(check for check in report["checks"] if check["name"] == "evaluation_report")
    assert evaluation_check["ok"] is False
    assert "items[0].uncertainty_assessment" in evaluation_check["data"]["missing_fields"]


def _write_quality_evaluation_report(path, include_uncertainty_assessment=True):
    item = {
        "index": 1,
        "question": "What is CXL?",
        "question_id": "q-eval",
        "classification": "research",
        "confidence": "medium",
        "answer_excerpt": "CXL answer.",
        "search_record": "搜索记录:\n- 联网搜索: 已执行",
        "search_record_structured": {
            "executed": True,
            "queries": ["CXL official documentation"],
            "sources": [
                {
                    "title": "CXL official documentation",
                    "url": "https://www.computeexpresslink.org/",
                    "snippet": "CXL official documentation.",
                    "provider": "unit",
                    "checked_at": "2026-07-03T00:00:00Z",
                }
            ],
            "engines": ["bing", "baidu", "google"],
            "skipped_reason": None,
        },
        "source_count": 1,
        "source_urls": ["https://www.computeexpresslink.org/"],
        "verified_claims": 0,
        "unverified_claims": 0,
        "unverified_claim_samples": [],
        "calibration_ran": True,
        "calibration_critique": "checked",
        "review_ran": True,
        "review_approved": True,
        "review_issues": [],
        "memory_updates": 0,
        "quality_flags": [],
    }
    if include_uncertainty_assessment:
        item["uncertainty_assessment"] = "not_applicable"
    path.write_text(
        json.dumps(
            {
                "total_questions": 1,
                "items": [item],
                "summary": {
                    "answers_recorded": 1,
                    "profile_snapshots": 1,
                    "learning_report_written": True,
                    "learning_report_path": str(path.parent.parent / "reports" / "learning-report.md"),
                    "evaluation_report_path": str(path),
                    "total_sources": 1,
                    "total_verified_claims": 0,
                    "total_unverified_claims": 0,
                    "calibrated_answers": 1,
                    "review_rejected_answers": 0,
                    "flagged_answers": 0,
                },
            }
        ),
        encoding="utf-8",
    )


def _set_created_at(store: MemoryStore, table: str, row_id: str, created_at: str) -> None:
    if table not in {"reply_attempts", "runtime_sessions"}:
        raise ValueError(table)
    with store._connect() as connection:
        connection.execute(f"UPDATE {table} SET created_at = ? WHERE id = ?", (created_at, row_id))
