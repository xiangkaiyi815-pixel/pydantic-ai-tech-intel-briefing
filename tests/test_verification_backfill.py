from search_assistant.contracts import AnswerPackage, SourceEvidence
from search_assistant.memory.store import MemoryStore
from search_assistant.verification.backfill import VerificationBackfillService


def test_verification_backfill_persists_verified_claims_from_historical_answers(tmp_path):
    store = MemoryStore(tmp_path / "assistant.sqlite3")
    store.initialize()
    package = AnswerPackage(
        question_id="q-historical",
        answer_text=(
            "### 证据核查\n"
            "\n"
            "| 来源 | 确认了什么 | 缺少什么 |\n"
            "|------|-----------|----------|\n"
            "| NVIDIA DGX Spark 官方规格 | 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡 | 无 DeepSeek-V4 性能实测 |\n"
        ),
        classification="research",
        confidence="medium",
        sources=[
            SourceEvidence(
                title="NVIDIA DGX Spark 官方规格",
                url="https://www.nvidia.com/en-us/products/workstations/dgx-spark/",
                snippet="NVIDIA DGX Spark 官方规格确认 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡。",
                provider="direct-official",
                checked_at="2026-07-03T00:00:00Z",
            )
        ],
    )
    store.record_answer(package)

    result = VerificationBackfillService(store).run()
    second_result = VerificationBackfillService(store).run()

    stored = store.list_answers()[0]
    evidence = store.list_evidence()
    assert result == {
        "answers_scanned": 1,
        "answers_with_sources": 1,
        "answers_updated": 1,
        "verified_claims": 1,
        "unverified_claims": 0,
        "evidence_inserted": 1,
    }
    assert second_result["evidence_inserted"] == 0
    assert [claim.claim for claim in stored.verified_claims] == [
        "NVIDIA DGX Spark 官方规格 确认 128GB 统一内存、273GB/s 带宽、ConnectX-7 网卡"
    ]
    assert len(evidence) == 1
    assert evidence[0]["source"] == "https://www.nvidia.com/en-us/products/workstations/dgx-spark/"
