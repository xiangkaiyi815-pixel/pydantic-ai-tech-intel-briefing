import importlib
import importlib.util

from search_assistant.contracts import AnswerPackage, SearchRecord, SourceEvidence


def test_answer_package_to_post_strips_markdown_and_keeps_links():
    formatting = _formatting_module()
    package = _answer_package(
        answer_text=(
            "## Short answer\n"
            "**CXL** is a cache-coherent interconnect.\n\n"
            "- Memory pooling\n"
            "- [Spec page](https://example.com/cxl)\n"
        )
    )

    post = formatting.answer_package_to_post(package)

    assert post["zh_cn"]["title"] == "搜索助手回答"
    flattened_text = _flatten_post_text(post)
    assert "Short answer" in flattened_text
    assert "CXL is a cache-coherent interconnect." in flattened_text
    assert "• Memory pooling" in flattened_text
    assert "##" not in flattened_text
    assert "**" not in flattened_text
    assert any(
        element.get("tag") == "a"
        and element.get("text") == "Spec page"
        and element.get("href") == "https://example.com/cxl"
        for line in post["zh_cn"]["content"]
        for element in line
    )


def test_answer_package_to_post_adds_compact_search_record():
    formatting = _formatting_module()
    package = _answer_package(answer_text="A sourced answer.")

    post = formatting.answer_package_to_post(package)

    flattened_text = _flatten_post_text(post)
    assert "搜索记录" in flattened_text
    assert "bing / google" in flattened_text
    assert "query one" in flattened_text
    assert "Example Source" in flattened_text
    assert "classification: hard" in flattened_text
    assert "confidence: medium" in flattened_text


def _formatting_module():
    module_name = "search_assistant.feishu.formatting"
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, "Feishu formatting module should exist"
    return importlib.import_module(module_name)


def _answer_package(answer_text):
    source = SourceEvidence(
        title="Example Source",
        url="https://example.com/source",
        snippet="source snippet",
        provider="browser",
        checked_at="2026-07-04T00:00:00Z",
    )
    return AnswerPackage(
        question_id="q_1",
        answer_text=answer_text,
        classification="hard",
        confidence="medium",
        sources=[source],
        search_record=SearchRecord(
            executed=True,
            queries=["query one"],
            engines=["bing", "google"],
            sources=[source],
        ),
    )


def _flatten_post_text(post):
    return "\n".join(
        str(element.get("text", ""))
        for line in post["zh_cn"]["content"]
        for element in line
    )
