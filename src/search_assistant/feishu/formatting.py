from __future__ import annotations

import re
from typing import Any

from search_assistant.contracts import AnswerPackage, SearchRecord, SourceEvidence


PostContent = dict[str, Any]

_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_MAX_ANSWER_LINES = 36
_MAX_SOURCES = 6
_MAX_LINE_CHARS = 900


def answer_package_to_post(package: AnswerPackage) -> PostContent:
    content: list[list[dict[str, str]]] = []
    for elements in _markdown_to_post_lines(package.answer_text):
        content.append(elements)
        if len(content) >= _MAX_ANSWER_LINES:
            content.append([_text("...")])
            break

    if content:
        content.append([_text(" ")])
    _append_search_record(content, package)
    content.append([_text(f"classification: {package.classification}")])
    content.append([_text(f"confidence: {package.confidence}")])

    return {
        "zh_cn": {
            "title": "搜索助手回答",
            "content": content or [[_text("没有生成可发送的回答。")]],
        }
    }


def package_to_plain_text(package: AnswerPackage) -> str:
    return (
        f"{package.answer_text}\n\n"
        f"classification: {package.classification}\n"
        f"confidence: {package.confidence}"
    )


def reply_answer_package(feishu_client: Any, message_id: str, package: AnswerPackage) -> dict[str, object]:
    reply_post = getattr(feishu_client, "reply_post", None)
    if callable(reply_post):
        return reply_post(message_id, answer_package_to_post(package))
    return feishu_client.reply_text(message_id, package_to_plain_text(package))


def _markdown_to_post_lines(markdown: str) -> list[list[dict[str, str]]]:
    lines: list[list[dict[str, str]]] = []
    inside_fence = False
    for raw_line in markdown.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("```"):
            inside_fence = not inside_fence
            continue
        if inside_fence:
            clean_code = _trim(stripped)
            if clean_code:
                lines.append([_text(clean_code)])
            continue
        elements = _line_to_elements(stripped)
        if elements:
            lines.append(elements)
    return lines


def _line_to_elements(line: str) -> list[dict[str, str]]:
    if not line:
        return []
    line = re.sub(r"^#{1,6}\s*", "", line)
    line = re.sub(r"^>\s?", "", line)
    bullet = re.match(r"^[-*+]\s+(.*)$", line)
    if bullet:
        line = f"• {bullet.group(1)}"
    checkbox = re.match(r"^• \[[ xX]\]\s+(.*)$", line)
    if checkbox:
        line = f"• {checkbox.group(1)}"
    line = _trim(line)
    if not line:
        return []
    return _split_markdown_links(line)


def _split_markdown_links(line: str) -> list[dict[str, str]]:
    elements: list[dict[str, str]] = []
    cursor = 0
    for match in _LINK_PATTERN.finditer(line):
        prefix = _clean_inline_markdown(line[cursor : match.start()])
        if prefix:
            elements.append(_text(prefix))
        label = _clean_inline_markdown(match.group(1)) or match.group(2)
        elements.append({"tag": "a", "text": _trim(label, 80), "href": match.group(2)})
        cursor = match.end()
    suffix = _clean_inline_markdown(line[cursor:])
    if suffix:
        elements.append(_text(suffix))
    return elements


def _append_search_record(content: list[list[dict[str, str]]], package: AnswerPackage) -> None:
    record = package.search_record
    content.append([_text("搜索记录")])
    if record and record.executed:
        if record.engines:
            content.append([_text(f"搜索引擎: {' / '.join(record.engines)}")])
        if record.queries:
            content.append([_text(f"检索词: {'; '.join(record.queries[:4])}")])
        _append_source_lines(content, _sources_for_record(record, package))
        return

    reason = record.skipped_reason if record else "no_search_record"
    content.append([_text(f"未联网检索: {reason or 'unknown'}")])
    _append_source_lines(content, package.sources)


def _sources_for_record(record: SearchRecord, package: AnswerPackage) -> list[SourceEvidence]:
    return list(record.sources or package.sources)


def _append_source_lines(content: list[list[dict[str, str]]], sources: list[SourceEvidence]) -> None:
    if not sources:
        content.append([_text("来源: 未返回可用链接")])
        return
    seen: set[str] = set()
    shown = 0
    for source in sources:
        if not source.url or source.url in seen:
            continue
        seen.add(source.url)
        shown += 1
        title = _clean_inline_markdown(source.title) or source.url
        content.append(
            [
                _text(f"来源 {shown}: "),
                {"tag": "a", "text": _trim(title, 90), "href": source.url},
            ]
        )
        if shown >= _MAX_SOURCES:
            break


def _clean_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"\*([^*\n]+)\*", r"\1", text)
    text = re.sub(r"_([^_\n]+)_", r"\1", text)
    text = text.replace("\\", "")
    return " ".join(text.split())


def _text(text: str) -> dict[str, str]:
    return {"tag": "text", "text": _trim(text)}


def _trim(text: str, max_chars: int = _MAX_LINE_CHARS) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1] + "..."
