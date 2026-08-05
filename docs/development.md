# Development Guide

## Install

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

## Test Commands

```powershell
$env:PYTHONUTF8 = "1"
py -3.12 -m pytest -q
py -3.12 -m pytest -q tests/test_briefing_service.py tests/test_search_provider.py
```

Use a temporary pytest cache when the repository owner differs from the active
Windows user:

```powershell
py -3.12 -m pytest -q -o cache_dir=$env:TEMP\pydantic-ai-briefing-pytest
```

## What Tests Prove

- Data models, configuration loading, SQLite behavior, and CLI contracts.
- Search parser behavior, domain enforcement, MCP result normalization, and
  known noisy-result regressions.
- Briefing report sections, original URLs, model JSON validation, timeout retry,
  source filtering, and free-form detailed synthesis rendering.
- Feishu event parsing, formatting, polling, and local adapter behavior.

## What Tests Do Not Prove

- Availability or ranking stability of public search engines and social sites.
- A live model provider's latency, quota, or response quality for every topic.
- Production Feishu tenant permissions, callback delivery, or message receipt.
- A configured RSSHub route's continued public availability.

Run a real briefing in an isolated data directory after changing model prompts,
search filters, provider timeouts, or platform configuration. Inspect the report
for source count, provider mix, original links, malformed search text, and the
quality of the detailed synthesis before calling the change production-ready.

## Pull Requests

Keep changes focused. Add a regression test for a bug before or alongside the
fix. Do not commit `.env.local`, SQLite databases, generated reports, credentials,
cookies, or raw external API responses.
