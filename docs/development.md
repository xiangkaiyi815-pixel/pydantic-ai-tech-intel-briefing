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
py -3.12 -m pytest -q tests/test_source_registry.py tests/test_memory_store.py
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
- Source capability contracts, provider trace events, candidate lifecycle
  reasons, feedback signal typing, and layered memory persistence.
- Feishu event parsing, formatting, polling, and local adapter behavior.

## Local Source Observability Checks

After a real `brief-run`, inspect the replay artifacts before changing filters
or provider settings:

```powershell
search-assistant source-contracts
search-assistant provider-trace-list --limit 20
search-assistant candidate-list --limit 20
search-assistant memory-layer-list --limit 20
```

`brief-run` writes both a Markdown report and a JSON sidecar.  The sidecar is
the stable replay artifact for comparing provider coverage, accepted sources,
and rejected candidate reasons across PRs.

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
