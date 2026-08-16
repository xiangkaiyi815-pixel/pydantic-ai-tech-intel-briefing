# Configuration

## Local Files

Copy `.env.example` to `.env.local` and keep the populated file out of Git.
The application loads `.env.local` only when no explicit environment mapping is
provided. Process environment variables take precedence.

```powershell
Copy-Item .env.example .env.local
```

Use a deployment secret manager for hosted instances. Do not place credentials
in command lines, reports, SQLite fixtures, or checked-in configuration files.

## Required Model Settings

| Variable | Purpose |
| --- | --- |
| `SEARCH_ASSISTANT_MODEL_PROVIDER` | `glm` by default; `deepseek` selects the legacy runtime. |
| `GLM_API_KEY` | Required only for the GLM runtime. |
| `GLM_BASE_URL` | OpenAI-compatible GLM endpoint. |
| `GLM_MODEL` | Default: `glm-4.7`. |
| `GLM_TIMEOUT_SECONDS` | Per-model request time budget. |

The project supports local encrypted configuration tooling in its development
environment, but the public repository intentionally documents only variable
names and never a secret value.

## Retrieval Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `SEARCH_ASSISTANT_SEARCH_PROVIDER` | `hybrid` | `hybrid`, `mcp`, `browser`, `brave`, `searxng`, or `duckduckgo`. |
| `SEARCH_ASSISTANT_MCP_SEARCH_CONFIG` | checked-in default when unset | MCP server and read-only search bindings. |
| `MCP_SEARCH_TIMEOUT_SECONDS` | `18` | MCP request timeout. |
| `BROWSER_SEARCH_ENGINES` | `bing,baidu,google` | Public engines for browser search; `duckduckgo` can also be included. |
| `BROWSER_SEARCH_BASE_URL` | `https://cn.bing.com/search` | Bing public-result endpoint. |
| `BAIDU_SEARCH_BASE_URL` | `https://www.baidu.com/baidu` | Baidu public-index base URL; the client can fall back to other public Baidu endpoints when one is challenged. |
| `BILIBILI_SEARCH_BASE_URL` | public Bilibili endpoint | No-login Bilibili search endpoint. |
| `RSSHUB_BASE_URL` | `http://127.0.0.1:1200` | Local optional RSSHub instance. |
| `SEARCH_ASSISTANT_DOMESTIC_RSS_CONFIG` | unset | Optional operator-owned RSS source catalog. |
| `BRAVE_SEARCH_API_KEY` | unset | Required only when the Brave provider is selected. |
| `SEARXNG_BASE_URL` | `http://localhost:8080/search` | Self-hosted SearXNG JSON endpoint. |

## Daily Briefing Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRIEFING_TIMEZONE` | `Asia/Shanghai` | Local calendar date used for daily report paths and database records. |
| `BRIEFING_MAX_QUERIES` | `28` | Maximum combined planned and deterministic queries. |
| `BRIEFING_RESULTS_PER_QUERY` | `10` | Per-query retrieval limit. |
| `BRIEFING_MAX_SOURCES` | `50` | Full retained source cap. |
| `BRIEFING_MODEL_MAX_SOURCES` | `12` | Ranked evidence cap supplied to GLM. |
| `BRIEFING_SEARCH_BUDGET_SECONDS` | `90` | Concurrent retrieval budget. |
| `BRIEFING_PLANNING_TIMEOUT_SECONDS` | `40` | GLM planning budget. |

## Optional Feishu Settings

Set `SEARCH_ASSISTANT_FEISHU_ENABLED=true` and provide the Feishu app
credentials through your local or deployment secret store. The app supports
webhook parsing, long connection, and polling. Start with `feishu-doctor` and
an isolated fixture before sending a real message.

```powershell
python -m search_assistant.cli feishu-doctor
python -m search_assistant.cli feishu-fixture tests/fixtures/feishu_message_event.json
```

## MCP Configuration

`configs/public-sources.mcp.json` starts two local stdio servers:

- `scripts/public_sources_mcp.py` for GitHub, arXiv, Hacker News, and Stack
  Exchange.
- `scripts/domestic_rss_mcp.py` for a configured local RSSHub instance.

Only add trusted, read-only tools that return URL-bearing result objects. Do
not add write-capable MCP servers to a retrieval binding.
