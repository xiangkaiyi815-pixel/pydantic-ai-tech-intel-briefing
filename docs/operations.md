# Operations

## One-Off Runs

Use an isolated data directory for a test or a separate deployment profile:

```powershell
python -m search_assistant.cli brief-run "industrial AI" --data-dir .local-data\demo
python -m search_assistant.cli doctor --data-dir .local-data\demo
```

Reports are stored under `<data-dir>/briefs/`; sources and briefing metadata are
stored in `<data-dir>/assistant.sqlite3`.

## Scheduled Daily Briefings

For supervised local operation, use the built-in loop:

```powershell
python -m search_assistant.cli brief-loop "industrial AI" --interval-seconds 86400
```

For Windows Task Scheduler, create a task that starts the virtual environment's
Python executable, uses the repository as its working directory, and invokes
`brief-run` once per day. Keep model and Feishu secrets in the task's secure
environment configuration or a local ignored `.env.local`, not in task
arguments.

## Local RSSHub

```powershell
Set-Location infra\rsshub
docker compose up -d
Invoke-WebRequest http://127.0.0.1:1200/healthz
```

The compose file binds the service locally. Validate an individual source before
claiming that a platform is covered for production retrieval.

## SearXNG Or Brave

Use a self-hosted SearXNG instance when public HTML parsing is unreliable, or
select Brave Search when an approved API credential is available. Both remain
subject to the upstream engines' coverage and access constraints.

## Feishu

Start with a fixture replay, then run the readiness checks. A green local test
does not prove the live tenant callback, permissions, or message delivery.

```powershell
python -m search_assistant.cli feishu-fixture tests/fixtures/feishu_message_event.json
python -m search_assistant.cli feishu-doctor
python -m search_assistant.cli feishu-long-connection
```

## Health And Evidence

`doctor` does not call external APIs. It reports configuration presence and
local evidence such as runtime sessions, reply attempts, reports, evaluation
records, and verified claims. Missing Feishu credentials or a missing runtime
session are explicit failed checks, not evidence that the briefing workflow is
broken.
