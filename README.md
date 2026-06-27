# Search Assistant

Feishu-connected search assistant powered by Microsoft Agent Framework. The Phase 1 MVP receives local CLI questions or Feishu-style event payloads, produces structured answer packages, records interactions in SQLite, runs conservative verification/calibration gates, generates learning reports, and creates reviewable skill drafts.

## Architecture

- `feishu`: parses Feishu callback events and sends replies through a client adapter.
- `workflow`: orchestrates answer generation, claim extraction, verification policy, calibration, and persistence.
- `memory`: stores interactions, answers, evidence, memory items, profile snapshots, reports, and skill drafts in SQLite.
- `profile`: turns answer history into compact user-profile snapshots.
- `reports`: exports learning reports as markdown.
- `skills`: creates draft `SKILL.md` files for human review.

Microsoft Agent Framework is integrated behind `MicrosoftAgentRuntime` in `search_assistant.workflow.runtime`. Tests and local commands use `FakeAgentRuntime` so the MVP is deterministic without live credentials.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Copy `.env.example` to `.env.local` for local notes. Do not commit real Feishu or model secrets.

## Local Commands

```powershell
python -m search_assistant.cli ask "What should be verified before answering current API questions?"
python -m search_assistant.cli report
python -m search_assistant.cli skill-draft "Reliable API Answers"
python -m search_assistant.cli feishu-fixture tests/fixtures/feishu_message_event.json
```

Use `--data-dir <path>` on any command to isolate local data.

## Feishu Integration

Configure a Feishu bot application with an event callback URL pointing at:

```text
POST /feishu/events
```

Phase 1 supports URL challenge responses, `im.message.receive_v1` events, and live text replies through Feishu's tenant access token and message reply APIs.

For a local tunnel or deployed server:

```powershell
$env:PYTHONPATH = "src"
$env:SEARCH_ASSISTANT_FEISHU_ENABLED = "true"
$env:FEISHU_APP_ID = "cli_xxx"
$env:FEISHU_APP_SECRET = "xxx"
python -m uvicorn search_assistant.server:create_app --factory --host 0.0.0.0 --port 8000
```

Then configure the public HTTPS URL in Feishu:

```text
https://<your-domain>/feishu/events
```

The Feishu app must enable bot capabilities, subscribe to `im.message.receive_v1`, and grant the message permissions required by Feishu for receiving and replying as a bot.

## Verification Model

The assistant classifies questions as `simple`, `research`, `hard`, or `high_stakes`. Research, hard, high-stakes, current, versioned, numeric, policy, API, and other fragile claims require verification. If a claim cannot be verified in Phase 1 local mode, it is recorded as unverified and confidence is downgraded.

## Development

Run tests:

```powershell
python -m pytest -q
```

Design and implementation plan:

- `docs/superpowers/specs/2026-06-26-search-assistant-design.md`
- `docs/superpowers/plans/2026-06-26-search-assistant-mvp.md`

## Phase 1 Limits

- Generated skills are drafts only and are not auto-enabled.
- Scheduled reports are represented by a CLI command.
- Encrypted Feishu callbacks and production model/search providers remain later-phase work.
- The default local answer runtime is deterministic and fake. Bind a real model/search runtime before relying on answer quality in daily use.
