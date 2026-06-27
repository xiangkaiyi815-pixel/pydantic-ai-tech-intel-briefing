# Code Manual

## Module Map

- `search_assistant.contracts`: Pydantic data contracts shared across the system.
- `search_assistant.config`: environment-backed settings.
- `search_assistant.memory.store`: SQLite persistence and append-first memory behavior.
- `search_assistant.verification.policy`: deterministic claim verification and calibration trigger rules.
- `search_assistant.workflow.runtime`: fake runtime plus Microsoft Agent Framework adapter boundary.
- `search_assistant.workflow.service`: main answer workflow.
- `search_assistant.feishu.events`: Feishu event parsing and challenge handling.
- `search_assistant.feishu.client`: fake and production Feishu client boundaries.
- `search_assistant.server`: FastAPI app factory.
- `search_assistant.profile.service`: user profile snapshots.
- `search_assistant.reports.service`: markdown learning reports.
- `search_assistant.skills.service`: reviewable skill draft generation.
- `search_assistant.cli`: local operation commands.

## Data Contracts

`IncomingMessage` normalizes CLI and Feishu messages. `AnswerPackage` is the core response envelope and always includes classification, confidence, verification lists, calibration data, and memory updates.

## Workflow Steps

1. Check for a stored answer by dedupe key.
2. Persist the incoming interaction.
3. Classify the question.
4. Retrieve memory context.
5. Generate a draft through `AgentRuntime`.
6. Extract fragile claims and mark unverified claims when no verifier is configured.
7. Run calibration for hard, research, high-stakes, or unverified answers.
8. Persist the answer and memory updates.

## Extension Points

- Replace `FakeAgentRuntime` with `MicrosoftAgentRuntime` plus a configured model provider.
- Extend `verification.policy` with real search and source-backed verdicts.
- `FeishuHttpClient.reply_text` already performs tenant-token retrieval and text replies. Extend it for encrypted callbacks, retry policy, token expiry refresh, and richer message types.
- Add review commands for promoting `skills/drafts/<name>/SKILL.md` into active skills.

## Feishu Binding Checklist

1. Deploy or tunnel the FastAPI app so Feishu can reach `POST /feishu/events` over HTTPS.
2. Set `SEARCH_ASSISTANT_FEISHU_ENABLED=true`, `FEISHU_APP_ID`, and `FEISHU_APP_SECRET`.
3. In Feishu Open Platform, enable the bot, configure the event request URL, subscribe to `im.message.receive_v1`, and grant the permissions needed to receive and reply to messages.
4. Run `python -m uvicorn search_assistant.server:create_app --factory --host 0.0.0.0 --port 8000`.
5. Send a bot message and confirm the reply includes `classification` and `confidence`.

## Development Rules

- Write tests first for behavior changes.
- Keep network and model calls behind interfaces.
- Do not commit secrets or local SQLite files.
- Generated skill drafts require human review before activation.
