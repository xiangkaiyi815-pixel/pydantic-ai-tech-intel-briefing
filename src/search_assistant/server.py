from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from search_assistant.feishu.client import FeishuClient, FakeFeishuClient
from search_assistant.feishu.events import handle_challenge, parse_feishu_event
from search_assistant.memory.store import MemoryStore
from search_assistant.workflow.runtime import AgentRuntime, FakeAgentRuntime
from search_assistant.workflow.service import SearchAssistantWorkflow


def create_app(
    data_dir: str | Path | None = None,
    store: MemoryStore | None = None,
    runtime: AgentRuntime | None = None,
    feishu_client: FeishuClient | None = None,
) -> FastAPI:
    app = FastAPI(title="Search Assistant")
    base_dir = Path(data_dir or ".local-data")
    memory_store = store or MemoryStore(base_dir / "assistant.sqlite3")
    memory_store.initialize()
    workflow = SearchAssistantWorkflow(store=memory_store, runtime=runtime or FakeAgentRuntime())
    client = feishu_client or FakeFeishuClient()

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/feishu/events")
    async def feishu_events(request: Request) -> dict[str, Any]:
        payload = await request.json()
        challenge = handle_challenge(payload)
        if challenge is not None:
            return challenge
        try:
            message = parse_feishu_event(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        package = workflow.answer(message)
        reply = client.reply_text(message.message_id, _format_feishu_reply(package.model_dump(mode="json")))
        return {"ok": True, "reply": reply}

    return app


def _format_feishu_reply(package: dict[str, Any]) -> str:
    return (
        f"{package['answer_text']}\n\n"
        f"classification: {package['classification']}\n"
        f"confidence: {package['confidence']}"
    )
