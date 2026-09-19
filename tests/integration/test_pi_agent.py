"""Exercise the real Pi binary against an OpenAI-compatible test endpoint, without API spend."""

from __future__ import annotations

import json
import shutil
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from servepilot.optimization.agent import AgentTool, PiAgent
from servepilot.optimization.budget import TimeBudget
from servepilot.optimization.schemas import AgentConfig
from servepilot.runtime.ports import ephemeral_port
from servepilot.runtime.server import HTTPServer


@pytest.mark.skipif(shutil.which("pi") is None, reason="real Pi CLI is not installed")
async def test_real_pi_uses_only_controller_tools_through_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "proxy-test-key")
    app = FastAPI()
    requests: list[dict[str, Any]] = []
    invocations = []

    @app.post("/v1/chat/completions")
    async def completions(request: Request) -> StreamingResponse:
        assert request.headers["authorization"] == "Bearer proxy-test-key"
        body = await request.json()
        requests.append(body)
        answered = any(message["role"] == "tool" for message in body["messages"])

        async def events():  # type: ignore[no-untyped-def]
            base = {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-engineer",
            }
            delta = (
                {"content": "The cluster has 2 GPUs."}
                if answered
                else {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_inspect",
                            "type": "function",
                            "function": {"name": "inspect_cluster", "arguments": "{}"},
                        }
                    ],
                }
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", **delta},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + "\n\n"
            )
            yield (
                "data: "
                + json.dumps(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop" if answered else "tool_calls",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                        },
                    }
                )
                + "\n\n"
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    async def inspect(arguments: dict[str, Any]) -> dict[str, int]:
        invocations.append(arguments)
        return {"gpus": 2}

    server = HTTPServer(app, "127.0.0.1", ephemeral_port())
    await server.start()
    try:
        agent = PiAgent(AgentConfig(base_url=server.base_url + "/v1", model="test-engineer"))
        turn = await agent.turn(
            {"task": "Inspect the cluster and describe it."},
            [
                AgentTool(
                    "inspect_cluster",
                    "Inspect available GPU nodes.",
                    {"type": "object", "properties": {}, "additionalProperties": False},
                    inspect,
                )
            ],
            TimeBudget(40),
        )
    finally:
        await server.stop()
    assert turn.tool_calls == 1 and invocations == [{}]
    assert "2 GPUs" in turn.text
    assert len(requests) == 2
    assert {tool["function"]["name"] for tool in requests[0]["tools"]} == {"inspect_cluster"}
    assert "proxy-test-key" not in json.dumps(turn.events)
