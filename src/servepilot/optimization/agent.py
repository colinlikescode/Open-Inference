"""Pi CLI integration using a dedicated LiteLLM OpenAI-compatible provider.

Pi's built-in shell/filesystem tools, discovered extensions and context files are disabled.
It can only invoke explicitly registered controller tools. Those tools expose experiment
containers and read-only copies of measurements, never the controller's filesystem.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from servepilot.cluster.transport import SSHTransport
from servepilot.exceptions import ConfigurationError, ServePilotError
from servepilot.logging import redact_secrets
from servepilot.optimization.budget import TimeBudget
from servepilot.optimization.schemas import AgentConfig, canonical_json
from servepilot.optimization.store import atomic_write
from servepilot.runtime.ports import ephemeral_port
from servepilot.runtime.server import HTTPServer

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass(frozen=True)
class AgentTurn:
    text: str
    events: list[dict[str, Any]]
    tool_calls: int


SYSTEM_PROMPT = """You are Pi, the inference performance engineer inside Open BaseTen.
Optimize the user's model on their existing GPUs within the remaining experiment budget.
The objective, constraints, correctness suite, measurements and history are owned by the
controller. You cannot change them or declare your own experiment successful.
Inspect the current best, failed experiments, hardware, network and profiler data. Form a
specific hypothesis. You may change serving configuration, distributed topology, engine
source code, Triton/CUDA kernels and dependencies using the provided experiment tools.
All shell/file operations apply only to the isolated experimental runtime. No host access.
Submit each proposal through the controller's experiment tools for independent correctness
and repeated performance verification. Use the returned evidence to decide what to try next.
Preserve the required workload and context length. Do not overstate what measurements prove.
When the controller asks for one proposal, submit one complete proposal and then stop.
"""


class PiAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def preflight(self) -> None:
        if shutil.which(self.config.executable) is None:
            raise ConfigurationError(
                "Pi is not installed on the controller",
                hints=[
                    "Install Node.js 22.19+ and run: npm install -g @earendil-works/pi-coding-agent@0.83.0"
                ],
            )
        if not os.environ.get(self.config.api_key_env):
            raise ConfigurationError(
                f"set {self.config.api_key_env} to the API key for your LiteLLM endpoint",
                hints=[
                    "The key is passed through the environment and is not written into reports or recipes."
                ],
            )

    def _redact(self, text: str) -> str:
        value = os.environ.get(self.config.api_key_env)
        if value:
            text = text.replace(value, "***")
        return redact_secrets(text)

    def models_config(self) -> dict[str, Any]:
        return {
            "providers": {
                "openbaseten-litellm": {
                    "baseUrl": self.config.base_url,
                    "api": "openai-completions",
                    "apiKey": "$OPENBASETEN_PI_API_KEY",
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                        "supportsStore": False,
                    },
                    "models": [
                        {
                            "id": self.config.model,
                            "name": self.config.model,
                            "reasoning": False,
                            "input": ["text"],
                            "contextWindow": self.config.context_window,
                            "maxTokens": self.config.max_tokens,
                        }
                    ],
                }
            }
        }

    async def turn(
        self,
        context: dict[str, Any],
        tools: list[AgentTool],
        budget: TimeBudget,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> AgentTurn:
        self.preflight()
        if len({tool.name for tool in tools}) != len(tools):
            raise ValueError("agent tool names must be unique")
        handlers = {tool.name: tool.handler for tool in tools}
        token = secrets.token_urlsafe(32)
        calls = 0
        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        # Serialize tool effects: one candidate owns the GPUs at a time.
        tool_lock = asyncio.Lock()

        @app.post("/tools/{name}")
        async def dispatch(name: str, request: Request) -> Any:
            nonlocal calls
            if not secrets.compare_digest(
                request.headers.get("Authorization", ""), f"Bearer {token}"
            ):
                raise HTTPException(403, "invalid tool capability")
            if name not in handlers:
                raise HTTPException(404, "tool is not available")
            body = await request.body()
            if len(body) > 2 * 1024 * 1024:
                raise HTTPException(413, "tool input is too large")
            try:
                arguments = json.loads(body)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                async with tool_lock:
                    result = await budget.run(lambda: handlers[name](arguments))
                calls += 1
                return result
            except (ValueError, KeyError, ServePilotError) as exc:
                raise HTTPException(
                    400,
                    self._redact(exc.render() if isinstance(exc, ServePilotError) else str(exc)),
                ) from exc

        server = HTTPServer(app, "127.0.0.1", ephemeral_port())
        proc: asyncio.subprocess.Process | None = None
        readers: list[asyncio.Task[None]] = []
        events: list[dict[str, Any]] = []
        texts: list[str] = []
        stderr: list[str] = []
        async with contextlib.AsyncExitStack():
            with tempfile.TemporaryDirectory(prefix="openbaseten-pi-") as temporary:
                directory = Path(temporary)
                atomic_write(
                    directory / "models.json", canonical_json(self.models_config()).encode()
                )
                atomic_write(
                    directory / "settings.json",
                    b'{"compaction":{"enabled":false},"retry":{"enabled":false}}',
                )
                env = {
                    key: value
                    for key, value in os.environ.items()
                    if key
                    in (
                        "PATH",
                        "HOME",
                        "LANG",
                        "LC_ALL",
                        "TMPDIR",
                        "SSL_CERT_FILE",
                        "NODE_EXTRA_CA_CERTS",
                        "HTTPS_PROXY",
                        "HTTP_PROXY",
                        "NO_PROXY",
                    )
                }
                env.update(
                    PI_CODING_AGENT_DIR=str(directory),
                    OPENBASETEN_PI_API_KEY=os.environ[self.config.api_key_env],
                    OPENBASETEN_AGENT_BRIDGE=server.base_url,
                    OPENBASETEN_AGENT_BRIDGE_TOKEN=token,
                    OPENBASETEN_AGENT_TOOLS=canonical_json([tool.schema() for tool in tools]),
                )
                extension = Path(__file__).with_name("agent-extension.mjs")
                argv = [
                    self.config.executable,
                    "--mode",
                    "json",
                    "--print",
                    "--no-session",
                    "--no-builtin-tools",
                    "--no-extensions",
                    "--no-skills",
                    "--no-prompt-templates",
                    "--no-themes",
                    "--no-context-files",
                    "--extension",
                    str(extension),
                    "--provider",
                    "openbaseten-litellm",
                    "--model",
                    self.config.model,
                    "--thinking",
                    "off",
                    "--system-prompt",
                    SYSTEM_PROMPT,
                ]
                try:
                    await server.start()
                    proc = await asyncio.create_subprocess_exec(
                        *argv,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        env=env,
                        cwd=directory,
                        start_new_session=True,
                        limit=4 * 1024 * 1024,
                    )
                    assert (
                        proc.stdin is not None
                        and proc.stdout is not None
                        and proc.stderr is not None
                    )
                    proc.stdin.write(canonical_json(context).encode() + b"\n")
                    await proc.stdin.drain()
                    proc.stdin.close()

                    async def read_events() -> None:
                        assert proc is not None and proc.stdout is not None
                        total = 0
                        while line := await proc.stdout.readline():
                            total += len(line)
                            if total > 32 * 1024 * 1024:
                                raise ConfigurationError("Pi event output exceeded the turn limit")
                            try:
                                event = json.loads(self._redact(line.decode(errors="replace")))
                            except ValueError:
                                continue
                            if not isinstance(event, dict):
                                continue
                            # Incremental text events can be very large; final messages retain it.
                            if event.get("type") == "message_update":
                                continue
                            events.append(event)
                            if on_event is not None:
                                on_event(event)
                            if event.get("type") == "message_end":
                                message = event.get("message", {})
                                if message.get("role") == "assistant":
                                    texts.extend(
                                        block["text"]
                                        for block in message.get("content", [])
                                        if block.get("type") == "text"
                                    )

                    async def read_stderr() -> None:
                        assert proc is not None and proc.stderr is not None
                        while line := await proc.stderr.readline():
                            stderr.append(self._redact(line.decode(errors="replace").rstrip()))
                            del stderr[:-100]

                    readers = [
                        asyncio.create_task(read_events()),
                        asyncio.create_task(read_stderr()),
                    ]

                    async def finish() -> None:
                        assert proc is not None
                        await asyncio.gather(proc.wait(), *readers)

                    await budget.run(finish, limit=self.config.turn_timeout_seconds)
                    if proc.returncode:
                        raise ConfigurationError(
                            f"Pi exited with status {proc.returncode}", hints=stderr[-8:]
                        )
                    errors = [
                        e["message"].get("errorMessage")
                        for e in events
                        if e.get("type") == "message_end"
                        and e.get("message", {}).get("stopReason") == "error"
                    ]
                    if errors:
                        raise ConfigurationError(
                            f"Pi model request failed: {self._redact(str(errors[-1]))}"
                        )
                    return AgentTurn(text="\n".join(texts), events=events, tool_calls=calls)
                finally:
                    if proc is not None:
                        await SSHTransport.stop_process(proc)
                    for reader in readers:
                        reader.cancel()
                    await asyncio.gather(*readers, return_exceptions=True)
                    await server.stop()
