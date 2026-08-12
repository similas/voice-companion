"""
Tool integrations — one module per external service, discovered by credentials.

DESIGN
------
A tool is a plain (OpenAI function schema, async handler) pair. Each module in
this package declares its tools with the @tool decorator and an enabled() check
that looks at .env. An integration whose credentials are missing simply does
not exist: its schema is never sent to the model, so it costs no prefill and
can never be called. Adding an integration = adding a file, nothing else.

Schemas are kept TERSE on purpose. They are prompt prefill — measured here at
~3.2 ms per token — and although llama-server's prefix cache absorbs the cost
after the first turn, every token still counts against the 2048 ctx budget.

Credentials live in .env at the project root (gitignored). Nothing else in the
project reads .env, and no credential ever appears in config.yaml or code.
"""

import asyncio
import json
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Tool:
    """One callable the LLM may invoke."""

    schema: dict
    run: Callable[..., Awaitable[str]]

    @property
    def name(self) -> str:
        return self.schema["function"]["name"]


def tool(name: str, description: str, parameters: dict = None):
    """Declare an async function as a Tool. The handler returns a SHORT string —
    it is fed back to the model, which phrases it as speech."""
    def wrap(fn) -> Tool:
        return Tool(schema={"type": "function", "function": {
            "name": name, "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        }}, run=fn)
    return wrap


def load_env(path: Path = ROOT / ".env") -> None:
    """Tiny .env loader (KEY=VALUE lines, # comments) — not worth a dependency.
    Real environment variables win over the file."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


async def api(url: str, *, method: str = "GET", headers: dict = None,
              data: dict = None, form: dict = None, timeout: float = 10) -> Any:
    """The one HTTP helper every integration shares. `data` is a JSON body,
    `form` is urlencoded. Returns parsed JSON, or {} for empty bodies (the
    Spotify player endpoints answer 204). Blocking urllib is pushed to a worker
    thread so a slow API cannot stall the event loop that feeds audio."""
    def go():
        body, hdrs = None, dict(headers or {})
        # A real User-Agent, always. WHOOP sits behind Cloudflare, which blocks
        # Python's default UA outright — 403 "error code: 1010" before the
        # request ever reaches the API. Cost nothing for the others.
        hdrs.setdefault("User-Agent", "roomi/0.4 (Jetson Orin Nano)")
        if form is not None:
            body = urllib.parse.urlencode(form).encode()
            hdrs.setdefault("Content-Type", "application/x-www-form-urlencoded")
        elif data is not None:
            body = json.dumps(data).encode()
            hdrs.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(url, data=body, headers=hdrs, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw.strip() else {}
    return await asyncio.to_thread(go)


def load() -> list:
    """Load .env, then return the tools of every enabled integration."""
    load_env()
    from integrations import govee, spotify, whoop

    tools = []
    for mod in (govee, spotify, whoop):
        short = mod.__name__.rsplit(".", 1)[-1]
        if mod.enabled():
            tools += mod.TOOLS
            logger.info(f"tools: {short} on "
                        f"({', '.join(t.name for t in mod.TOOLS)})")
        else:
            logger.info(f"tools: {short} off — credentials not in .env")

    # The WHOOP x Govee collaboration needs both ends live.
    if whoop.enabled() and govee.enabled():
        tools.append(whoop.recovery_light)
        logger.info("tools: recovery_light on (whoop + govee)")

    return tools
