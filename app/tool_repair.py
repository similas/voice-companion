"""
Catch tool calls the model wrote as TEXT instead of invoking.

gemma-4-E2B at conversational temperature sometimes emits the call into the
content channel: the whole reply is literally 'whoop_sleep' or 'music pause'.
Measured across 8 eval reps at temperatures 0.4-0.7: roughly 1 in 10 tool
requests, and neither prompt wording nor temperature eliminated it. Spoken
aloud that is gibberish, so it gets a mechanical net rather than more prompt
pleading.

HOW: sit between the LLM and TTS. Hold text chunks only while the accumulated
reply still looks like a bare lowercase tool name ("whoo", "music pau" ...) —
a normal sentence diverges on the first chunk and flushes with no added
latency ("Music is a wonderful topic" starts uppercase). If the COMPLETE
reply turns out to be a verbalized call, execute the real tool and speak its
result instead — every integration already returns a speakable string, so
"how did I sleep" degrades to hearing "slept 8.9 hours, sleep performance
82%" rather than the word "whoop underscore sleep".
"""

import re
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# What a verbalized call looks like: the tool name, optionally "()" and/or a
# couple of bare argument words. Anything longer is a real sentence.
_MAX_HOLD_CHARS = 40

# Spoken-word -> music action, for "music pause" / "music continue".
_MUSIC_WORDS = {"play": "resume", "resume": "resume", "continue": "resume",
                "pause": "pause", "stop": "pause", "next": "next",
                "skip": "next", "previous": "previous",
                "now_playing": "now_playing", "song": "now_playing"}


class ToolCallRepair(FrameProcessor):
    """Executes tool calls that leaked into the text channel."""

    def __init__(self, tools):
        super().__init__()
        self._tr_tools = {t.name: t for t in tools}
        self._tr_buf: list = []
        self._tr_hold = False

    def _still_tool_like(self, text: str) -> bool:
        if len(text) > _MAX_HOLD_CHARS or "\n" in text:
            return False
        if not re.match(r"^[a-z_]", text):
            return False
        head = text.split()[0].rstrip("().,!") if text.split() else text
        return any(n.startswith(head) or head.startswith(n)
                   for n in self._tr_tools)

    async def _execute(self, text: str) -> Optional[str]:
        m = re.match(r"^([a-z_]+)\s*(?:\(\s*\))?\s*(.*?)[.!]?$", text)
        if not m or m.group(1) not in self._tr_tools:
            return None
        name, rest = m.group(1), m.group(2).strip().lower()
        kwargs = {}
        if name == "music":
            action = next((_MUSIC_WORDS[w] for w in rest.split()
                           if w in _MUSIC_WORDS), None)
            if action is None:
                return None
            kwargs = {"action": action}
        elif name == "lamp" and rest:
            if rest in ("on", "off"):
                kwargs = {"state": rest}
            else:
                kwargs = {"color": rest}
        try:
            result = await self._tr_tools[name].run(**kwargs)
            logger.info(f"tool repair: executed verbalized {name}({kwargs}) "
                        f"-> {result!r}")
            return result
        except Exception as e:
            logger.warning(f"tool repair: {name} failed ({e})")
            return f"that didn't work: {e}"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._tr_buf, self._tr_hold = [], True

        elif isinstance(frame, LLMTextFrame) and self._tr_hold:
            self._tr_buf.append(frame)
            text = "".join(f.text for f in self._tr_buf).lstrip()
            if self._still_tool_like(text):
                return                          # keep holding, push nothing
            self._tr_hold = False               # ordinary reply: flush as-is
            for f in self._tr_buf:
                await self.push_frame(f, direction)
            self._tr_buf = []
            return

        elif isinstance(frame, LLMFullResponseEndFrame) and self._tr_hold:
            text = "".join(f.text for f in self._tr_buf).strip()
            self._tr_hold = False
            repaired = await self._execute(text) if text else None
            if repaired is not None:
                await self.push_frame(LLMTextFrame(text=repaired), direction)
            else:                                # short but not a tool call
                for f in self._tr_buf:
                    await self.push_frame(f, direction)
            self._tr_buf = []

        await self.push_frame(frame, direction)
