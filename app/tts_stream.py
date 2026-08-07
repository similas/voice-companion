"""
Clause-level text aggregation, so speech starts sooner.

Pipecat's default aggregator waits for end-of-sentence punctuation before handing
anything to the TTS engine. For a voice loop that is a real cost: with a reply like

    "The capital of France is Paris, and it has about two million people."

nothing is spoken until the final full stop, even though "The capital of France is
Paris," was complete and speakable much earlier. Piper synthesises at ~0.09x
realtime, so the wait is pure dead air.

This aggregator flushes on the first of:

  * end of sentence            . ! ?
  * a clause boundary          , ; : —   once min_clause_chars have accumulated
  * first_chunk_chars          a word boundary near this length, FIRST chunk only
  * max_chars                  a hard cap, for models that ramble without punctuation

The first_chunk cap matters more than the clause rule in practice. Measured on real
replies from this assistant, only about a third contain a comma early enough to
help; "I see a modern living room with a kitchen island and a comfortable sofa."
has none, so without a length cap the listener waits for all 72 characters to be
generated — about 1.0 s at 17.6 tok/s — before hearing anything.

Only the FIRST chunk of a reply is cut aggressively. Once audio is playing, the
listener cannot tell whether the next chunk arrived early or late, so later chunks
use full sentences — which sound better, since Piper's prosody improves with a
complete clause and chopping mid-sentence produces audible seams.
"""

import re
from typing import AsyncIterator, Optional

from pipecat.utils.string import match_endofsentence
from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)

# Boundaries that are safe to break a spoken phrase on. A comma is the workhorse;
# em dash and colon are included because small models use them constantly.
_CLAUSE = re.compile(r"[,;:—–]\s")


class ClauseAggregator(BaseTextAggregator):
    """Sentence aggregator that also breaks on clauses for the first utterance."""

    def __init__(self, min_clause_chars: int = 18, max_chars: int = 160,
                 first_chunk_chars: int = 46, first_chunk_only: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self._text = ""
        self._min_clause = min_clause_chars
        self._max_chars = max_chars
        self._first_chunk = first_chunk_chars
        self._first_only = first_chunk_only
        self._emitted = 0

    @property
    def text(self) -> Aggregation:
        return Aggregation(text=self._text.strip(" "), type=AggregationType.SENTENCE)

    def _clause_cut(self, buf: str) -> Optional[int]:
        """Index to cut at for a clause break, or None."""
        if self._first_only and self._emitted > 0:
            return None
        if len(buf) < self._min_clause:
            return None
        m = None
        for m in _CLAUSE.finditer(buf):
            pass                      # take the LAST boundary we have seen so far
        if not m:
            return None
        cut = m.end() - 1             # keep the punctuation, drop the space
        return cut if cut >= self._min_clause else None

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        if self._aggregation_type == AggregationType.TOKEN:
            if text:
                yield Aggregation(text=text, type=AggregationType.TOKEN)
            return

        for char in text:
            self._text += char

            # 1. a real sentence end always wins
            eos = match_endofsentence(self._text)
            if eos:
                chunk, self._text = self._text[:eos], self._text[eos:]
                if chunk.strip():
                    self._emitted += 1
                    yield Aggregation(text=chunk.strip(" "),
                                      type=AggregationType.SENTENCE)
                continue

            # 2. clause boundary, early in the reply only
            cut = self._clause_cut(self._text)
            if cut:
                chunk, self._text = self._text[:cut], self._text[cut:]
                if chunk.strip():
                    self._emitted += 1
                    yield Aggregation(text=chunk.strip(" "),
                                      type=AggregationType.SENTENCE)
                continue

            # 3. first chunk only: break at a word boundary near first_chunk_chars
            # even with no punctuation at all, so audio starts promptly.
            if self._emitted == 0 and len(self._text) >= self._first_chunk:
                sp = self._text.rfind(" ")
                if sp >= self._min_clause:
                    chunk, self._text = self._text[:sp], self._text[sp:]
                    if chunk.strip():
                        self._emitted += 1
                        yield Aggregation(text=chunk.strip(" "),
                                          type=AggregationType.SENTENCE)
                    continue

            # 4. hard cap so an unpunctuated ramble still gets spoken
            if len(self._text) >= self._max_chars:
                sp = self._text.rfind(" ")
                cut = sp if sp > self._max_chars // 2 else len(self._text)
                chunk, self._text = self._text[:cut], self._text[cut:]
                if chunk.strip():
                    self._emitted += 1
                    yield Aggregation(text=chunk.strip(" "),
                                      type=AggregationType.SENTENCE)

    async def flush(self) -> Optional[Aggregation]:
        if not self._text.strip():
            self._text = ""
            return None
        chunk, self._text = self._text, ""
        self._emitted += 1
        return Aggregation(text=chunk.strip(" "), type=AggregationType.SENTENCE)

    async def handle_interruption(self):
        self._text = ""
        self._emitted = 0

    async def reset(self):
        self._text = ""
        self._emitted = 0
