"""
Per-turn latency measurement.

Every stage boundary is recorded as a monotonic timestamp, then written to CSV as
milliseconds RELATIVE to the moment you stopped speaking. That reference point is
chosen deliberately: it is the instant the user's experience of waiting begins, so
every number in a row answers "how long after I finished talking did this happen".

All absolute times come from time.perf_counter() (monotonic, unaffected by clock
adjustments). Wall-clock is recorded once per turn, for correlating with logs.

Columns
-------
turn                     1-based turn number this run
wall_clock               ISO timestamp of turn end (when you stopped speaking)
t_user_stopped_speaking  always 0.0 — the reference point, kept for clarity
t_stt_done               transcription complete
t_vision_done            camera frame captured+encoded, or empty if no vision
t_llm_first_token        first token out of Gemma
t_llm_done               last token out of Gemma
t_tts_first_audio        first PCM chunk out of Piper
t_audio_playback_start   audio actually started going to the speaker
total_latency            = t_audio_playback_start (user stopped -> first sound)
vision_used              true/false
transcript_chars         length of the transcription
transcript               the text itself
reply_tokens             approximate token count of the reply
reply_chars              length of the reply
reply                    the text itself
interrupted              true if you barged in over the reply
"""

import csv
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

class SpeakingState:
    """
    Shared "is the bot talking right now?" flag.

    Needed because the two components that care sit at opposite ends of the
    pipeline. BotStartedSpeakingFrame is emitted by the OUTPUT transport and
    travels downstream, so a processor near the microphone — where echo
    suppression has to live — never sees it. The observer does see every frame at
    every link, so it owns the flag and the gate reads it.

    Timing note that took real debugging to find: BotStoppedSpeakingFrame fires
    when Pipecat finishes *writing* audio to the device, NOT when the speaker
    finishes making sound. Over Bluetooth the buffers are deep, so playback
    continues for seconds afterwards. Measured: the bot's own words were
    transcribed 6.3 s and 8.75 s "after it stopped".

    So a fixed tail cannot work — no constant is both safe and responsive. Instead
    we track how many seconds of audio were actually handed over, and stay muted
    until that much time has really elapsed since playback began, plus a small
    tail. Self-correcting, and independent of how deep the buffer happens to be.
    """

    __slots__ = ("speaking", "quiet_until", "tail", "last_stopped",
                 "started_count", "_play_start", "_audio_secs")

    def __init__(self, tail: float = 0.5):
        self.speaking = False
        self.quiet_until = 0.0
        self.tail = tail
        self.last_stopped = 0.0
        self.started_count = 0
        self._play_start = 0.0
        self._audio_secs = 0.0

    def add_audio(self, seconds: float):
        """Called for every chunk of bot audio produced."""
        self._audio_secs += seconds

    def started(self):
        if not self.speaking:
            self.started_count += 1
            self._play_start = time.monotonic()
            self._audio_secs = 0.0
        self.speaking = True

    def stopped(self):
        self.speaking = False
        now = time.monotonic()
        self.last_stopped = now
        # Whichever is later: now, or when the audio we handed over will actually
        # have finished playing. The max() is what absorbs Bluetooth's buffering.
        audio_ends = self._play_start + self._audio_secs
        self.quiet_until = max(now, audio_ends) + self.tail
        from loguru import logger
        logger.debug(f"gate: stopped(); booked {self._audio_secs:.2f}s audio, "
                     f"audio_ends {audio_ends-now:+.2f}s from now, "
                     f"mic reopens in {self.quiet_until-now:.2f}s")

    def muted(self) -> bool:
        return self.speaking or time.monotonic() < self.quiet_until


COLUMNS = [
    "turn", "wall_clock",
    "t_user_stopped_speaking", "t_stt_done", "t_vision_done",
    "t_llm_first_token", "t_llm_done", "t_tts_first_audio",
    "t_audio_playback_start", "total_latency",
    "vision_used", "transcript_chars", "transcript",
    "reply_tokens", "reply_chars", "reply", "interrupted",
]


@dataclass
class Turn:
    """One user utterance and the reply to it."""

    index: int
    wall_clock: str
    # Absolute perf_counter values. None until the stage happens.
    t0: float                                  # user stopped speaking
    stt_done: Optional[float] = None
    vision_done: Optional[float] = None
    llm_first_token: Optional[float] = None
    llm_done: Optional[float] = None
    tts_first_audio: Optional[float] = None
    playback_start: Optional[float] = None

    vision_used: bool = False
    transcript: str = ""
    reply_parts: list = field(default_factory=list)
    interrupted: bool = False
    written: bool = False

    @property
    def reply(self) -> str:
        return "".join(self.reply_parts).strip()

    def ms(self, t: Optional[float]) -> Optional[float]:
        """Convert an absolute stamp to ms since the user stopped speaking."""
        if t is None:
            return None
        return round((t - self.t0) * 1000.0, 1)

    def row(self) -> dict:
        reply = self.reply
        # Rough token estimate. We deliberately do NOT load a tokenizer just to
        # count tokens — it would cost memory we don't have for a logging detail.
        # ~4 chars/token is the standard approximation for English.
        approx_tokens = max(1, round(len(reply) / 4)) if reply else 0
        return {
            "turn": self.index,
            "wall_clock": self.wall_clock,
            "t_user_stopped_speaking": 0.0,
            "t_stt_done": self.ms(self.stt_done),
            "t_vision_done": self.ms(self.vision_done),      # None -> empty cell
            "t_llm_first_token": self.ms(self.llm_first_token),
            "t_llm_done": self.ms(self.llm_done),
            "t_tts_first_audio": self.ms(self.tts_first_audio),
            "t_audio_playback_start": self.ms(self.playback_start),
            "total_latency": self.ms(self.playback_start),
            "vision_used": str(self.vision_used).lower(),
            "transcript_chars": len(self.transcript),
            "transcript": self.transcript,
            "reply_tokens": approx_tokens,
            "reply_chars": len(reply),
            "reply": reply,
            "interrupted": str(self.interrupted).lower(),
        }

    def summary(self) -> str:
        """Compact one-block console summary. Shows per-stage DELTAS, because
        'STT took 900ms' is more actionable than 'STT finished at 900ms'."""
        def d(a, b):
            if a is None or b is None:
                return "    —"
            return f"{(b - a) * 1000:5.0f}"

        stt = d(self.t0, self.stt_done)
        vis = d(self.stt_done, self.vision_done) if self.vision_done else "    —"
        # First token is measured from whichever came last: STT, or the vision
        # capture. Otherwise a vision turn makes the LLM look slower than it is.
        llm_in = self.vision_done or self.stt_done
        ttft = d(llm_in, self.llm_first_token)
        gen = d(self.llm_first_token, self.llm_done)
        tts = d(self.llm_first_token, self.tts_first_audio)
        play = d(self.tts_first_audio, self.playback_start)
        total = self.ms(self.playback_start)
        totals = f"{total:.0f}ms" if total is not None else "n/a"

        eye = "  [vision]" if self.vision_used else ""
        cut = "  [interrupted]" if self.interrupted else ""
        return (
            f"\n  turn {self.index}{eye}{cut}\n"
            f"    you: {self.transcript[:70]}\n"
            f"    bot: {self.reply[:70]}\n"
            f"    stt {stt}ms | vision {vis}ms | ttft {ttft}ms | "
            f"gen {gen}ms | tts {tts}ms | play {play}ms\n"
            f"    TOTAL (you stopped -> first sound): {totals}"
        )


class MetricsLogger:
    """Owns the CSV file and the current turn."""

    def __init__(self, csv_dir: Path, console: bool = True):
        csv_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = csv_dir / f"latency_{stamp}.csv"
        self.console = console
        self.turn_count = 0
        self.current: Optional[Turn] = None
        # Turns whose reply has not arrived yet, oldest first.
        #
        # Without this, replies were attributed to the WRONG question. Observed in a
        # real conversation: a vision turn logged 2587 ms (impossible — the encoder
        # alone is 4.5 s) carrying a reply that belonged to the previous question,
        # while the actual vision answer appeared on the next row. If you speak
        # again while a reply is still streaming, "current" has already moved on and
        # every later stamp lands on the new turn.
        self._awaiting: deque = deque()

        # Header written up front and flushed on every row, so a run that is
        # killed with ctrl-c still leaves a complete, readable file.
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._fh.flush()

        # Timestamp of the most recent "user stopped speaking", held until a
        # transcript proves the utterance was real. See mark_turn_end().
        self._pending_t0: Optional[float] = None

    # -- turn lifecycle -----------------------------------------------------
    def mark_turn_end(self):
        """
        VAD says the user stopped talking. Remember WHEN, but do not open a turn.

        VAD fires on coughs, chair scrapes, and door noise. Opening a turn per
        event produced phantom turns that (a) cluttered the CSV and (b) worse,
        prematurely finalised the real turn before its reply arrived — which is
        why early runs logged an empty reply and no LLM timings.

        The timestamp is still captured here, because this instant is the correct
        reference point for every latency in the row. It is only *promoted* to a
        real turn once STT returns actual text.
        """
        self._pending_t0 = time.perf_counter()

    def start_turn(self) -> Turn:
        """Open a turn now, using the pending VAD timestamp as t=0 if we have one."""
        self.turn_count += 1
        self.current = Turn(
            index=self.turn_count,
            wall_clock=datetime.now().isoformat(timespec="milliseconds"),
            # Fall back to now() only if STT somehow beat VAD's end-of-turn.
            t0=self._pending_t0 if self._pending_t0 is not None
            else time.perf_counter(),
        )
        self._pending_t0 = None
        return self.current

    def finish_turn(self):
        # Finalise the turn whose reply just finished — the head of the queue —
        # rather than whatever happens to be "current".
        t = self._target()
        if t is None or t.written:
            return
        # A turn with nothing but a start is noise (e.g. a cough that tripped
        # VAD but produced no transcript). Don't pollute the dataset.
        if t.stt_done is None and not t.transcript:
            self.current = None
            return
        t.written = True
        self._writer.writerow(t.row())
        self._fh.flush()
        if self.console:
            print(t.summary(), flush=True)
        while self._awaiting and self._awaiting[0].written:
            self._awaiting.popleft()
        if self.current is t:
            self.current = None

    def close(self):
        # Flush every turn still queued, so ctrl-c never loses data.
        for _ in range(len(self._awaiting) + 1):
            self.finish_turn()
        try:
            self._fh.close()
        except Exception:
            pass

    # -- stage stamps -------------------------------------------------------
    # Each of these is a no-op if there is no active turn, which keeps callers
    # free of None-checks. Stamps are first-write-wins where a stage can fire
    # repeatedly (first token, first audio chunk).
    def _target(self) -> Optional["Turn"]:
        """The turn a reply belongs to: the oldest one still awaiting one."""
        while self._awaiting and self._awaiting[0].written:
            self._awaiting.popleft()
        return self._awaiting[0] if self._awaiting else self.current

    def _stamp(self, attr: str, once: bool = True):
        t = self._target()
        if t is None:
            return
        if once and getattr(t, attr) is not None:
            return
        setattr(t, attr, time.perf_counter())

    def stt_done(self, text: str):
        # A transcript proves the utterance was real, so this is where a turn
        # begins. If a previous turn is still waiting for its reply we do NOT
        # finalise it here — its stamps are still to come. It goes on the queue and
        # is written when its own reply completes.
        if self.current is not None and self.current.transcript:
            self._awaiting.append(self.current)
            self.current = None
        if self.current is None:
            self.start_turn()
        t = self.current
        t.stt_done = time.perf_counter()
        t.transcript = text
        self._awaiting.append(t)

    def discard_last_turn_end(self):
        """
        The utterance that just ended was swallowed before a transcript was ever
        pushed (asleep and no wake phrase, or a bare 'hey Roomi'). Its turn-end
        stamp is already in the FIFO though — mark_turn_end fires on
        UserStoppedSpeaking, before any transcript exists. Left there, the NEXT
        real transcript would pair with this stale stamp and book a total that
        includes the silence in between. Pop it.
        """
        if self._awaiting and not getattr(self._awaiting[-1], "written", False):
            self._awaiting.pop()

    def vision_done(self):
        self._stamp("vision_done")
        t = self._target()
        if t:
            t.vision_used = True

    def llm_first_token(self):
        self._stamp("llm_first_token")

    def llm_text(self, text: str):
        t = self._target()
        if t:
            t.reply_parts.append(text)

    def llm_done(self):
        self._stamp("llm_done", once=False)

    def tts_first_audio(self):
        self._stamp("tts_first_audio")

    def playback_start(self):
        self._stamp("playback_start")

    def interrupted(self):
        t = self._target()
        if t:
            t.interrupted = True
