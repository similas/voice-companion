"""
"Hey Roomi" — wake gating, phase 1.

WHERE THIS RUNS, AND WHY IT IS NOT A PIPELINE PROCESSOR
-------------------------------------------------------
The obvious design is a FrameProcessor after STT that swallows transcripts which
lack the wake phrase. It has a subtle flaw: the latency observer sees every push
on every link, so the moment STT pushes a transcript — even one a downstream gate
will discard — the metrics logger opens a turn for it. Every ignored utterance
would leave a half-open turn and skew the next real one (the turn-end FIFO would
pair the next transcript with a stale timestamp, inflating totals by minutes).

So the filter runs INSIDE the STT service, at the single point where transcripts
are born. A swallowed utterance is never pushed, so to the rest of the pipeline —
and to the metrics — it simply never happened. The one bit of bookkeeping that
already escaped (mark_turn_end fires on UserStoppedSpeaking, before any
transcript exists) is unwound via metrics.discard_last_turn_end().

BEHAVIOUR
---------
Two states, like every wake-word assistant:

  ASLEEP  only utterances beginning with the wake phrase get through. The phrase
          itself is stripped: "Hey Roomi, what time is it?" becomes a normal turn
          reading "what time is it?". A bare "Hey Roomi." plays the chime and
          opens the window without starting an LLM turn.
  AWAKE   for window_secs after any accepted interaction, everything passes —
          a conversation should not require re-addressing the thing per sentence.
          Each accepted turn refreshes the window (a roommate does not fall
          asleep mid-chat).

The chime plays on every wake detection. It is fired via paplay as a detached
process: PipeWire mixes it over whatever else is happening, nothing blocks, and
the agent's own output stream stays untouched.

Matching is on whisper's TRANSCRIPT, so it inherits whisper's spelling of the
name. Observed spellings go in wake.phrases — "roomy", "roomie", "rumi" are all
the same sound, and refusing to answer over orthography would be absurd.

Phase 2 moves detection to the XIAO ESP32-S3 on the reSpeaker board (the mic
hardware can run the wake word itself and let the Jetson idle); this class is the
seam where that lands — the pipeline above and below does not know or care where
the wake decision comes from.
"""

import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from loguru import logger

_norm_re = re.compile(r"[^a-z0-9 ]+")

# The NAME, as a shape rather than a spelling. The ASR transcribes the sound of
# "Roomi" differently on every take — observed across ~20 real wake attempts on
# 2026-08-12, BOTH engines: 'Roomy', 'Roomi', 'Rumi', 'Ruby', 'Rooney',
# 'roommate', 'roomie', 'broom', 'brooming', 'loomy'. Enumerating spellings
# lost that race twice in one evening; the first shape regex (^b?r[ou]+m...)
# then lost it again to 'Ruby' (b, not m), 'Rooney' (n) and 'loomy' (l) — the
# user measured wake at 1-in-5. So: optional b, an r-or-l onset, a rounded
# vowel run, then any of m/n/b, any tail. Every observed spelling matches;
# 'bob', 'noon' and 'buddy' still do not.
_NAME_RE = re.compile(r"^b?[rl][ou]+[mnb][a-z]*$")
# 'he' and 'a' are what "hey" becomes at distance ('A ruby', 2026-08-12). The
# bare-vowel trigger is riskier, so it only counts in a two-word utterance —
# "a robot appeared on screen" from the TV must not chime.
_WAKE_TRIGGERS = {"hey", "hi", "hay", "he"}
_WAKE_TRIGGERS_STRICT = {"a", "uh", "oh"}
_SLEEP_TRIGGERS = {"bye", "by", "buy", "goodbye"}


def _pair_match(pair, triggers) -> bool:
    return (len(pair) == 2 and pair[0] in triggers
            and _NAME_RE.match(pair[1]) is not None)


def _norm(text: str) -> list:
    return _norm_re.sub(" ", text.lower()).split()


class WakeFilter:
    """Decides a transcript's fate: pass (possibly stripped), or never happened."""

    def __init__(self, phrases, window_secs: float = 45.0,
                 chime: Optional[str] = None, metrics=None, enabled: bool = True,
                 sleep_phrases=None, sleep_chime: Optional[str] = None,
                 on_wake=None):
        self.enabled = enabled
        # Fired on every EXPLICIT wake ("hey Roomi"), not on window refreshes —
        # so "hey Roomi" over playing music pauses it to listen, but the
        # conversation that follows (which refreshes the window) does not
        # re-pause what the user just asked to resume.
        self._on_wake = on_wake
        # Each phrase pre-normalised to a word tuple for prefix comparison.
        self._phrases = [tuple(_norm(p)) for p in (phrases or []) if _norm(p)]
        self._sleep_phrases = [tuple(_norm(p)) for p in (sleep_phrases or [])
                               if _norm(p)]
        self._sleep_chime = str(Path(sleep_chime)) if sleep_chime else None
        self._window = window_secs
        self._until = 0.0
        self._chime = str(Path(chime)) if chime else None
        self._metrics = metrics

    # -- state ---------------------------------------------------------------
    def awake(self) -> bool:
        return time.monotonic() < self._until

    def _wake_up(self):
        self._until = time.monotonic() + self._window

    def chime(self, path: Optional[str] = None):
        """Fire-and-forget acknowledgement sound; never blocks, never raises."""
        path = path or self._chime
        if not path:
            return
        try:
            subprocess.Popen(["paplay", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.warning(f"wake: chime failed ({e})")

    # -- the decision ----------------------------------------------------------
    def filter(self, text: str) -> Optional[str]:
        """
        Returns the text the pipeline should see, or None to swallow the
        utterance entirely. Called at the transcript's birthplace in the STT
        service, before any push.
        """
        if not self.enabled:
            return text

        words = _norm(text)

        # "bye Roomi" — matched at the START or the END of the utterance, because
        # both are natural ("bye Roomi" / "ok thanks, bye Roomi"). Closes the
        # window immediately, answers with the descending chime, and the
        # utterance itself never reaches the LLM — a goodbye that triggers a
        # spoken reply would defeat its own purpose.
        if self._sleep_phrases or True:
            sleep_hit = (_pair_match(words[:2], _SLEEP_TRIGGERS)
                         or _pair_match(words[-2:], _SLEEP_TRIGGERS)
                         or any(tuple(words[:len(p)]) == p
                                or tuple(words[-len(p):]) == p
                                for p in self._sleep_phrases))
            if sleep_hit:
                    was_awake = self.awake()
                    self._until = 0.0
                    if was_awake:
                        self.chime(self._sleep_chime)
                        logger.info("wake: 'bye Roomi' — going back to sleep")
                    if self._metrics is not None:
                        self._metrics.discard_last_turn_end()
                    return None

        matched = next((p for p in self._phrases if tuple(words[:len(p)]) == p),
                       None)
        # Shape-based wake: "hey <name-shaped word>" in any spelling the ASR
        # invents. The explicit config list still works and can carry longer
        # phrases; this catches the spellings nobody predicted.
        if matched is None and _pair_match(words[:2], _WAKE_TRIGGERS):
            matched = tuple(words[:2])
        # Degraded triggers ("a ruby") and the bare name ("Roomi?") count only
        # when the whole utterance is just the call — someone summoning it,
        # not the TV mentioning a robot.
        if matched is None and len(words) == 2 and _pair_match(
                words, _WAKE_TRIGGERS_STRICT):
            matched = tuple(words)
        if matched is None and len(words) == 1 and _NAME_RE.match(words[0]):
            matched = tuple(words)

        if matched:
            # Strip the wake phrase by word count from the ORIGINAL text, so the
            # command keeps its capitalisation and punctuation.
            remainder = " ".join(text.split()[len(matched):]).strip()
            self._wake_up()
            self.chime()
            if self._on_wake:
                try:
                    self._on_wake()
                except Exception as e:      # never break a wake over a hook
                    logger.debug(f"wake: on_wake hook failed ({e})")
            if remainder:
                logger.info(f"wake: 'hey Roomi' + command -> {remainder!r}")
                return remainder
            logger.info("wake: Roomi is listening "
                        f"(window {self._window:.0f}s)")
            if self._metrics is not None:
                self._metrics.discard_last_turn_end()
            return None                     # bare wake: chime is the whole reply

        if self.awake():
            self._wake_up()                 # conversation refreshes the window
            return text

        logger.debug(f"wake: asleep, ignored {text!r}")
        if self._metrics is not None:
            self._metrics.discard_last_turn_end()
        return None
