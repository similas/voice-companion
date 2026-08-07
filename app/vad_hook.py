"""
A VAD wrapper that tells us the moment speech *probably* ended.

Why this exists
---------------
Turn detection deliberately waits `stop_secs` of silence before declaring the turn
over, so a pause mid-sentence does not cut you off. With stop_secs = 0.5 the
sequence is:

    last sound ──0.5s hangover──▶ UserStoppedSpeaking ──692ms decode──▶ transcript

Nothing happens during that hangover. But the speech is already complete when it
begins — that is the entire premise of the hangover. So we can decode *then*, and by
the time the turn is confirmed the transcript is largely or wholly done.

Silero exposes exactly the signal needed: VADState.STOPPING is entered on the first
silent frame and held until either stop_secs elapses (→ QUIET, turn over) or speech
resumes (→ SPEAKING, false alarm). This wrapper reports both edges so a consumer can
start a speculative decode and throw it away if the user carries on talking.

An earlier attempt at "streaming STT" — decoding partial windows during speech and
committing segments closed off by silence — was implemented and MEASURED SLOWER:
714 ms → 1074 ms with zero commits. Real utterances to a voice assistant are 2-4 s
with no internal pauses, so there is never a segment that ends far enough from the
buffer edge to be safely committed, and the wasted partial decodes delayed the final
one. Decoding during the hangover works because it exploits idle time that exists by
construction, rather than hoping for pauses that are not there.
"""

from typing import Callable, Optional

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADState


class HookedSileroVAD(SileroVADAnalyzer):
    """Silero VAD that reports entry to / exit from the STOPPING hangover."""

    def __init__(self, *args, on_maybe_stopped: Optional[Callable] = None,
                 on_resumed: Optional[Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._hk_on_stopping = on_maybe_stopped
        self._hk_on_resumed = on_resumed
        self._hk_prev = VADState.QUIET

    def set_callbacks(self, on_maybe_stopped=None, on_resumed=None):
        """Wire callbacks after construction (the STT service is built later)."""
        if on_maybe_stopped:
            self._hk_on_stopping = on_maybe_stopped
        if on_resumed:
            self._hk_on_resumed = on_resumed

    async def analyze_audio(self, buffer: bytes) -> VADState:
        state = await super().analyze_audio(buffer)
        prev, self._hk_prev = self._hk_prev, state

        if state != prev:
            if state == VADState.STOPPING:
                # Speech just went quiet. It may still resume, but the audio we have
                # is a complete utterance either way — safe to start decoding.
                if self._hk_on_stopping:
                    try:
                        self._hk_on_stopping()
                    except Exception as e:      # never break the audio path
                        logger.debug(f"vad hook (stopping) failed: {e}")
            elif prev == VADState.STOPPING and state == VADState.SPEAKING:
                # False alarm — a pause, not the end. Discard the speculation.
                if self._hk_on_resumed:
                    try:
                        self._hk_on_resumed()
                    except Exception as e:
                        logger.debug(f"vad hook (resumed) failed: {e}")

        return state
