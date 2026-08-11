"""
Latency instrumentation via Pipecat's observer hook.

An observer sees every frame pushed between every pair of processors, without
sitting in the pipeline itself. That matters for measurement: a FrameProcessor
inserted to take timestamps would add its own await points to the hot path and
perturb the very numbers we're collecting. The observer is passive.

One wrinkle: the same frame is observed once per hop it makes through the
pipeline, so every stamp must be deduplicated by frame id, or "first token" would
be recorded at whichever hop happened to be seen first.
"""

from collections import OrderedDict

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed


class LatencyObserver(BaseObserver):
    """Turns frame traffic into per-stage timestamps on the MetricsLogger."""

    def __init__(self, metrics, speaking=None, echo_suppressed=False):
        super().__init__()
        self.metrics = metrics
        # When echo suppression is on, barge-in is deliberately disabled: the mic
        # audio is dropped while the bot talks. Any "interruption" seen in that
        # window is the bot's own voice, not the user, so it must not be logged
        # as one — that flagged every turn and made the column meaningless.
        self.echo_suppressed = echo_suppressed
        # Shared with EchoGate — see SpeakingState's docstring for why the flag
        # has to travel out-of-band rather than as a frame.
        self.speaking = speaking
        self._bot_speaking = False
        # Bounded LRU of frame ids we've already acted on. Bounded because a long
        # session pushes a lot of audio frames and an unbounded set would leak.
        self._seen = OrderedDict()

    def _first_time(self, frame) -> bool:
        fid = id(frame) if getattr(frame, "id", None) is None else frame.id
        if fid in self._seen:
            return False
        self._seen[fid] = True
        if len(self._seen) > 4096:
            self._seen.popitem(last=False)
        return True

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        # The one link every playable audio frame crosses exactly once: INTO the
        # output transport. Counting there needs no id-based dedupe at all — which
        # matters, because dedupe by frame id was tried for the gate's duration
        # accounting and still triple-counted (frames are re-emitted as new
        # instances at some hops, so ids are not stable across the pipeline).
        dst_is_output = "OutputTransport" in type(data.destination).__name__
        m = self.metrics

        # ---- turn boundaries ----------------------------------------------
        if isinstance(frame, UserStoppedSpeakingFrame):
            if self._first_time(frame):
                # Records the timestamp only. The turn is created when a
                # transcript arrives — see MetricsLogger.mark_turn_end().
                m.mark_turn_end()
            return

        if isinstance(frame, TranscriptionFrame):
            if self._first_time(frame) and (frame.text or "").strip():
                m.stt_done(frame.text or "")
            return

        # ---- LLM -----------------------------------------------------------
        if isinstance(frame, LLMTextFrame):
            if self._first_time(frame):
                m.llm_first_token()
                m.llm_text(frame.text or "")
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            if self._first_time(frame):
                m.llm_done()
            return

        # ---- TTS + playback ------------------------------------------------
        if isinstance(frame, TTSAudioRawFrame):
            # tts_first_audio is first-write-wins, so no dedupe needed for it.
            m.tts_first_audio()
            # Close the echo gate as soon as bot audio EXISTS, not when playback
            # is reported. BotStartedSpeaking can lag the first samples reaching
            # the speaker, and anything that leaks through in that window gets
            # transcribed as a user turn.
            if self.speaking:
                self.speaking.started()
                # Count duration ONLY on the hop into the output transport. Every
                # playable chunk crosses that link exactly once, so no dedupe is
                # needed — and dedupe by frame id demonstrably failed here: with it
                # in place, a ~3.5 s reply was still booked as 10.1 s, keeping the
                # mic gated for ~5 phantom seconds after every reply.
                # 16-bit PCM, so 2 bytes per sample per channel.
                if dst_is_output:
                    audio = getattr(frame, "audio", None)
                    rate = getattr(frame, "sample_rate", 0) or 0
                    ch = getattr(frame, "num_channels", 1) or 1
                    if audio and rate:
                        self.speaking.add_audio(len(audio) / (rate * ch * 2))
            return

        if isinstance(frame, BotStartedSpeakingFrame):
            if self.speaking:
                self.speaking.started()
            if self._first_time(frame):
                self._bot_speaking = True
                m.playback_start()
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            if self.speaking:
                self.speaking.stopped()
            if self._first_time(frame):
                self._bot_speaking = False
                # The turn is complete here: playback has finished, so every
                # stamp including llm_done is in. Write the row.
                m.finish_turn()
            return

        # ---- barge-in ------------------------------------------------------
        # Only count it when the bot was ACTUALLY speaking. Pipecat also emits
        # InterruptionFrame as a routine pipeline flush at the start of a
        # response, and treating that as barge-in flagged every single turn as
        # "interrupted" — which made the column useless.
        if (self._bot_speaking and not self.echo_suppressed and isinstance(
                frame, (InterruptionFrame, UserStartedSpeakingFrame))):
            if self._first_time(frame):
                m.interrupted()
                logger.debug("real interruption: user spoke over the bot")
            return
