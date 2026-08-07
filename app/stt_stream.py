"""
Speculative STT: decode during the VAD hangover, not after it.

THE PROBLEM
-----------
Pipecat's segmented STT buffers the whole utterance and calls Whisper only once VAD
declares the turn over. Since the llama-server switch removed the LLM bottleneck,
that 692-742 ms is now the largest single component of a text turn.

WHAT DOES NOT WORK (measured, not assumed)
------------------------------------------
Classic streaming — decode partial windows during speech, commit segments that
silence has closed off — was implemented and benchmarked: **714 ms -> 1074 ms, with
zero commits.** Utterances to a voice assistant are 2-4 s of continuous speech with
no internal pauses, so no segment ever ends far enough from the buffer edge to be
committed safely, and the wasted partial decodes delayed the final one. The idea is
sound for dictation and wrong for this workload.

WHAT DOES WORK
--------------
Turn detection waits `stop_secs` (0.5 s) of silence before ending the turn, so a
thinking pause does not cut you off. That hangover is dead time by construction — and
the speech is already complete when it starts. So:

    last sound ─▶ VAD STOPPING ─▶ [decode HERE] ─▶ turn confirmed ─▶ transcript ready

If the decode finishes inside the hangover, the transcript is ready the instant the
turn is confirmed. If the user resumes talking, the speculation is discarded and
nothing is lost but some idle CPU.

Correctness: the speculative decode sees exactly the same audio the final decode
would, because everything after the last sound is silence. It is the same
computation, moved earlier — not an approximation.
"""

import asyncio
import threading
import time
from typing import AsyncGenerator, Optional

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601


class SpeculativeWhisperSTTService(STTService):
    """faster-whisper that starts decoding as soon as speech goes quiet."""

    def __init__(self, *, model: str = "tiny", device: str = "cpu",
                 compute_type: str = "int8", language: str = "en",
                 no_speech_prob: float = 0.6, min_chars: int = 8,
                 speculative: bool = True, sample_rate: int = 16000, **kwargs):
        # Declare every settings field explicitly. STTService logs an ERROR for any
        # field left NOT_GIVEN, and an error-level line about our own service being
        # misconfigured is noise that hides real problems.
        kwargs.setdefault("settings", STTSettings(model=model, language=language))
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._no_speech_prob = no_speech_prob
        self._min_chars = min_chars
        self._speculative = speculative

        self._model = None
        self._lock = threading.Lock()
        self._audio = np.zeros(0, dtype=np.float32)
        self._speaking = False

        # Speculation state
        self._spec_task: Optional[asyncio.Task] = None
        self._spec_text: Optional[str] = None
        self._spec_samples = 0          # how much audio the speculation covered
        self._spec_started = 0.0
        self.spec_hits = 0
        self.spec_misses = 0

    # -- lifecycle ---------------------------------------------------------
    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._model is None:
            from faster_whisper import WhisperModel
            t0 = time.perf_counter()
            self._model = await asyncio.to_thread(
                WhisperModel, self._model_name, device=self._device,
                compute_type=self._compute_type)
            logger.info(
                f"stt: faster-whisper {self._model_name} ({self._compute_type}) "
                f"loaded in {time.perf_counter()-t0:.1f}s"
                f"{', speculative decoding on' if self._speculative else ''}")

    async def stop(self, frame: EndFrame):
        self._cancel_speculation()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        self._cancel_speculation()
        await super().cancel(frame)

    # -- decode ------------------------------------------------------------
    def _decode(self, audio: np.ndarray) -> str:
        segs, _ = self._model.transcribe(
            audio, language=self._language, beam_size=1, temperature=0.0,
            condition_on_previous_text=False, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            no_speech_threshold=0.6)
        return " ".join(s.text.strip() for s in segs
                        if s.no_speech_prob < self._no_speech_prob
                        and s.text.strip()).strip()

    # -- speculation -------------------------------------------------------
    def on_maybe_stopped(self):
        """VAD entered STOPPING: speech has gone quiet, decode now."""
        if not self._speculative or self._model is None or not self._speaking:
            return
        if self._spec_task and not self._spec_task.done():
            return
        with self._lock:
            audio = self._audio.copy()
        if len(audio) / self._sample_rate < 0.25:
            return
        self._spec_started = time.perf_counter()
        self._spec_samples = len(audio)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._spec_task = loop.create_task(self._speculate(audio))

    async def _speculate(self, audio: np.ndarray):
        try:
            text = await asyncio.to_thread(self._decode, audio)
            self._spec_text = text
            logger.debug(f"stt: speculative decode ready in "
                         f"{(time.perf_counter()-self._spec_started)*1000:.0f}ms "
                         f"({len(text)} chars)")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"stt: speculative decode failed ({e})")
            self._spec_text = None

    def _cancel_speculation(self):
        """User resumed talking, or the turn ended — drop any in-flight guess."""
        if self._spec_task and not self._spec_task.done():
            self._spec_task.cancel()
        self._spec_task = None
        self._spec_text = None
        self._spec_samples = 0

    def on_resumed(self):
        """VAD went STOPPING -> SPEAKING: it was a pause, not the end."""
        if self._spec_task:
            self.spec_misses += 1
            logger.debug("stt: speech resumed, discarding speculation")
        self._cancel_speculation()

    # -- frames ------------------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            with self._lock:
                self._audio = np.zeros(0, dtype=np.float32)
            self._cancel_speculation()
            self._speaking = True

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._speaking = False
            t0 = time.perf_counter()
            text = await self._finalise()
            took = (time.perf_counter() - t0) * 1000
            if len(text) >= self._min_chars:
                logger.debug(f"stt: transcript in {took:.0f}ms after turn end")
                # NOTE: no _handle_transcription() call — that is a tracing hook on
                # WhisperSTTService, not on the STTService base we subclass.
                await self.push_frame(TranscriptionFrame(
                    text, "", time_now_iso8601(), self._language))
            elif text:
                logger.info(f"stt: discarded too-short {text!r}")

    async def _finalise(self) -> str:
        """Use the speculation if it covered the whole utterance, else decode."""
        with self._lock:
            audio = self._audio.copy()
            self._audio = np.zeros(0, dtype=np.float32)

        task, spec_samples = self._spec_task, self._spec_samples
        self._spec_task = None

        if task is not None:
            # Everything after the speculation started is silence (that is what
            # STOPPING means), so a speculation covering nearly all the audio is
            # equivalent to decoding the lot. Allow 0.35 s of slack for the frames
            # that arrived between the hook firing and the snapshot.
            slack = int(0.35 * self._sample_rate)
            if spec_samples >= len(audio) - slack:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=8.0)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                if self._spec_text is not None:
                    self.spec_hits += 1
                    text, self._spec_text = self._spec_text, None
                    return text
            else:
                # Too much audio arrived after the guess: it is stale.
                task.cancel()
                self.spec_misses += 1

        self._spec_text = None
        if len(audio) / self._sample_rate < 0.08:
            return ""
        try:
            return await asyncio.to_thread(self._decode, audio)
        except Exception as e:
            logger.warning(f"stt: decode failed: {e}")
            return ""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Accumulate audio. Transcripts are pushed on UserStoppedSpeaking."""
        if self._model is None:
            yield ErrorFrame("whisper model not loaded")
            return
        chunk = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self._audio = np.concatenate([self._audio, chunk])
        return
        yield  # pragma: no cover — keeps this an async generator
