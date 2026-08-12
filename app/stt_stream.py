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
import wave
from collections import deque
from pathlib import Path
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
                 speculative: bool = True, sample_rate: int = 16000,
                 save_dir=None, cpu_threads: int = 2, wake_filter=None,
                 engine: str = "whisper", **kwargs):
        # Declare every settings field explicitly. STTService logs an ERROR for any
        # field left NOT_GIVEN, and an error-level line about our own service being
        # misconfigured is noise that hides real problems.
        kwargs.setdefault("settings", STTSettings(model=model, language=language))
        super().__init__(sample_rate=sample_rate, **kwargs)
        # ENGINE (v0.4): "moonshine" or "whisper". Same speculative machinery,
        # different decoder. Measured on this device, identical Piper-voiced
        # prompts, cores 3-5:
        #     whisper-tiny int8   median 1017 ms   WER 16.7%
        #     moonshine-tiny f32  median  142 ms   WER  4.2%
        # 7x faster AND 4x more accurate — moonshine has no 30 s padding
        # window, so a 3 s utterance costs 3 s of encoder, not 30. It also
        # transcribes pure silence and noise as '' (verified), where whisper
        # invented sentences and needed no_speech_prob + vad_filter to cope.
        # Wake phrases verified against moonshine's spellings on the recorded
        # test audio: 'Hey, roomy.' / 'Hey roommie' / 'By Rumi.' all match.
        self._engine = engine
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._no_speech_prob = no_speech_prob
        self._min_chars = min_chars
        self._speculative = speculative
        # Wake gating happens HERE, where transcripts are born, not in a pipeline
        # processor downstream — the observer sees every push, so a transcript
        # that a later gate would swallow still opens a metrics turn. One that is
        # never pushed never happened. See app/wake.py.
        self._wake = wake_filter
        # Sized to the agent's core partition (3 cores, set in run.sh). Whisper
        # and Piper barely overlap inside a turn, so whisper gets the whole
        # partition; 2 threads was measured ~500 ms slower per decode.
        self._cpu_threads = cpu_threads
        # Where to dump the audio whisper actually sees (None = off).
        self._save_dir = Path(save_dir) if save_dir else None
        self._utt = 0

        self._model = None
        self._tokenizer = None          # moonshine only
        self._lock = threading.Lock()
        self._audio = np.zeros(0, dtype=np.float32)
        self._speaking = False

        # PRE-ROLL RING BUFFER — fixes two bugs at once, both observed live:
        #
        # 1. UNBOUNDED GROWTH. run_stt used to append every chunk to _audio forever,
        #    resetting only when a turn STARTED. During silence that is a leak
        #    (~225 MB/hour at 16 kHz float32) and an O(n^2) re-copy every 20 ms; the
        #    agent was found at 2.3 GB RSS after an evening idle. Idle audio now
        #    lands here, bounded.
        #
        # 2. FIRST-WORD CLIPPING. VAD needs start_secs (0.2 s) of speech before it
        #    opens the turn, and everything before that moment was discarded — so
        #    "Can you hear me?" was transcribed as 'You hear me?' and "Tell me a
        #    joke" as 'me a joke.' (both in logs/turns/*.wav, where the first word
        #    is simply absent from the file). Seeding the utterance from this ring
        #    recovers the audio that convinced the detector to open.
        self._preroll: deque = deque()
        self._preroll_samples = 0
        self._preroll_max = int(0.6 * sample_rate)

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
        if self._model is not None:
            return
        t0 = time.perf_counter()
        if self._engine == "moonshine":
            def load_moonshine():
                from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
                m = MoonshineOnnxModel(model_name=self._model_name,
                                       model_precision="float")
                # Warm decode: the first generate() pays ONNX graph
                # optimisation (~1 s); half a second of silence absorbs it
                # here instead of on the user's first sentence.
                m.generate(np.zeros((1, 8000), dtype=np.float32))
                return m, load_tokenizer()
            self._model, self._tokenizer = await asyncio.to_thread(load_moonshine)
        else:
            from faster_whisper import WhisperModel
            self._model = await asyncio.to_thread(
                WhisperModel, self._model_name, device=self._device,
                compute_type=self._compute_type, cpu_threads=self._cpu_threads)
        logger.info(
            f"stt: {self._engine} {self._model_name} "
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
        if self._engine == "moonshine":
            # No hallucination gating needed: moonshine returns '' on silence
            # and noise (verified on this device), so whisper's no_speech_prob
            # machinery has no equivalent here — there is nothing to filter.
            tokens = self._model.generate(audio[None, ...])
            return self._tokenizer.decode_batch(tokens)[0].strip()
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
                # Start the utterance with the pre-roll, not with nothing: the
                # audio VAD spent deciding "this is speech" is part of the speech.
                if self._preroll:
                    self._audio = np.concatenate(list(self._preroll))
                else:
                    self._audio = np.zeros(0, dtype=np.float32)
            self._cancel_speculation()
            self._speaking = True

        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._speaking = False
            t0 = time.perf_counter()
            text = await self._finalise()
            took = (time.perf_counter() - t0) * 1000
            # One uniform line for EVERY transcript, before any gate touches it —
            # wake filter, noise gate, min-chars all act downstream of this. This
            # is the complete record of what the ASR heard, whatever became of it.
            if text:
                logger.info(f"stt: heard {text!r} ({took:.0f}ms)")
            if len(text) >= self._min_chars:
                logger.debug(f"stt: transcript in {took:.0f}ms after turn end")
                if self._wake is not None:
                    text = self._wake.filter(text)
                    if not text:
                        return          # asleep, or a bare wake — never happened
                # NOTE: no _handle_transcription() call — that is a tracing hook on
                # WhisperSTTService, not on the STTService base we subclass.
                await self.push_frame(TranscriptionFrame(
                    text, "", time_now_iso8601(), self._language))
            elif text:
                logger.info(f"stt: discarded too-short {text!r}")

    def _save_utterance(self, audio: np.ndarray) -> Optional[str]:
        """
        Write the EXACT audio whisper is about to see, so a bad transcript can be
        blamed on the right component.

        "It misheard me" has two very different causes — the microphone captured
        something unintelligible, or it captured fine and the model got it wrong —
        and they lead to opposite fixes (hardware/gain vs a bigger model). Listening
        to this file settles it in seconds. It is the audio AFTER VAD segmentation,
        so it also reveals clipped starts and utterances split at a pause.
        """
        if self._save_dir is None or len(audio) < self._sample_rate * 0.1:
            return None
        try:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._utt += 1
            path = self._save_dir / f"{self._utt:03d}_{time.strftime('%H%M%S')}.wav"
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self._sample_rate)
                w.writeframes(pcm.tobytes())
            return str(path)
        except Exception as e:                       # never break a turn over this
            logger.warning(f"stt: could not save utterance audio: {e}")
            return None

    async def _finalise(self) -> str:
        """Use the speculation if it covered the whole utterance, else decode."""
        with self._lock:
            audio = self._audio.copy()
            self._audio = np.zeros(0, dtype=np.float32)

        wav_path = self._save_utterance(audio)
        secs = len(audio) / self._sample_rate
        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0

        def done(text: str) -> str:
            if wav_path:
                # Level is logged next to the text so "quiet mic" is visible without
                # opening the file: ~0.001 is near-silence, ~0.05+ is healthy speech.
                logger.info(f"stt: saved {wav_path} ({secs:.1f}s, rms {rms:.4f}) "
                            f"-> {text.strip()!r}")
            return text

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
                    return done(text)
            else:
                # Too much audio arrived after the guess: it is stale.
                task.cancel()
                self.spec_misses += 1

        self._spec_text = None
        if len(audio) / self._sample_rate < 0.08:
            return done("")
        try:
            return done(await asyncio.to_thread(self._decode, audio))
        except Exception as e:
            logger.warning(f"stt: decode failed: {e}")
            return done("")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Accumulate audio. Transcripts are pushed on UserStoppedSpeaking."""
        if self._model is None:
            yield ErrorFrame("whisper model not loaded")
            return
        chunk = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            if self._speaking:
                # In-turn: grow the utterance. Utterances are seconds long, so the
                # concatenate here is fine — it was the idle path that was not.
                self._audio = np.concatenate([self._audio, chunk])
            else:
                # Idle: keep only the last ~0.6 s, waiting to become a pre-roll.
                self._preroll.append(chunk)
                self._preroll_samples += len(chunk)
                while self._preroll_samples > self._preroll_max and self._preroll:
                    self._preroll_samples -= len(self._preroll.popleft())
        return
        yield  # pragma: no cover — keeps this an async generator
