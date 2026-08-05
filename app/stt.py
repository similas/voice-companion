"""
A Whisper STT service that does not invent speech.

Pipecat's WhisperSTTService calls faster-whisper with defaults, and two of those
defaults make a small model hallucinate badly in a live voice loop:

  condition_on_previous_text=True   feeds the previous transcript back in as a
                                    prompt. On silence or noise, Whisper then
                                    writes a plausible CONTINUATION of the
                                    conversation instead of nothing.
  vad_filter=False                  no speech/silence gating before decoding, so
                                    silence gets decoded into words.

Observed with the stock settings, from an empty room:

    "Jetson Ornum"
    "Plaint to the Illumature to..."
    "Be my desktop, that's what you get."
    "The answer is four."     <- while the bot had NOT yet said the answer

That last one mattered: it looks exactly like acoustic echo of the bot's reply,
and it sent me looking for an echo leak that was not there. The bot had spoken
zero times at that point (SpeakingState.started_count == 1 from an earlier turn),
so the sentence could only have been invented. It is a continuation of the
question Whisper had just transcribed — which is precisely what
condition_on_previous_text produces.

So: no context carry-over, VAD gating on, and a single greedy temperature (the
default temperature fallback ladder retries with increasing randomness, which is
another way invented text appears).
"""

import asyncio
from typing import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.utils.time import time_now_iso8601


class StrictWhisperSTTService(WhisperSTTService):
    """WhisperSTTService with hallucination suppression."""

    def __init__(self, *args, min_chars: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._min_chars = min_chars

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()

        # 16-bit signed PCM -> float32 in [-1, 1]
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments, info = await asyncio.to_thread(
            self._model.transcribe,
            audio_float,
            language=self._settings.language,
            beam_size=1,                      # greedy: fastest, and we do not
                                              # need alternatives for short speech
            temperature=0.0,                  # no fallback ladder — the retries at
                                              # higher temperature invent text
            condition_on_previous_text=False,  # THE important one, see module docs
            vad_filter=True,                  # drop silence before decoding
            vad_parameters={"min_silence_duration_ms": 300},
            no_speech_threshold=0.6,
        )

        text = ""
        for seg in segments:
            # Pipecat's own no_speech_prob gate, kept.
            if seg.no_speech_prob < self._settings.no_speech_prob:
                text += f"{seg.text} "

        await self.stop_processing_metrics()

        text = text.strip()
        if len(text) < self._min_chars:
            if text:
                logger.info(f"stt: discarded too-short {text!r}")
            return

        if text:
            await self._handle_transcription(text, True, self._settings.language)
            logger.debug(f"transcript: {text}")
            yield TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), self._settings.language,
            )
