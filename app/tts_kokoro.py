"""
Pipecat TTS service that speaks through the Kokoro GPU sidecar.

The synthesis itself lives in tools/kokoro_server.py, in a separate venv and
process — see its docstring for why in-process is not possible (onnxruntime
CPU/GPU package conflict, and Silero VAD must stay on the CPU build). This
class is just the thin client: one HTTP POST per clause, PCM back, resampled
to the transport rate.

Latency shape: the sidecar synthesises a whole clause before replying (Kokoro
has no incremental decode worth using at RTF 0.15), so time-to-first-audio for
a clause equals its synth time — measured 357 ms for a typical first clause,
within 100 ms of Piper, at far higher voice quality. The ClauseAggregator
upstream keeps first clauses short, which is exactly what bounds this number.
"""

from typing import AsyncGenerator, Optional
from urllib.request import urlopen

import aiohttp
from loguru import logger

from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import ErrorFrame, Frame, TTSAudioRawFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService


def sidecar_healthy(url: str, timeout: float = 3.0) -> bool:
    """Cheap synchronous probe for startup-time fallback decisions."""
    try:
        with urlopen(url.rstrip("/") + "/health", timeout=timeout) as r:
            return b"ok" in r.read()
    except Exception:
        return False


class KokoroSidecarTTSService(TTSService):
    """Clause-at-a-time TTS against the local Kokoro sidecar."""

    def __init__(self, *, url: str = "http://127.0.0.1:8092",
                 voice: str = "af_heart", speed: float = 1.0, **kwargs):
        kwargs.setdefault("settings",
                          TTSSettings(model="kokoro", voice=voice, language=None))
        super().__init__(push_start_frame=True, push_stop_frames=True, **kwargs)
        self._url = url.rstrip("/")
        self._voice = voice
        self._speed = speed
        self._resampler = create_stream_resampler()
        self._session: Optional[aiohttp.ClientSession] = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame):
        await super().start(frame)
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def stop(self, frame):
        if self._session is not None:
            await self._session.close()
            self._session = None
        await super().stop(frame)

    async def cancel(self, frame):
        if self._session is not None:
            await self._session.close()
            self._session = None
        await super().cancel(frame)

    async def run_tts(self, text: str,
                      context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: synthesising [{text}]")
        try:
            await self.start_tts_usage_metrics(text)
            async with self._session.post(
                    self._url + "/tts",
                    json={"text": text, "voice": self._voice,
                          "speed": self._speed},
                    timeout=aiohttp.ClientTimeout(total=30)) as r:
                if r.status != 200:
                    body = (await r.read())[:120]
                    yield ErrorFrame(f"kokoro sidecar HTTP {r.status}: {body!r}")
                    return
                rate = int(r.headers.get("X-Sample-Rate", "24000"))
                pcm = await r.read()
            await self.stop_ttfb_metrics()
            audio = await self._resampler.resample(pcm, rate, self.sample_rate)
            yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate,
                                   num_channels=1, context_id=context_id)
        except Exception as e:
            # The sidecar being down mid-session lands here; the reply's text
            # is lost but the pipeline survives to the next turn.
            yield ErrorFrame(f"kokoro sidecar: {e}")
        finally:
            await self.stop_ttfb_metrics()
