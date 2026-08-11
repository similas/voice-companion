"""
Local voice + vision companion for Jetson Orin Nano.

    mic -> Silero VAD -> faster-whisper -> [camera?] -> Gemma 3 4B -> Piper -> speaker

Everything runs on this device. Nothing leaves it.

Run with:  ./run.sh          (or: python -m app.main)
"""

import asyncio
import base64
import contextlib
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService

from app import config as config_mod
from app.metrics import MetricsLogger, SpeakingState
from app.observer import LatencyObserver
from app.stt import StrictWhisperSTTService
from app.stt_stream import SpeculativeWhisperSTTService
from app.tts_stream import ClauseAggregator
from app.vad_hook import HookedSileroVAD
from app.vision import Camera, VisionTrigger


def volunteer_as_oom_victim(adj: int = 900) -> None:
    """
    Make THIS process a preferred kill target, so the kernel never picks sshd.

    If memory does run out system-wide, the kernel kills whatever it judges worst,
    scored by oom_score_adj. Measured on this box during a real incident:

        sshd (listener)   adj -1000   score   0     protected
        sshd (session)    adj     0   score 666     <- what actually died
        this agent        adj     0

    With everything at 0 the login session was a legitimate candidate, and losing it
    means losing the ability to even diagnose the problem. Raising our own score to
    900 makes us far more attractive than any sshd, so we die first and you keep the
    shell. llama-server sets 1000 and goes before us, because it is bigger and
    cheaper to restart.

    Raising oom_score_adj needs no privileges; only LOWERING it does. So this works
    as an ordinary user, whereas protecting sshd itself requires root.
    """
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write(str(adj))
        logger.info(f"oom: this process volunteers first (oom_score_adj={adj})")
    except Exception as e:                        # not fatal — it is a safety net
        logger.warning(f"oom: could not set oom_score_adj ({e}); "
                       "a system-wide OOM could pick your SSH session instead")

    # NO CORE DUMPS. This is what turned a crash into a machine-wide freeze.
    # /proc/sys/kernel/core_pattern on this box pipes cores to apport, so when a
    # process dies the kernel streams its entire address space into a Python
    # helper. For a multi-gigabyte process on a 7.6 GB box that is fatal, and the
    # kernel log caught it happening:
    #
    #     do_coredump -> elf_core_dump -> __alloc_pages -> allocation failure
    #
    # i.e. the machine ran out of memory WHILE trying to record why it ran out of
    # memory. A crash we can restart from is fine; a crash that takes the host
    # down with it is not.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception as e:
        logger.warning(f"oom: could not disable core dumps ({e})")


def check_memory(min_free_mb: int) -> bool:
    """
    Refuse to start without enough headroom, rather than becoming the OOM victim.

    On unified memory there is no graceful degradation: when the pool runs out the
    kernel picks a process and kills it. It picked this one, mid-conversation, with
    no message in our log — the user spoke to a process that no longer existed.
    Failing loudly at startup is strictly better than dying silently later.
    """
    mem = {}
    try:
        for ln in open("/proc/meminfo"):
            k, _, v = ln.partition(":")
            mem[k] = int(v.split()[0]) // 1024
    except Exception:
        return True
    avail = mem.get("MemAvailable", 0)
    if avail >= min_free_mb:
        logger.info(f"memory: {avail} MB available")
        return True
    logger.error(f"memory: only {avail} MB available, need {min_free_mb} MB")
    logger.error("  the kernel OOM killer has taken this process before; refusing "
                 "to start rather than die mid-conversation")
    logger.error("  free some: docker ps / ollama ps / "
                 "tools/llama_server.sh stop")
    return False


def resolve_audio_device(spec, want_output: bool) -> Optional[int]:
    """
    Turn a config value into a PortAudio device index.

    Accepts an int (used as-is), None (system default), or a STRING matched against
    device names — which is what you want, because PortAudio indices are not stable.
    Plugging in one USB speaker renumbered everything on this machine: "pulse" moved
    from index 30 to 31, silently pointing the config at "pipewire" instead. Names
    survive that; indices do not.
    """
    if spec is None or isinstance(spec, int):
        return spec
    name = str(spec).strip().lower()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    try:
        exact, partial = None, None
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            chans = d["maxOutputChannels"] if want_output else d["maxInputChannels"]
            if chans < 1:
                continue
            dn = str(d["name"]).lower()
            if dn == name:
                exact = i
                break
            if partial is None and name in dn:
                partial = i
        idx = exact if exact is not None else partial
        if idx is None:
            logger.warning(f"audio: no {'output' if want_output else 'input'} "
                           f"device matching {spec!r}; using the system default")
            return None
        logger.info(f"audio: {'out' if want_output else 'in'} {spec!r} -> index "
                    f"{idx} ({pa.get_device_info_by_index(idx)['name']})")
        return idx
    finally:
        pa.terminate()


def resolve_llm(cfg):
    """
    Pick the LLM endpoint and model name for the configured backend.

    Both backends speak the OpenAI-compatible /v1/chat/completions API, so the
    Pipecat service is identical either way — only the URL and the model name
    change. llama-server serves whatever single model it was launched with and
    ignores the name, so any placeholder works there.
    """
    backend = (cfg.get("llm.backend") or "ollama").strip()
    if backend == "llama_server":
        port = cfg.get("llm.llama_server.port", 8081)
        return backend, f"http://127.0.0.1:{port}/v1", "gemma3"
    return backend, cfg.get("llm.base_url",
                            "http://localhost:11434/v1"), cfg.get("llm.model")


def warm_up_llama_server(base_url: str) -> bool:
    """
    Confirm llama-server is up, on the GPU, and has a warm slot.

    Failing loudly here matters: llama-server serves requests perfectly happily
    with no GPU at roughly half the speed, and the only signal is a warning line
    in its own log at startup.
    """
    import json
    import urllib.request
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(root + "/health", timeout=10) as r:
            if b"ok" not in r.read():
                raise RuntimeError("health not ok")
    except Exception as e:
        logger.error(f"llama-server not reachable at {root}: {e}")
        logger.error("start it with:  tools/llama_server.sh start")
        return False

    log = Path(__file__).resolve().parent.parent / "logs" / "llama-server.log"
    if log.exists() and "no usable GPU found" in log.read_text(errors="ignore"):
        logger.warning("llama-server is running WITHOUT the GPU — expect ~2x "
                       "slower generation. Restart it: "
                       "tools/llama_server.sh restart")

    # One tiny completion to fill the slot and touch the graph caches.
    t0 = time.perf_counter()
    body = json.dumps({"model": "gemma3", "max_tokens": 1, "stream": False,
                       "messages": [{"role": "user", "content": "hi"}]}).encode()
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            r.read()
        logger.info(f"llm warm: llama-server ready in "
                    f"{time.perf_counter()-t0:.1f}s")
        return True
    except Exception as e:
        logger.warning(f"llama-server warm-up failed: {e}")
        return False


def warm_up_llm(base_url: str, model: str, keep_alive: str = "60m"):
    """
    Load the model into GPU memory before the first turn, and pin it there.

    Two separate problems this solves:

    1. Cold load costs ~10 s. Without a warm-up the FIRST thing you say gets a
       10-second pause, which reads as "broken".
    2. Ollama unloads after 5 minutes idle by default. A conversation with gaps in
       it then pays the reload on and off, unpredictably — measured 3100 ms
       time-to-first-token on a turn that should have been ~400 ms. That also
       poisons the latency dataset with bimodal outliers.

    `keep_alive` is an Ollama extension, so this goes to the native /api/generate
    endpoint rather than through the OpenAI-compatible path Pipecat uses.
    """
    import json
    import urllib.request
    url = base_url.rstrip("/").removesuffix("/v1") + "/api/generate"
    body = json.dumps({
        "model": model,
        "prompt": "hi",
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_predict": 1},
    }).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            r.read()
        logger.info(f"llm warm: {model} loaded in {time.perf_counter()-t0:.1f}s, "
                    f"pinned for {keep_alive}")
    except Exception as e:
        logger.warning(f"llm warm-up failed ({e}); first turn will be slow")


def set_output_volume(percent: int):
    """
    Force the default sink's volume at startup.

    Worth doing explicitly: a Bluetooth speaker remembers whatever volume it was
    last set to, including 100% from an earlier test, and 100% on a room speaker
    is genuinely unpleasant. Failure here is non-fatal — it is a convenience, not
    a requirement.
    """
    import subprocess
    try:
        sink = subprocess.run(["pactl", "get-default-sink"], capture_output=True,
                              text=True, timeout=5).stdout.strip()
        if not sink:
            return
        subprocess.run(["pactl", "set-sink-volume", sink, f"{percent}%"],
                       capture_output=True, timeout=5)
        subprocess.run(["pactl", "set-sink-mute", sink, "0"],
                       capture_output=True, timeout=5)
        logger.info(f"output: {sink} at {percent}%")
    except Exception as e:
        logger.debug(f"could not set volume: {e}")


@contextlib.contextmanager
def quiet_stderr():
    """
    Silence stderr at the file-descriptor level.

    PortAudio probes every ALSA plugin when it initialises and prints ~25 lines of
    harmless noise ("Unknown PCM cards.pcm.rear", "Cannot open device /dev/dsp").
    It comes from C, so Python-level redirection does not catch it. Left alone it
    buries real errors — a genuine crash in this program scrolled past 25 lines of
    ALSA complaints and was easy to miss.
    """
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


class EchoGate(FrameProcessor):
    """
    Drops the bot's own voice coming back in through the microphone.

    With a room speaker and an open mic and no acoustic echo cancellation, the bot
    hears itself. Observed live: it answered a question, its own answer was
    transcribed as the next user turn, and it held a six-turn conversation with
    itself — every turn flagged as an interruption, because its own voice also
    triggered barge-in.

    It gates the AUDIO, upstream of STT — not the transcriptions. Gating
    transcriptions does not work, and the reason is worth recording: STT takes
    ~1.2 s, so the bot's own words are transcribed and delivered *after* it has
    stopped speaking and the gate has already reopened. Measured exactly that
    failure: the reply "Four" still got through a transcript-level gate. Dropping
    the audio means STT never hears the bot at all, and the timing race is gone.

    It is deliberately simple and deterministic, which matters for latency
    research: the alternative (PipeWire's webrtc AEC) is available on this box but
    changes the audio path, and any residual echo it leaks would show up as
    phantom turns in the CSV.

    Cost: no barge-in while the bot talks. That is unavoidable with one shared
    speaker — with no AEC, "user interrupting" and "bot hearing itself" are
    literally the same signal. Wear headphones, or set up AEC, if you need
    barge-in (see README).
    """

    def __init__(self, state):
        super().__init__()
        # State is owned by the LatencyObserver, which is the only component that
        # sees frames from BOTH ends of the pipeline. A gate this close to the
        # microphone never receives BotStartedSpeakingFrame — that frame is
        # emitted by the output transport and only travels downstream.
        self._eg_state = state
        self._eg_dropped = 0
        # A turn-start that arrived while the gate was closed. Dropping it outright
        # loses the ENTIRE utterance (STT only buffers between Started and Stopped,
        # and VAD will not re-fire until the user goes silent and starts over) —
        # measured as a ~3 s "it will not hear me" dead zone after every reply. So
        # remember it and inject it the moment the gate reopens; the STT pre-roll
        # ring covers the audio missed in between.
        self._eg_pending_start = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._eg_state.muted() and isinstance(
                frame, (InputAudioRawFrame, UserStartedSpeakingFrame,
                        UserStoppedSpeakingFrame, TranscriptionFrame,
                        InterimTranscriptionFrame)):
            # Drop the mic audio itself, and the VAD turn markers derived from it,
            # so STT is never fed the bot's voice and no phantom turn is opened.
            self._eg_dropped += 1
            if isinstance(frame, UserStartedSpeakingFrame):
                # This is the worst case for the user: they started talking while
                # the gate was closed, the turn-start marker dies here, and the
                # WHOLE utterance is lost — STT only buffers between Started and
                # Stopped. Log it loudly; if this fires often the gate is too slow.
                logger.warning("gate: swallowed a turn START — user began speaking "
                               "while the gate was still closed")
            return  # swallow — do not push downstream

        await self.push_frame(frame, direction)


class NoiseGate(FrameProcessor):
    """
    Drops transcripts that are almost certainly not speech.

    Whisper hallucinates confidently on near-silence. Observed live, from an empty
    room: "Jetson Ornum" and "Plaint to the Illumature to..." — each of which
    opened a turn and got a real reply, so the bot was answering the air. The
    smaller the model the worse this is, and we run `tiny`.

    Two cheap filters catch nearly all of it:
      - too short to be a real request
      - no letters at all (pure punctuation, "...", "[BLANK_AUDIO]")
    Whisper's own no_speech_prob threshold handles the rest and is set on the STT
    service; this is the belt to that braces.
    """

    def __init__(self, min_chars: int = 8):
        super().__init__()
        self._ng_min = min_chars
        self._ng_dropped = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            letters = sum(c.isalpha() for c in text)
            if len(text) < self._ng_min or letters < 3:
                self._ng_dropped += 1
                logger.info(f"noise gate: ignored {text[:40]!r}")
                return

        await self.push_frame(frame, direction)


class ImageMerger(FrameProcessor):
    """
    Folds a captured frame into the user message the aggregator just built.

    Necessary because a chat template needs one user message per turn. Adding the
    image as a separate message produces two user messages back to back, which
    llama-server refuses:

        HTTP 400 Unable to generate parser for this template

    So VisionGate stashes the JPEG and this processor — which sits AFTER the
    context aggregator, once the transcript message exists — rewrites that message
    from a plain string into [image, text].
    """

    def __init__(self, gate: "VisionGate"):
        super().__init__()
        self._im_gate = gate

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, OpenAILLMContextFrame):
            img = self._im_gate.take_pending_image()
            if img:
                ctx = frame.context
                msgs = ctx.get_messages()
                idx = next((i for i in range(len(msgs) - 1, -1, -1)
                            if isinstance(msgs[i], dict)
                            and msgs[i].get("role") == "user"), None)
                if idx is None:
                    logger.warning("vision: no user message to attach the frame to")
                else:
                    text = msgs[idx].get("content")
                    if isinstance(text, list):      # already multimodal
                        parts = text
                    else:
                        parts = [{"type": "text", "text": str(text or "")}]
                    merged = list(msgs)
                    merged[idx] = {"role": "user", "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{img}"}},
                    ] + parts}
                    ctx.set_messages(merged)
                    logger.debug("vision: merged frame into the user message")

        await self.push_frame(frame, direction)


class VisionGate(FrameProcessor):
    """
    Sits between STT and the LLM and decides, per utterance, whether to attach a
    camera frame to the conversation.

    This is the "don't send a frame every turn" requirement. A frame costs Gemma a
    full vision-encoder pass before it can emit its first token, so text-only turns
    stay fast and only explicit requests to look pay for it.
    """

    def __init__(self, camera: Optional[Camera], trigger: VisionTrigger,
                 context: OpenAILLMContext, metrics: MetricsLogger,
                 max_edge: int, quality: int, reuse_thresh: float = 0.0,
                 max_turns: int = 8):
        super().__init__()
        # NOTE the vg_ prefix on every attribute. FrameProcessor already owns
        # several private names — `self._metrics` in particular holds Pipecat's
        # own FrameProcessorMetrics, and shadowing it makes pipeline setup call
        # .setup() on the wrong object:
        #     AttributeError: 'MetricsLogger' object has no attribute 'setup'
        # Prefixing keeps us clear of every current and future internal name.
        self._vg_camera = camera
        self._vg_trigger = trigger
        self._vg_context = context
        self._vg_latency = metrics
        self._vg_max_edge = max_edge
        self._vg_quality = quality
        self._vg_reuse = reuse_thresh
        self._vg_max_turns = max_turns
        self._vg_pending_image = None

    def take_pending_image(self):
        img, self._vg_pending_image = self._vg_pending_image, None
        return img

    @staticmethod
    def _is_image_message(msg) -> bool:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            return False
        return any(isinstance(p, dict) and p.get("type") == "image_url"
                   for p in content)

    def _trim_history(self, max_turns: int) -> int:
        """
        Keep the conversation from growing without bound.

        Nothing trimmed history before this: only images were purged, so text
        messages accumulated for the life of the process. Two consequences, both
        bad and both slow to appear — prefill grows every single turn, and once the
        total passes num_ctx (2048 here) the model starts losing the beginning of
        the conversation with no signal that it happened.

        The system message is always kept; the most recent max_turns*2 messages
        (user + assistant per turn) are kept after it.
        """
        if max_turns <= 0:
            return 0
        try:
            msgs = self._vg_context.get_messages()
        except Exception:
            return 0
        # Count BEFORE mutating. get_messages() hands back a live reference to the
        # context's own list, and set_messages() mutates that same list in place —
        # so reading len(msgs) afterwards reports the NEW length and the delta
        # always comes out as zero.
        n_before = len(msgs)
        head = [m for m in msgs[:1] if isinstance(m, dict)
                and m.get("role") == "system"]
        body = msgs[len(head):]
        keep = max_turns * 2
        if len(body) <= keep:
            return 0
        trimmed = head + body[-keep:]
        self._vg_context.set_messages(trimmed)
        return n_before - len(trimmed)

    def _purge_images(self) -> int:
        """
        Remove every image already in the conversation.

        This is a correctness fix, not an optimisation. A VLM cannot tell a stale
        photo from a live one — an image left in the context gets answered as if it
        were current. Observed in a real conversation: after one "what do you
        see?", later questions ("what number am I showing with my hands?", "what
        colour is my shirt?") were answered confidently from the OLD frame, with
        no new capture taken, and some answers were simply invented.

        So the invariant is: at most one image in context, belonging to the turn
        that asked for it. Anything visual must capture fresh or not answer.
        """
        try:
            msgs = self._vg_context.get_messages()
        except Exception:
            return 0
        kept = [m for m in msgs if not self._is_image_message(m)]
        removed = len(msgs) - len(kept)
        if removed:
            self._vg_context.set_messages(kept)
        return removed

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            # ALWAYS drop previous images first, whether or not this turn uses
            # vision. Otherwise a text-only turn inherits the last picture and
            # answers from it.
            dropped = self._purge_images()
            if dropped:
                logger.debug(f"vision: cleared {dropped} stale image(s)")
            cut = self._trim_history(self._vg_max_turns)
            if cut:
                logger.debug(f"context: trimmed {cut} old message(s)")

            if self._vg_trigger.wants_vision(frame.text):
                ok = await self._attach_frame(frame.text)
                self._vg_trigger.note_result(ok)
            else:
                self._vg_trigger.note_result(False)
                logger.debug("text-only turn")

        await self.push_frame(frame, direction)

    async def _attach_frame(self, text: str) -> bool:
        if self._vg_camera is None:
            logger.warning("vision asked for but camera is unavailable")
            return False

        # Capture happens on a worker thread: cv2 encode is CPU-bound and would
        # otherwise stall the event loop that is also feeding audio.
        got = await asyncio.to_thread(
            self._vg_camera.grab_jpeg, self._vg_max_edge, self._vg_quality,
            self._vg_reuse)
        if got is None:
            logger.warning("vision: failed to grab a frame")
            return False
        jpeg, (w, h), age = got

        # We build the image message by hand rather than using
        # OpenAILLMContext.add_image_frame_message(), because that helper takes RAW
        # pixel bytes and re-encodes them to JPEG through PIL. We already have JPEG
        # from cv2, so going through the helper would mean an extra decode+encode,
        # and it assumes RGB while OpenCV produces BGR (swapped colours).
        # Stash it; ImageMerger folds it into the user message that the context
        # aggregator is about to build.
        #
        # It used to be added here as its OWN user message, which Ollama tolerated
        # but llama-server rejects outright:
        #     HTTP 400 Unable to generate parser for this template
        # Gemma 3's chat template cannot handle two consecutive user messages, and
        # an image-only message followed by the transcript is exactly that.
        self._vg_pending_image = base64.b64encode(jpeg).decode("utf-8")

        self._vg_latency.vision_done()
        # Keep the drain thread hot: follow-up questions usually come right after.
        self._vg_camera.mark_active()
        logger.info(f"vision: attached {w}x{h} frame "
                    f"({len(jpeg) // 1024} KB, {age * 1000:.0f} ms old)")
        return True


async def build_and_run(cfg) -> int:
    # Imported here, not at module top: this raises ImportError with a helpful
    # message if pyaudio is missing, and we want that error explained rather than
    # dumped as a traceback before anything else has initialised.
    try:
        from pipecat.transports.local.audio import (
            LocalAudioTransport, LocalAudioTransportParams)
    except Exception as e:
        logger.error(f"local audio transport unavailable: {e}")
        logger.error("Install PortAudio and PyAudio:")
        logger.error("    sudo apt install -y portaudio19-dev")
        logger.error("    ~/.venvs/voice-companion/bin/pip install pyaudio")
        return 2

    # Before allocating anything substantial, make ourselves the preferred victim.
    volunteer_as_oom_victim(cfg.get("runtime.oom_score_adj", 900))

    if not check_memory(cfg.get("runtime.min_free_mb", 700)):
        return 4

    metrics = MetricsLogger(cfg.path("metrics.csv_dir", "logs"),
                            console=cfg.get("metrics.console", True))
    logger.info(f"latency CSV: {metrics.path}")

    # Do this first: cold start costs ~10 s, and paying it here rather than on the
    # user's first sentence is the difference between "instant" and "broken".
    backend, llm_url, llm_model = resolve_llm(cfg)
    logger.info(f"llm backend: {backend} -> {llm_url} ({llm_model})")
    if backend == "llama_server":
        if not warm_up_llama_server(llm_url):
            return 3
    else:
        warm_up_llm(llm_url, llm_model, cfg.get("llm.keep_alive", "60m"))

    # ---- camera ------------------------------------------------------------
    camera = None
    if cfg.get("vision.enabled", True):
        camera = Camera(
            device=cfg.get("camera.device", 0),
            width=cfg.get("camera.width", 1280),
            height=cfg.get("camera.height", 720),
            poll_fps=cfg.get("camera.poll_fps", 4),
            idle_fps=cfg.get("camera.idle_fps", 0.4),
            active_secs=cfg.get("camera.active_secs", 25),
        )
        if not camera.start():
            logger.warning("continuing without vision")
            camera = None
    trigger = VisionTrigger(cfg.get("vision.triggers", []),
                            enabled=cfg.get("vision.enabled", True),
                            smart=cfg.get("vision.smart_detect", True),
                            sticky=cfg.get("vision.sticky_followups", True))

    # ---- transport (mic + speaker + VAD) -----------------------------------
    vad_cls = (HookedSileroVAD if cfg.get("stt.speculative", True)
               else SileroVADAnalyzer)
    vad = vad_cls(params=VADParams(
        confidence=cfg.get("vad.confidence", 0.7),
        start_secs=cfg.get("vad.start_secs", 0.2),
        stop_secs=cfg.get("vad.stop_secs", 0.8),
        min_volume=cfg.get("vad.min_volume", 0.6),
    ))
    sample_rate = cfg.get("audio.sample_rate", 16000)
    out_rate = cfg.get("audio.output_sample_rate", 48000)
    params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        # Input stays at 16 kHz: that is what Silero VAD and Whisper both want.
        audio_in_sample_rate=sample_rate,
        # Output runs at the Bluetooth sink's NATIVE rate (48 kHz). Sending 16 kHz
        # made PipeWire resample 16k->48k on top of Pipecat's 22.05k->16k from
        # Piper — two conversions, and the extra one produced audible choppiness
        # over A2DP. One conversion, done once, at the device's own rate.
        audio_out_sample_rate=out_rate,
        audio_in_channels=cfg.get("audio.channels", 1),
        audio_out_channels=1,
        # Bigger output chunks. Bluetooth A2DP has far more latency jitter than a
        # local ALSA card, and 10 ms chunks underrun on it — which is heard as
        # crackling. Larger chunks cost a little latency and buy stability.
        audio_out_10ms_chunks=cfg.get("audio.out_chunk_10ms", 4),
        # vad_enabled / vad_audio_passthrough are deprecated in 0.0.108:
        # audio_in_enabled above covers the first, and passthrough is now always
        # on. vad_analyzer here still works (it warns about moving to
        # LLMUserAggregator/VADProcessor, which is a 1.x-era API).
        vad_analyzer=vad,
        input_device_index=resolve_audio_device(cfg.get("audio.input_device"),
                                                False),
        output_device_index=resolve_audio_device(cfg.get("audio.output_device"),
                                                 True),
    )
    set_output_volume(cfg.get("audio.output_volume", 55))
    with quiet_stderr():   # PortAudio's ALSA probe noise, see quiet_stderr()
        transport = LocalAudioTransport(params)

    # ---- STT ---------------------------------------------------------------
    if cfg.get("stt.speculative", True):
        # Decodes during the VAD hangover instead of after it — see app/stt_stream.
        stt = SpeculativeWhisperSTTService(
            model=cfg.get("stt.model", "tiny"),
            device=cfg.get("stt.device", "cpu"),
            compute_type=cfg.get("stt.compute_type", "int8"),
            language=cfg.get("stt.language", "en"),
            no_speech_prob=cfg.get("stt.no_speech_prob", 0.6),
            min_chars=cfg.get("stt.min_chars", 8),
            cpu_threads=cfg.get("stt.cpu_threads", 2),
            sample_rate=sample_rate,
            # Dump the exact audio whisper sees, so "it misheard me" can be pinned
            # on the microphone or on the model instead of argued about.
            save_dir=(cfg.path("stt.save_audio_dir", "logs/turns")
                      if cfg.get("stt.save_audio", False) else None),
        )
        if isinstance(vad, HookedSileroVAD):
            vad.set_callbacks(on_maybe_stopped=stt.on_maybe_stopped,
                              on_resumed=stt.on_resumed)
    else:
        stt = StrictWhisperSTTService(
            settings=WhisperSTTService.Settings(model=cfg.get("stt.model", "tiny")),
            device=cfg.get("stt.device", "cpu"),
            compute_type=cfg.get("stt.compute_type", "int8"),
            # Whisper's own confidence that a segment contains no speech. Above
            # this the segment is discarded rather than hallucinated into words.
            no_speech_prob=cfg.get("stt.no_speech_prob", 0.6),
            min_chars=cfg.get("stt.min_chars", 8),
        )

    # ---- LLM ---------------------------------------------------------------
    # temperature and max_tokens go through the OpenAI-compatible API, which both
    # backends honour. num_ctx does NOT — it is a server-side setting; see the
    # README section "Context length" for how to change it (it affects whether
    # the model fits on the GPU, so it is worth knowing about).
    #
    # USE THE RESOLVED VALUES. This previously read cfg("llm.base_url") and
    # cfg("llm.model") directly, which meant resolve_llm() above was computed,
    # logged, and then thrown away — the process announced
    #
    #     llm backend: llama_server -> http://127.0.0.1:8081/v1
    #
    # while actually building a client for Ollama on :11434. Every single user turn
    # therefore hit Ollama, which had no model resident, so Ollama loaded a full
    # second copy of gemma3 plus its 942 MB vision projector to serve it. On 7.6 GB
    # of shared memory that froze the machine hard enough to need a power cycle —
    # four times — and the request never returned, so the agent also never replied.
    # The log looked healthy throughout because our llama-server WAS up and warm;
    # nothing was ever sending it traffic except the warm-up ping.
    llm = OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=llm_model,
            temperature=cfg.get("llm.temperature", 0.7),
            max_tokens=cfg.get("llm.max_tokens", 150),
        ),
        base_url=llm_url,
    )
    logger.info(f"llm client -> {llm_url} (model {llm_model})")
    context = OpenAILLMContext(messages=[
        {"role": "system", "content": cfg.get("llm.system_prompt", "").strip()},
    ])
    aggregator = llm.create_context_aggregator(context)

    # ---- TTS ---------------------------------------------------------------
    tts_kwargs = {}
    if cfg.get("tts.clause_streaming", True):
        tts_kwargs["text_aggregator"] = ClauseAggregator(
            min_clause_chars=cfg.get("tts.min_clause_chars", 18),
            first_chunk_chars=cfg.get("tts.first_chunk_chars", 46),
        )
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(
            voice=cfg.get("tts.voice", "en_US-lessac-medium")),
        download_dir=cfg.path("tts.models_dir", "models"),
        use_cuda=cfg.get("tts.use_cuda", False),
        sample_rate=out_rate,      # match the transport; avoids a second resample
        **tts_kwargs,
    )

    gate = VisionGate(camera, trigger, context, metrics,
                      max_edge=cfg.get("camera.max_edge", 896),
                      quality=cfg.get("camera.jpeg_quality", 85),
                      reuse_thresh=cfg.get("camera.reuse_threshold", 0.0),
                      max_turns=cfg.get("llm.max_history_turns", 8))

    speaking = SpeakingState(cfg.get("audio.echo_tail_secs", 0.4))
    stages = [transport.input()]  # mic -> audio frames, VAD marks turn boundaries
    if cfg.get("audio.suppress_echo", True):
        # MUST be before stt. Placed after it, the gate is downstream of the very
        # thing it needs to starve: STT already consumed the bot's audio and
        # emitted a transcript, which is exactly the bug it exists to prevent.
        stages.append(EchoGate(speaking))
    stages += [
        stt,                    # audio -> TranscriptionFrame
        NoiseGate(cfg.get("stt.min_chars", 8)),   # discard hallucinated noise
        gate,                   # decide whether to attach a camera frame
        aggregator.user(),      # transcript -> conversation context
        ImageMerger(gate),      # fold any captured frame into that user message
        llm,                    # context -> streamed reply tokens
        tts,                    # tokens -> PCM audio
        transport.output(),     # PCM -> speaker
        aggregator.assistant(), # reply -> conversation context
    ]
    pipeline = Pipeline(stages)

    task = PipelineTask(
        pipeline,
        # NEVER self-terminate. Pipecat defaults to cancel_on_idle_timeout=True
        # with idle_timeout_secs=300, so after five minutes of silence the whole
        # pipeline cancelled itself and the process exited — you would walk up to
        # a companion that had quietly died. An always-on assistant is idle almost
        # all of the time by definition; that is not a fault condition.
        idle_timeout_secs=cfg.get("runtime.idle_timeout_secs", 300),
        cancel_on_idle_timeout=cfg.get("runtime.cancel_on_idle", False),
        params=PipelineParams(
            allow_interruptions=cfg.get("interruption.enabled", True),
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[LatencyObserver(metrics, speaking, cfg.get("audio.suppress_echo", True))],
    )

    runner = PipelineRunner(handle_sigint=False)

    # Graceful ctrl-c: end the pipeline so the CSV is flushed and the camera
    # released, instead of dying mid-write.
    loop = asyncio.get_running_loop()
    stopping = False

    def _stop():
        nonlocal stopping
        if stopping:
            return
        stopping = True
        logger.info("stopping...")
        loop.create_task(task.queue_frame(EndFrame()))

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    banner(cfg, metrics)
    try:
        await runner.run(task)
    finally:
        metrics.close()
        if camera:
            camera.stop()
        logger.info(f"latency data written to {metrics.path}")
    return 0


def banner(cfg, metrics):
    # Report the backend we ACTUALLY resolved, not a hardcoded guess. This line
    # read "<llm.model> via Ollama" unconditionally, so it announced a stale model
    # name and the wrong engine while every request went to llama-server.
    _backend, _base, _ = resolve_llm(cfg)
    if _backend == "llama_server":
        _brain = f"{Path(cfg.get('llm.llama_server.model_path', '')).stem} via llama-server"
    else:
        _brain = f"{cfg.get('llm.model')} via Ollama"
    print(f"""
  ┌─ local voice + vision companion ──────────────────────────────
  │  STT     faster-whisper {cfg.get('stt.model')} ({cfg.get('stt.compute_type')}, {cfg.get('stt.device')})
  │  brain   {_brain}
  │  TTS     Piper {cfg.get('tts.voice')}
  │  vision  {'on — camera used only when you ask' if cfg.get('vision.enabled') else 'off'}
  │  barge-in {'enabled' if cfg.get('interruption.enabled') else 'disabled'}
  │  metrics {metrics.path.name}
  └───────────────────────────────────────────────────────────────

  Speak when ready. Ctrl-C to stop.
""", flush=True)


def main() -> int:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss.SSS}</green> <level>{level: <7}</level> {message}")
    cfg = config_mod.load()
    try:
        return asyncio.run(build_and_run(cfg))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
