#!/usr/bin/env python3
"""
Component + end-to-end benchmark for the local voice/vision companion.

Runs without a human: speech prompts are synthesised with the same Piper voice the
companion speaks with, so STT is measured on real audio rather than a corpus that
does not resemble the deployment.

What it measures
----------------
  stt          faster-whisper wall time per utterance, and word error rate
  llm_text     time-to-first-token and generation rate, text-only
  llm_vision   the same with a live camera frame attached
  tts          Piper synthesis time and real-time factor
  camera       frame grab + JPEG encode, and frame freshness
  trigger      accuracy of the "should I look?" classifier
  system       memory, GPU offload, model/versions

TTFA (time to first audio) is then composed from the measured stage medians:

    TTFA = stt + [vision_capture] + llm_first_token + tts_first_audio

which is exactly what the live pipeline's t_audio_playback_start reports, so the
two are directly comparable. Composed numbers are labelled as such; measured
live numbers come from logs/latency_*.csv and are reported separately.

Usage
-----
    python bench/benchmark.py                    # full run
    python bench/benchmark.py --reps 5           # fewer repetitions
    python bench/benchmark.py --no-vision        # skip camera + VLM
    python bench/benchmark.py --tag v0.1         # output directory name
"""

import argparse
import base64
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time
import urllib.request
import wave
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_VENV = Path.home() / ".venvs" / "voice-companion"
if _VENV.exists() and Path(sys.prefix) != _VENV:
    os.execv(str(_VENV / "bin" / "python"),
             [str(_VENV / "bin" / "python"), str(Path(__file__).resolve())]
             + sys.argv[1:])

from app import config as config_mod  # noqa: E402

# Prompts a person would actually say to this thing. Kept short, because short
# utterances are what a voice loop sees and long-form benchmarks mislead.
VOICE_PROMPTS = [
    "What is the capital of France?",
    "Can you hear me clearly?",
    "Tell me one interesting fact about the ocean.",
    "What time does the sun usually set in summer?",
    "Count from one to five for me.",
    "How would you describe the colour blue to someone?",
    "Give me a short suggestion for dinner tonight.",
    "What is two plus two?",
]
VISION_PROMPTS = [
    "What do you see right now?",
    "Describe this room briefly.",
    "What objects are on the table?",
    "What am I holding right now?",
]

# (utterance, should_use_camera) — drawn from a real recorded conversation, which
# is where the original phrase-list approach was found to be far too narrow.
TRIGGER_CASES = [
    ("Can you hear me?", False),
    ("What do you see right now?", True),
    ("Count from 1 to 10.", False),
    ("What is the number that I'm showing with my hands?", True),
    ("What am I holding right now?", True),
    ("How are you right now?", False),
    ("What am I doing right now?", True),
    ("What is written on this marker?", True),
    ("What is the color of my shirt?", True),
    ("List all the things you are seeing in my living room.", True),
    ("What is the capital of France?", False),
    ("Tell me a joke.", False),
    ("What time is it?", False),
    ("How is the weather?", False),
    ("What is the brand of the marker that I'm holding right now?", True),
    ("Read this label for me.", True),
    ("Name the objects on the table.", True),
]


def p(msg=""):
    print(msg, flush=True)


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    i = min(len(s) - 1, max(0, int(round(q / 100 * (len(s) - 1)))))
    return s[i]


def stats(vals):
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": round(min(vals), 1),
        "median": round(st.median(vals), 1),
        "mean": round(st.fmean(vals), 1),
        "p90": round(pct(vals, 90), 1),
        "max": round(max(vals), 1),
        "stdev": round(st.pstdev(vals), 1) if len(vals) > 1 else 0.0,
    }


def wer(ref: str, hyp: str) -> float:
    """Word error rate via Levenshtein on word sequences."""
    def norm(t):
        keep = "".join(c.lower() if (c.isalnum() or c.isspace()) else " " for c in t)
        return keep.split()
    r, h = norm(ref), norm(hyp)
    if not r:
        return 0.0
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)] / len(r)


# ---------------------------------------------------------------------------
# system facts
# ---------------------------------------------------------------------------
def sh(cmd, timeout=15):
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout).stdout.strip()
    except Exception:
        return ""


def system_info(cfg):
    mem = {}
    for ln in open("/proc/meminfo"):
        k, _, v = ln.partition(":")
        mem[k] = int(v.split()[0]) // 1024
    cpu_model = ""
    for ln in open("/proc/cpuinfo"):
        if "model name" in ln or "Model" in ln:
            cpu_model = ln.split(":", 1)[1].strip()
            break
    l4t = ""
    try:
        l4t = open("/etc/nv_tegra_release").read().strip().splitlines()[0]
    except Exception:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": "NVIDIA Jetson Orin Nano Super (8GB)",
        "cpu": cpu_model or platform.processor(),
        "cores": os.cpu_count(),
        "gpu": "Ampere, 1024 CUDA cores, compute 8.7 (unified memory)",
        "l4t": l4t,
        "kernel": platform.release(),
        "python": platform.python_version(),
        "mem_total_mb": mem.get("MemTotal"),
        "mem_available_mb": mem.get("MemAvailable"),
        "power_mode": sh(["nvpmodel", "-q"]).replace("\n", " ")[:80],
        "ollama_version": sh(["ollama", "--version"]),
        "ollama_ps": sh(["ollama", "ps"]),
        "models": {
            "stt": (f"moonshine {cfg.get('stt.model')} (float, cpu)"
                    if cfg.get("stt.engine") == "moonshine" else
                    f"faster-whisper {cfg.get('stt.model')} "
                    f"({cfg.get('stt.compute_type')}, {cfg.get('stt.device')})"),
            "llm": f"{cfg.get('llm.model')} via {(cfg.get('llm.backend') or 'ollama')}",
            "tts": (f"kokoro {cfg.get('tts.kokoro.voice')} (GPU sidecar)"
                    if cfg.get("tts.engine") == "kokoro" else
                    f"piper {cfg.get('tts.voice')}"),
            "vad": "silero (onnxruntime)",
        },
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def synth_prompts(cfg, texts, outdir):
    """Speak each prompt with Piper so STT is measured on realistic audio."""
    from piper import PiperVoice
    voice_path = cfg.path("tts.models_dir") / f"{cfg.get('tts.voice')}.onnx"
    voice = PiperVoice.load(str(voice_path))
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for i, text in enumerate(texts):
        path = outdir / f"prompt_{i:02d}.wav"
        with wave.open(str(path), "wb") as wf:
            voice.synthesize_wav(text, wf)
        with wave.open(str(path), "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        out.append({"text": text, "path": path, "duration_s": round(dur, 2)})
    return out


def bench_tts(cfg, reps):
    # A typical spoken reply from this assistant: one to two short sentences.
    sample = "That's four. Is there anything else you would like to know?"

    if cfg.get("tts.engine") == "kokoro":
        # The sidecar synthesises a whole clause per request, so one request's
        # wall time IS time-to-first-audio for that clause — same quantity the
        # piper branch measures.
        port = cfg.get("tts.kokoro.port", 8092)
        url = f"http://127.0.0.1:{port}/tts"
        body = json.dumps({"text": sample,
                           "voice": cfg.get("tts.kokoro.voice", "af_heart")}).encode()
        times, rtfs = [], []
        for _ in range(reps + 1):               # first rep warms, then measure
            t0 = time.perf_counter()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                pcm = r.read()
                sr = int(r.headers.get("X-Sample-Rate", "24000"))
            dt = (time.perf_counter() - t0) * 1000
            times.append(dt)
            rtfs.append(dt / 1000 / (len(pcm) / 2 / sr))
        times, rtfs = times[1:], rtfs[1:]
        return {"load_ms": 0.0, "synth_ms": stats(times),
                "realtime_factor": round(st.median(rtfs), 3),
                "sample_text": sample, "engine": "kokoro (GPU sidecar)"}

    from piper import PiperVoice
    voice_path = cfg.path("tts.models_dir") / f"{cfg.get('tts.voice')}.onnx"
    t0 = time.perf_counter()
    voice = PiperVoice.load(str(voice_path))
    load_ms = (time.perf_counter() - t0) * 1000

    times, rtfs = [], []
    tmp = Path("/tmp/bench_tts.wav")
    for _ in range(reps):
        t0 = time.perf_counter()
        with wave.open(str(tmp), "wb") as wf:
            voice.synthesize_wav(sample, wf)
        dt = (time.perf_counter() - t0) * 1000
        with wave.open(str(tmp), "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        times.append(dt)
        rtfs.append(dt / 1000 / dur)
    tmp.unlink(missing_ok=True)
    return {"load_ms": round(load_ms, 1), "synth_ms": stats(times),
            "realtime_factor": round(st.median(rtfs), 3),
            "sample_text": sample, "engine": "piper"}


def bench_stt(cfg, prompts, reps):
    t0 = time.perf_counter()
    if cfg.get("stt.engine") == "moonshine":
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer
        from moonshine_onnx.transcribe import load_audio
        model = MoonshineOnnxModel(model_name=cfg.get("stt.model", "tiny"),
                                   model_precision="float")
        tok = load_tokenizer()
        load_ms = (time.perf_counter() - t0) * 1000

        def transcribe(path):
            return tok.decode_batch(model.generate(load_audio(str(path))))[0].strip()
    else:
        from faster_whisper import WhisperModel
        model = WhisperModel(cfg.get("stt.model"), device=cfg.get("stt.device"),
                             compute_type=cfg.get("stt.compute_type"))
        load_ms = (time.perf_counter() - t0) * 1000

        def transcribe(path):
            segs, _ = model.transcribe(
                str(path), language=cfg.get("stt.language", "en"),
                beam_size=1, temperature=0.0,
                condition_on_previous_text=False, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
                no_speech_threshold=0.6)
            return " ".join(s.text for s in segs).strip()

    transcribe(prompts[0]["path"])          # warm

    times, wers, rtfs, examples = [], [], [], []
    for item in prompts:
        best = None
        text = ""
        for _ in range(reps):
            t0 = time.perf_counter()
            text = transcribe(item["path"])
            dt = (time.perf_counter() - t0) * 1000
            best = dt if best is None else min(best, dt)
        times.append(best)
        rtfs.append(best / 1000 / item["duration_s"])
        e = wer(item["text"], text)
        wers.append(e)
        examples.append({"said": item["text"], "heard": text,
                         "wer": round(e, 3), "ms": round(best, 1)})
    return {
        "load_ms": round(load_ms, 1),
        "latency_ms": stats(times),
        "realtime_factor": round(st.median(rtfs), 3),
        "wer_mean": round(st.fmean(wers), 4),
        "wer_median": round(st.median(wers), 4),
        "exact_match_rate": round(sum(1 for w in wers if w == 0) / len(wers), 3),
        "examples": examples,
    }


def resolve_backend(cfg):
    """Endpoint + model name for whichever backend config.yaml selects."""
    backend = (cfg.get("llm.backend") or "ollama").strip()
    if backend == "llama_server":
        port = cfg.get("llm.llama_server.port", 8081)
        # llama-server serves whatever single model it was started with and ignores
        # the "model" field, so this name is purely for logs and result files. It was
        # hardcoded to "gemma3", which meant a run against a completely different
        # model still recorded itself as gemma3. Derive it from the configured path.
        path = (cfg.get("llm.llama_server.model_path") or "").strip()
        name = Path(path).stem if path else (cfg.get("llm.model") or "llama-server")
        return backend, f"http://127.0.0.1:{port}/v1", name
    return backend, cfg.get("llm.base_url",
                            "http://localhost:11434/v1"), cfg.get("llm.model")


def llm_stream(base_url, model, messages, max_tokens, images=None):
    """
    One streamed chat completion over the OpenAI-compatible API.

    Deliberately uses /v1/chat/completions for BOTH backends so the comparison is
    apples to apples — the earlier version called Ollama's native /api/chat, which
    is a different code path with different overhead.

    Returns (ttft_ms, total_ms, text, n_tok). n_tok is counted from the stream
    because llama-server does not report eval_count the way Ollama does.
    """
    msgs = [dict(m) for m in messages]
    if images:
        # OpenAI-shaped multimodal content: text part + image part.
        last = msgs[-1]
        last["content"] = [{"type": "text", "text": last.get("content", "")}] + [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{im}"}}
            for im in images]
    body = json.dumps({"model": model, "messages": msgs, "stream": True,
                       "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            s = raw.decode().strip()
            if not s.startswith("data:") or s == "data: [DONE]":
                continue
            try:
                d = json.loads(s[5:])
            except Exception:
                continue
            tok = (d.get("choices") or [{}])[0].get("delta", {}).get("content")
            if tok:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                chunks.append(tok)
    return ttft, (time.perf_counter() - t0) * 1000, "".join(chunks).strip(), len(chunks)


def bench_llm(cfg, prompts, reps, image_b64=None, label="text",
              images_pool=None):
    """
    Measure TTFT / throughput.

    images_pool: for vision runs, a list of DISTINCT frames. Each call takes a
    different one, because repeating the same (prompt, image) pair hits
    llama.cpp's exact-prefix cache and produces a time real conversation never
    sees. Measured proof: same image with 5 different questions ran 4900-5250 ms
    every time, while a repeated identical pair returned in ~300 ms. Reporting the
    latter as "vision latency" would be simply false.
    """
    _, base, model = resolve_backend(cfg)
    sysmsg = {"role": "system", "content": cfg.get("llm.system_prompt", "").strip()}
    max_tok = cfg.get("llm.max_tokens", 60)

    # warm-up so the first measured call is not paying a model load
    llm_stream(base, model, [sysmsg, {"role": "user", "content": "hi"}], 4,
                  images=[image_b64] if image_b64 else None)

    ttfts, totals, rates, outs = [], [], [], []
    call_i = 0
    for text in prompts:
        for _ in range(reps):
            msgs = [sysmsg, {"role": "user", "content": text}]
            if images_pool:
                img = images_pool[call_i % len(images_pool)]
            else:
                img = image_b64
            call_i += 1
            ttft, total, reply, n = llm_stream(
                base, model, msgs, max_tok, images=[img] if img else None)
            if ttft is None:
                continue
            ttfts.append(ttft)
            totals.append(total)
            gen_ms = max(total - ttft, 1)
            if n:
                rates.append(n / (gen_ms / 1000))
            outs.append({"prompt": text, "reply": reply[:160],
                         "ttft_ms": round(ttft, 1), "tokens": n})
    return {
        "mode": label,
        "ttft_ms": stats(ttfts),
        "total_ms": stats(totals),
        "tokens_per_s": round(st.median(rates), 2) if rates else None,
        "samples": outs[:8],
    }


def bench_stt_speculative(cfg, prompts, stt_bench):
    """
    Latency the user actually waits for STT, when speculative decoding is on.

    The raw decode time (bench_stt) is not what a person experiences. Turn detection
    holds `stop_secs` of silence before ending the turn, and with speculation the
    decode runs inside that window — so the wait after the turn ends is only whatever
    is left. Measured here by replaying the same audio in 20 ms frames, firing the
    VAD hangover hook, then timing from turn-end to transcript.
    """
    if not cfg.get("stt.speculative", True):
        return None
    import asyncio
    import numpy as np
    from pipecat.frames.frames import (StartFrame, TranscriptionFrame,
                                       UserStartedSpeakingFrame,
                                       UserStoppedSpeakingFrame)
    from pipecat.processors.frame_processor import FrameDirection
    from app.stt_stream import SpeculativeWhisperSTTService

    sr = cfg.get("audio.sample_rate", 16000)
    stop_secs = cfg.get("vad.stop_secs", 0.5)

    def as_pcm16(path):
        with wave.open(str(path), "rb") as wf:
            a = np.frombuffer(wf.readframes(wf.getnframes()),
                              dtype=np.int16).astype(np.float32) / 32768.0
            wsr = wf.getframerate()
        if wsr != sr:
            idx = (np.arange(int(len(a) * sr / wsr)) * wsr / sr).astype(int)
            a = a[np.clip(idx, 0, len(a) - 1)]
        return (a * 32768).astype(np.int16)

    async def run():
        got = {}

        async def capture(frame, direction=None):
            if isinstance(frame, TranscriptionFrame):
                got["t"] = time.perf_counter()
                got["text"] = frame.text

        svc = SpeculativeWhisperSTTService(
            engine=cfg.get("stt.engine", "whisper"),
            model=cfg.get("stt.model", "tiny"), device=cfg.get("stt.device", "cpu"),
            compute_type=cfg.get("stt.compute_type", "int8"), sample_rate=sr)
        svc.push_frame = capture
        await svc.start(StartFrame(audio_in_sample_rate=sr, audio_out_sample_rate=sr))

        waits, texts = [], []
        step = int(0.02 * sr)
        for n, item in enumerate(prompts):
            pcm = as_pcm16(item["path"])
            got.clear()
            await svc.process_frame(UserStartedSpeakingFrame(),
                                    FrameDirection.DOWNSTREAM)
            for i in range(0, len(pcm), step):
                async for _ in svc.run_stt(pcm[i:i+step].tobytes()):
                    pass
                await asyncio.sleep(0.02)
            svc.on_maybe_stopped()             # VAD enters the hangover
            await asyncio.sleep(stop_secs)
            t0 = time.perf_counter()
            await svc.process_frame(UserStoppedSpeakingFrame(),
                                    FrameDirection.DOWNSTREAM)
            if "t" in got and n > 0:           # skip the first, it warms caches
                waits.append((got["t"] - t0) * 1000)
                texts.append({"said": item["text"], "heard": got.get("text", "")})
        return waits, texts, svc.spec_hits, svc.spec_misses

    waits, texts, hits, misses = asyncio.run(run())
    if not waits:
        return None
    raw = (stt_bench or {}).get("latency_ms", {}).get("median")
    return {
        "wait_after_turn_end_ms": stats(waits),
        "raw_decode_median_ms": raw,
        "saved_ms": round(raw - st.median(waits), 1) if raw else None,
        "speculation_hits": hits, "speculation_misses": misses,
        "wer_mean": round(st.fmean(wer(t["said"], t["heard"]) for t in texts), 4),
        "examples": texts[:4],
    }


def bench_camera(cfg, reps):
    from app.vision import Camera
    cam = Camera(cfg.get("camera.device"), cfg.get("camera.width"),
                 cfg.get("camera.height"), cfg.get("camera.poll_fps"),
                 cfg.get("camera.idle_fps", 0.4),
                 cfg.get("camera.active_secs", 25))
    if not cam.start():
        return None, None
    time.sleep(1.5)
    grabs, ages, sizes, pool = [], [], [], []
    jpeg_b64 = None
    dims = None
    for _ in range(reps):
        t0 = time.perf_counter()
        got = cam.grab_jpeg(cfg.get("camera.max_edge"), cfg.get("camera.jpeg_quality"))
        dt = (time.perf_counter() - t0) * 1000
        if not got:
            continue
        jpeg, (w, h), age = got
        grabs.append(dt)
        ages.append(age * 1000)
        sizes.append(len(jpeg) / 1024)
        jpeg_b64 = base64.b64encode(jpeg).decode()
        pool.append(jpeg_b64)
        dims = f"{w}x{h}"
        time.sleep(0.6)
    reuse_thresh = cfg.get("camera.reuse_threshold", 0.0)
    reuse_stats = None
    if reuse_thresh > 0:
        # Does the reuse path actually fire on a static scene, and does it return
        # byte-identical bytes (which is what lets llama.cpp skip the encoder)?
        cam.reuse_hits = 0
        first = cam.grab_jpeg(cfg.get("camera.max_edge"),
                              cfg.get("camera.jpeg_quality"), reuse_thresh)
        again = [cam.grab_jpeg(cfg.get("camera.max_edge"),
                              cfg.get("camera.jpeg_quality"), reuse_thresh)
                 for _ in range(4)]
        identical = sum(1 for g in again if g and first and g[0] is first[0])
        reuse_stats = {"threshold": reuse_thresh, "probes": len(again),
                       "byte_identical": identical,
                       "hits": cam.reuse_hits}
    cam.stop()
    return {
        "capture_ms": stats(grabs),
        "frame_reuse": reuse_stats,
        "frame_age_ms": stats(ages),
        "jpeg_kb": stats(sizes),
        "encoded_size": dims,
        "capture_resolution": f"{cfg.get('camera.width')}x{cfg.get('camera.height')}",
        "distinct_frames_for_vision_bench": len(set(pool)),
    }, jpeg_b64, pool


def bench_trigger(cfg):
    from app.vision import VisionTrigger
    t = VisionTrigger(cfg.get("vision.triggers"), True,
                      cfg.get("vision.smart_detect", True),
                      cfg.get("vision.sticky_followups", True))
    wrong = []
    for text, expect in TRIGGER_CASES:
        t.note_result(False)
        got = t.wants_vision(text)
        if got != expect:
            wrong.append({"text": text, "expected": expect, "got": got})
    t.note_result(True)
    sticky = t.wants_vision("How about now?")
    n = len(TRIGGER_CASES)
    return {"cases": n, "correct": n - len(wrong),
            "accuracy": round((n - len(wrong)) / n, 3),
            "sticky_followup_works": bool(sticky), "failures": wrong}


def compose_ttfa(stt, llm, tts, camera=None):
    """End-to-end TTFA composed from measured stage medians."""
    if not (stt and llm and tts):
        return None
    stage = {
        "stt_ms": stt["latency_ms"]["median"],
        "llm_first_token_ms": llm["ttft_ms"]["median"],
        "tts_first_audio_ms": tts["synth_ms"]["median"],
    }
    if camera:
        stage["camera_capture_ms"] = camera["capture_ms"]["median"]
    stage["ttfa_ms"] = round(sum(stage.values()), 1)
    return stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="v0.1")
    ap.add_argument("--no-vision", action="store_true")
    args = ap.parse_args()

    cfg = config_mod.load()
    outdir = ROOT / "bench" / "results" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)

    p("=" * 68)
    p(f"  BENCHMARK  {args.tag}   reps={args.reps}")
    p("=" * 68)

    p("\n[1/7] system")
    sysinfo = system_info(cfg)
    p(f"      {sysinfo['device']}  |  {sysinfo['cores']} cores  |  "
      f"{sysinfo['mem_available_mb']} MB free")

    p("\n[2/7] synthesising speech prompts with Piper")
    prompts = synth_prompts(cfg, VOICE_PROMPTS, Path("/tmp/bench_prompts"))
    p(f"      {len(prompts)} prompts, "
      f"{sum(x['duration_s'] for x in prompts):.1f}s of audio")

    p(f"\n[3/7] TTS ({cfg.get('tts.engine', 'piper')})")
    tts = bench_tts(cfg, args.reps)
    p(f"      synth median {tts['synth_ms']['median']:.0f} ms   "
      f"RTF {tts['realtime_factor']:.3f}x")

    p(f"\n[4/7] STT ({cfg.get('stt.engine', 'whisper')})")
    stt = bench_stt(cfg, prompts, args.reps)
    p(f"      median {stt['latency_ms']['median']:.0f} ms   "
      f"RTF {stt['realtime_factor']:.3f}x   "
      f"WER {stt['wer_mean']*100:.1f}%   exact {stt['exact_match_rate']*100:.0f}%")

    stt_spec = None
    if cfg.get("stt.speculative", True):
        p("\n[4b/7] STT with speculative decoding (what you actually wait for)")
        stt_spec = bench_stt_speculative(cfg, prompts, stt)
        if stt_spec:
            p(f"      wait after turn end median "
              f"{stt_spec['wait_after_turn_end_ms']['median']:.0f} ms   "
              f"(raw decode {stt_spec['raw_decode_median_ms']:.0f} ms, "
              f"saved {stt_spec['saved_ms']:.0f} ms)   "
              f"hits={stt_spec['speculation_hits']} "
              f"misses={stt_spec['speculation_misses']}   "
              f"WER {stt_spec['wer_mean']*100:.1f}%")

    # Read the model from config rather than hardcoding it. The label said "gemma3"
    # through an entire run of a different model, which is exactly the kind of thing
    # that makes an old result file impossible to trust later.
    _backend, _base, _model = resolve_backend(cfg)
    p(f"\n[5/7] LLM text-only ({_model} via {_backend})")
    llm_text = bench_llm(cfg, VOICE_PROMPTS[:6], args.reps, label="text")
    p(f"      TTFT median {llm_text['ttft_ms']['median']:.0f} ms   "
      f"{llm_text['tokens_per_s']} tok/s")

    camera = llm_vision = None
    jpeg_b64 = None
    frame_pool = []
    if not args.no_vision:
        p("\n[6/7] camera + LLM vision")
        camera, jpeg_b64, frame_pool = bench_camera(cfg, max(6, args.reps * 2))
        if camera:
            p(f"      capture median {camera['capture_ms']['median']:.0f} ms   "
              f"age {camera['frame_age_ms']['median']:.0f} ms   "
              f"{camera['encoded_size']}  {camera['jpeg_kb']['median']:.0f} KB")
        if jpeg_b64:
            llm_vision = bench_llm(cfg, VISION_PROMPTS, 1,
                                   label="vision", images_pool=frame_pool)
            p(f"      VLM TTFT median {llm_vision['ttft_ms']['median']:.0f} ms   "
              f"{llm_vision['tokens_per_s']} tok/s")
            # NOTE: no "cached frame" variant is reported. Repeating an
            # identical (prompt, image) pair returns in ~300 ms, but that is an
            # exact-prefix cache hit, not vision latency — with the same image and
            # five DIFFERENT questions the cost is 4900-5250 ms every time. See
            # camera.reuse_threshold in config.yaml.
    else:
        p("\n[6/7] vision skipped (--no-vision)")

    p("\n[7/7] vision trigger classifier")
    trig = bench_trigger(cfg)
    p(f"      {trig['correct']}/{trig['cases']} correct "
      f"({trig['accuracy']*100:.0f}%)   sticky={trig['sticky_followup_works']}")

    stt_for_ttfa = stt
    if stt_spec:
        # Substitute the wait a person actually experiences for the raw decode time.
        stt_for_ttfa = dict(stt)
        stt_for_ttfa["latency_ms"] = stt_spec["wait_after_turn_end_ms"]
    voice_ttfa = compose_ttfa(stt_for_ttfa, llm_text, tts)
    vision_ttfa = (compose_ttfa(stt_for_ttfa, llm_vision, tts, camera)
                   if llm_vision else None)
    vision_cached_ttfa = None

    results = {
        "tag": args.tag,
        "reps": args.reps,
        "system": sysinfo,
        "config": {
            "stt": cfg.get("stt"), "llm": {k: v for k, v in (cfg.get("llm") or {}).items()
                                           if k != "system_prompt"},
            "tts": cfg.get("tts"), "vad": cfg.get("vad"),
            "camera": cfg.get("camera"),
            "audio": {k: v for k, v in (cfg.get("audio") or {}).items()},
        },
        "stt": stt, "stt_speculative": stt_spec, "tts": tts, "llm_text": llm_text,
        "llm_vision": llm_vision,
        "camera": camera, "vision_trigger": trig,
        "composed_ttfa": {
            "voice_to_voice": voice_ttfa,
            "voice_image_to_voice": vision_ttfa,
        },
    }

    out = outdir / "benchmark.json"
    out.write_text(json.dumps(results, indent=2))

    p("\n" + "=" * 68)
    p("  COMPOSED TTFA (time to first audio, from measured stage medians)")
    p("=" * 68)
    if voice_ttfa:
        p(f"  voice -> voice          {voice_ttfa['ttfa_ms']:7.0f} ms")
        p(f"     stt {voice_ttfa['stt_ms']:.0f} + llm {voice_ttfa['llm_first_token_ms']:.0f}"
          f" + tts {voice_ttfa['tts_first_audio_ms']:.0f}")
    if vision_ttfa:
        p(f"  voice+image -> voice    {vision_ttfa['ttfa_ms']:7.0f} ms")
        p(f"     stt {vision_ttfa['stt_ms']:.0f} + cam "
          f"{vision_ttfa['camera_capture_ms']:.0f} + llm "
          f"{vision_ttfa['llm_first_token_ms']:.0f} + tts "
          f"{vision_ttfa['tts_first_audio_ms']:.0f}")
    p(f"\n  written: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
