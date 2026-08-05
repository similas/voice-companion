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
            "stt": f"faster-whisper {cfg.get('stt.model')} "
                   f"({cfg.get('stt.compute_type')}, {cfg.get('stt.device')})",
            "llm": cfg.get("llm.model"),
            "tts": f"piper {cfg.get('tts.voice')}",
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
    from piper import PiperVoice
    voice_path = cfg.path("tts.models_dir") / f"{cfg.get('tts.voice')}.onnx"
    t0 = time.perf_counter()
    voice = PiperVoice.load(str(voice_path))
    load_ms = (time.perf_counter() - t0) * 1000

    # A typical spoken reply from this assistant: one to two short sentences.
    sample = "That's four. Is there anything else you would like to know?"
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
            "sample_text": sample}


def bench_stt(cfg, prompts, reps):
    from faster_whisper import WhisperModel
    t0 = time.perf_counter()
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


def ollama_stream(base_url, model, messages, max_tokens, images=None):
    """One streamed chat completion. Returns (ttft_ms, total_ms, text, n_tok)."""
    url = base_url.rstrip("/").removesuffix("/v1") + "/api/chat"
    msgs = [dict(m) for m in messages]
    if images:
        msgs[-1]["images"] = images
    body = json.dumps({"model": model, "messages": msgs, "stream": True,
                       "keep_alive": "60m",
                       "options": {"num_predict": max_tokens}}).encode()
    req = urllib.request.Request(url, data=body,
                                headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    n = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            if not raw.strip():
                continue
            d = json.loads(raw)
            tok = (d.get("message") or {}).get("content", "")
            if tok and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000
            if tok:
                chunks.append(tok)
            if d.get("done"):
                n = d.get("eval_count", 0)
                break
    return ttft, (time.perf_counter() - t0) * 1000, "".join(chunks).strip(), n


def bench_llm(cfg, prompts, reps, image_b64=None, label="text"):
    base = cfg.get("llm.base_url")
    model = cfg.get("llm.model")
    sysmsg = {"role": "system", "content": cfg.get("llm.system_prompt", "").strip()}
    max_tok = cfg.get("llm.max_tokens", 60)

    # warm-up so the first measured call is not paying a model load
    ollama_stream(base, model, [sysmsg, {"role": "user", "content": "hi"}], 4,
                  images=[image_b64] if image_b64 else None)

    ttfts, totals, rates, outs = [], [], [], []
    for text in prompts:
        for _ in range(reps):
            msgs = [sysmsg, {"role": "user", "content": text}]
            ttft, total, reply, n = ollama_stream(
                base, model, msgs, max_tok,
                images=[image_b64] if image_b64 else None)
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


def bench_camera(cfg, reps):
    from app.vision import Camera
    cam = Camera(cfg.get("camera.device"), cfg.get("camera.width"),
                 cfg.get("camera.height"), cfg.get("camera.poll_fps"))
    if not cam.start():
        return None, None
    time.sleep(1.5)
    grabs, ages, sizes = [], [], []
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
        dims = f"{w}x{h}"
        time.sleep(0.3)
    cam.stop()
    return {
        "capture_ms": stats(grabs),
        "frame_age_ms": stats(ages),
        "jpeg_kb": stats(sizes),
        "encoded_size": dims,
        "capture_resolution": f"{cfg.get('camera.width')}x{cfg.get('camera.height')}",
    }, jpeg_b64


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

    p("\n[3/7] TTS (piper)")
    tts = bench_tts(cfg, args.reps)
    p(f"      synth median {tts['synth_ms']['median']:.0f} ms   "
      f"RTF {tts['realtime_factor']:.3f}x")

    p("\n[4/7] STT (faster-whisper)")
    stt = bench_stt(cfg, prompts, args.reps)
    p(f"      median {stt['latency_ms']['median']:.0f} ms   "
      f"RTF {stt['realtime_factor']:.3f}x   "
      f"WER {stt['wer_mean']*100:.1f}%   exact {stt['exact_match_rate']*100:.0f}%")

    p("\n[5/7] LLM text-only (gemma3 via ollama)")
    llm_text = bench_llm(cfg, VOICE_PROMPTS[:6], args.reps, label="text")
    p(f"      TTFT median {llm_text['ttft_ms']['median']:.0f} ms   "
      f"{llm_text['tokens_per_s']} tok/s")

    camera = llm_vision = None
    jpeg_b64 = None
    if not args.no_vision:
        p("\n[6/7] camera + LLM vision")
        camera, jpeg_b64 = bench_camera(cfg, max(3, args.reps))
        if camera:
            p(f"      capture median {camera['capture_ms']['median']:.0f} ms   "
              f"age {camera['frame_age_ms']['median']:.0f} ms   "
              f"{camera['encoded_size']}  {camera['jpeg_kb']['median']:.0f} KB")
        if jpeg_b64:
            llm_vision = bench_llm(cfg, VISION_PROMPTS, max(2, args.reps - 1),
                                   image_b64=jpeg_b64, label="vision")
            p(f"      VLM TTFT median {llm_vision['ttft_ms']['median']:.0f} ms   "
              f"{llm_vision['tokens_per_s']} tok/s")
    else:
        p("\n[6/7] vision skipped (--no-vision)")

    p("\n[7/7] vision trigger classifier")
    trig = bench_trigger(cfg)
    p(f"      {trig['correct']}/{trig['cases']} correct "
      f"({trig['accuracy']*100:.0f}%)   sticky={trig['sticky_followup_works']}")

    voice_ttfa = compose_ttfa(stt, llm_text, tts)
    vision_ttfa = compose_ttfa(stt, llm_vision, tts, camera) if llm_vision else None

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
        "stt": stt, "tts": tts, "llm_text": llm_text,
        "llm_vision": llm_vision, "camera": camera, "vision_trigger": trig,
        "composed_ttfa": {"voice_to_voice": voice_ttfa,
                          "voice_image_to_voice": vision_ttfa},
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
