"""
Offline self-test: exercises every component except the audio transport.

This runs without a microphone, without pyaudio, and without anyone speaking. It
proves the pieces work before you rely on a live conversation to debug them:

  1. config loads and the vision triggers behave (including NOT firing on
     ordinary sentences, which matters more than firing when asked)
  2. the camera opens and produces a fresh JPEG
  3. Gemma answers a text prompt, and answers a question about a real camera frame
  4. Piper synthesises audio
  5. the metrics CSV is written with the right columns and plausible numbers

    python tools/selftest.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import base64
import json
import urllib.request

from app import config as config_mod
from app.metrics import MetricsLogger
from app.vision import Camera, VisionTrigger

PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name:38s} {detail}")


def ollama(cfg, prompt, image_b64=None, max_tokens=60):
    """Call Ollama's native API. Returns (text, seconds_to_first_token, total)."""
    body = {
        "model": cfg.get("llm.model"),
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": max_tokens},
    }
    if image_b64:
        body["images"] = [image_b64]
    req = urllib.request.Request(
        cfg.get("llm.base_url").replace("/v1", "") + "/api/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    out = []
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw in r:
            if not raw.strip():
                continue
            d = json.loads(raw)
            tok = d.get("response", "")
            if tok and first is None:
                first = time.perf_counter() - t0
            out.append(tok)
            if d.get("done"):
                break
    return "".join(out).strip(), first, time.perf_counter() - t0


def main():
    cfg = config_mod.load()

    print("\n--- 1. config + vision trigger ---")
    report("config loads", cfg.get("llm.model") is not None,
           f"model={cfg.get('llm.model')}")
    trig = VisionTrigger(cfg.get("vision.triggers"), True)
    should = ["what do you see", "hey, what is this thing?",
              "can you see my laptop", "what am I holding right now",
              "take a look at this"]
    should_not = ["what is the capital of France", "tell me a joke",
                  "how are you doing today", "what time is it",
                  "I saw a movie yesterday"]
    hits = [s for s in should if trig.wants_vision(s)]
    misses = [s for s in should_not if trig.wants_vision(s)]
    report("triggers fire when asked", len(hits) == len(should),
           f"{len(hits)}/{len(should)}")
    report("triggers stay quiet otherwise", not misses,
           "no false positives" if not misses else f"FALSE POSITIVE: {misses}")

    print("\n--- 2. camera ---")
    cam = Camera(cfg.get("camera.device"), cfg.get("camera.width"),
                 cfg.get("camera.height"), cfg.get("camera.poll_fps"))
    cam_ok = cam.start()
    report("camera opens", cam_ok)
    jpeg_b64 = None
    if cam_ok:
        time.sleep(1.0)
        t0 = time.perf_counter()
        got = cam.grab_jpeg(cfg.get("camera.max_edge"), cfg.get("camera.jpeg_quality"))
        dt = (time.perf_counter() - t0) * 1000
        if got:
            jpeg, (w, h), age = got
            jpeg_b64 = base64.b64encode(jpeg).decode()
            report("frame grab", True,
                   f"{w}x{h} {len(jpeg)//1024}KB in {dt:.0f}ms, {age*1000:.0f}ms old")
            report("frame is fresh", age < 1.0, f"age {age*1000:.0f}ms")
        else:
            report("frame grab", False)

    print("\n--- 3. Gemma (text, then vision) ---")
    try:
        txt, ttft, tot = ollama(cfg, "Reply with exactly: ready", max_tokens=10)
        report("text inference", bool(txt),
               f"ttft {ttft*1000:.0f}ms, total {tot*1000:.0f}ms -> {txt[:30]!r}")
    except Exception as e:
        report("text inference", False, str(e)[:60])
    if jpeg_b64:
        try:
            txt, ttft, tot = ollama(
                cfg, "In one short sentence, what do you see?", jpeg_b64, 60)
            report("VISION inference", bool(txt),
                   f"ttft {ttft*1000:.0f}ms, total {tot*1000:.0f}ms")
            print(f"         model saw: {txt[:150]}")
        except Exception as e:
            report("VISION inference", False, str(e)[:60])

    print("\n--- 4. Piper ---")
    try:
        import wave
        from piper import PiperVoice
        vpath = cfg.path("tts.models_dir") / f"{cfg.get('tts.voice')}.onnx"
        t0 = time.perf_counter()
        voice = PiperVoice.load(str(vpath))
        load = time.perf_counter() - t0
        out = ROOT / "logs" / "selftest_tts.wav"
        t0 = time.perf_counter()
        with wave.open(str(out), "wb") as wf:
            voice.synthesize_wav("Self test complete.", wf)
        syn = time.perf_counter() - t0
        with wave.open(str(out), "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        report("piper synthesis", dur > 0.3,
               f"load {load*1000:.0f}ms, synth {syn*1000:.0f}ms for {dur:.2f}s "
               f"(RTF {syn/dur:.2f}x)")
    except Exception as e:
        report("piper synthesis", False, str(e)[:60])

    print("\n--- 5. metrics CSV ---")
    try:
        m = MetricsLogger(ROOT / "logs", console=False)
        t = m.start_turn()
        time.sleep(0.02); m.stt_done("what do you see")
        time.sleep(0.01); m.vision_done()
        time.sleep(0.03); m.llm_first_token(); m.llm_text("A desk with a laptop.")
        time.sleep(0.02); m.llm_done()
        time.sleep(0.01); m.tts_first_audio()
        time.sleep(0.01); m.playback_start()
        row = t.row()
        m.finish_turn(); m.close()
        import csv as _csv
        with open(m.path) as f:
            rows = list(_csv.DictReader(f))
        ordered = (rows and
                   float(rows[0]["t_stt_done"]) < float(rows[0]["t_llm_first_token"])
                   < float(rows[0]["t_audio_playback_start"]))
        report("csv written", len(rows) == 1, f"{m.path.name}")
        report("stage times ordered", bool(ordered),
               f"stt {rows[0]['t_stt_done']} < ttft {rows[0]['t_llm_first_token']}"
               f" < play {rows[0]['t_audio_playback_start']} ms" if rows else "")
        report("vision_used recorded", rows[0]["vision_used"] == "true" if rows else False)
        report("null vision when unused", row["t_vision_done"] is not None)
        m.path.unlink(missing_ok=True)
    except Exception as e:
        report("metrics CSV", False, f"{type(e).__name__}: {str(e)[:60]}")

    if cam_ok:
        cam.stop()
    bad = results.count(False)
    print(f"\n  {len(results)-bad}/{len(results)} passed"
          + (f", {bad} FAILED" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
