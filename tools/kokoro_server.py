#!/usr/bin/env python3
"""
Kokoro TTS sidecar — GPU synthesis behind a local HTTP socket.

WHY A SIDECAR, NOT IN-PROCESS
-----------------------------
The agent's venv pins onnxruntime 1.23.2 (CPU) — pipecat requires it and
Silero VAD runs on it. Kokoro on the GPU needs the Jetson-specific
onnxruntime-gpu 1.24 wheel, and the two packages cannot coexist in one venv
(same import name; a mixed install fails with ImportError, verified). Worse,
with the GPU build installed, Silero's default provider list would put VAD on
the GPU — a memcpy round-trip per 32 ms audio frame. So Kokoro lives in its
own venv (~/.venvs/tts-lab) and its own process, and the agent speaks to it
over localhost. Isolation also means its ~850 MB RSS is capped by its own
systemd unit, and a crash restarts the voice, not the agent.

Runs under ~/.venvs/tts-lab (see tools/kokoro_server.sh for the venv recipe).

API
---
GET  /health          -> {"status":"ok","gpu":true}
POST /tts             {"text": "...", "voice": "af_heart", "speed": 1.0}
                      -> raw s16le mono PCM, X-Sample-Rate header (24000)

Measured on this device (warm, cores 0-2, CUDAExecutionProvider):
    30-char clause   357 ms   RTF 0.15
    64-char sentence 531 ms   RTF 0.11
The first synth after load pays ~1.1 s of CUDA graph warm-up; the startup
warm-up call below absorbs it so the first real clause never does.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--model", required=True)
    ap.add_argument("--voices", required=True)
    ap.add_argument("--default-voice", default="af_heart")
    args = ap.parse_args()

    import onnxruntime as ort
    from kokoro_onnx import Kokoro

    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    kokoro = Kokoro.from_session(sess, args.voices)
    gpu = sess.get_providers()[0] == "CUDAExecutionProvider"
    kokoro.create("Warming up the graph.", voice=args.default_voice)
    print(f"kokoro: ready in {time.perf_counter() - t0:.1f}s "
          f"({'GPU' if gpu else 'CPU — will be ~5x slower'})", flush=True)

    # One synth at a time. The pipeline is serial anyway (clauses of one
    # reply), and two concurrent graphs on this GPU would just slow both.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body=b"", ctype="text/plain", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, json.dumps({"status": "ok", "gpu": gpu}).encode(),
                           "application/json")
            else:
                self._send(404)

        def do_POST(self):
            if self.path != "/tts":
                self._send(404)
                return
            try:
                req = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                text = (req.get("text") or "").strip()
                if not text:
                    self._send(400, b"empty text")
                    return
                t0 = time.perf_counter()
                with lock:
                    samples, sr = kokoro.create(
                        text, voice=req.get("voice", args.default_voice),
                        speed=float(req.get("speed", 1.0)))
                pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
                self._send(200, pcm, "audio/L16", {"X-Sample-Rate": str(sr)})
                print(f"tts: {len(text)}ch -> {len(pcm) / 2 / sr:.2f}s audio "
                      f"in {(time.perf_counter() - t0) * 1000:.0f}ms", flush=True)
            except Exception as e:                      # a bad clause must not
                self._send(500, str(e).encode())        # take the server down
                print(f"tts: FAILED {e}", flush=True)

        def log_message(self, *_):
            pass                                        # keep the log readable

    print(f"kokoro: listening on 127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
