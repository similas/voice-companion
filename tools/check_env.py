"""
Environment doctor. Run this first, and again whenever hardware changes.

    python tools/check_env.py               # full check
    python tools/check_env.py --list-audio  # PyAudio device indices for config.yaml

The --list-audio output is what you need when swapping in the reSpeaker XVF3800:
PyAudio device indices are NOT ALSA card numbers, so you cannot guess them.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Re-exec under the project venv if we were started with the wrong interpreter.
# Running `python tools/check_env.py` with the system python reports every
# dependency as missing, which is confusing and wrong — the packages are in the
# venv. Rather than document that, just fix it.
_VENV_DIR = Path.home() / ".venvs" / "voice-companion"
_VENV_PY = _VENV_DIR / "bin" / "python"
# Compare sys.prefix, NOT the resolved interpreter path: virtualenv makes
# bin/python a symlink to /usr/bin/python3, so resolve() collapses both to the
# same file and an executable-path comparison always looks "already correct".
if _VENV_PY.exists() and Path(sys.prefix) != _VENV_DIR:
    os.execv(str(_VENV_PY),
             [str(_VENV_PY), str(Path(__file__).resolve())] + sys.argv[1:])

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "


def line(status, label, detail=""):
    print(f"[{status}] {label:26s} {detail}")


def check_python():
    v = sys.version_info
    good = v >= (3, 10)
    line(OK if good else BAD, "python", f"{v.major}.{v.minor}.{v.micro}")
    return good


def check_imports():
    ok = True
    mods = ["pipecat", "faster_whisper", "ctranslate2", "onnxruntime", "piper",
            "cv2", "numpy", "yaml"]
    for m in mods:
        try:
            mod = __import__(m)
            line(OK, m, getattr(mod, "__version__", ""))
        except Exception as e:
            line(BAD, m, f"{type(e).__name__}: {str(e)[:50]}")
            ok = False
    # pyaudio is checked separately because its fix is an apt package
    try:
        import pyaudio  # noqa: F401
        line(OK, "pyaudio", "")
    except Exception:
        line(BAD, "pyaudio", "missing -> sudo apt install -y portaudio19-dev "
                             "&& pip install pyaudio")
        ok = False
    return ok


def check_ollama(model):
    exe = shutil.which("ollama")
    if not exe:
        line(BAD, "ollama", "not installed")
        return False
    try:
        r = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=10)
        tags = json.load(r)
    except Exception as e:
        line(BAD, "ollama server", f"not responding: {str(e)[:40]}")
        return False
    names = [m["name"] for m in tags.get("models", [])]
    line(OK, "ollama server", f"{len(names)} model(s)")
    if any(n == model or n.startswith(model.split(":")[0] + ":") for n in names):
        line(OK, f"model {model}", "present")
        return True
    line(BAD, f"model {model}", f"NOT pulled — run: ollama pull {model}")
    return False


def check_llama_server(cfg):
    """Health of the llama.cpp server, when that is the configured backend."""
    port = cfg.get("llm.llama_server.port", 8081)
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                    timeout=5) as r:
            up = b"ok" in r.read()
    except Exception:
        up = False
    if not up:
        line(BAD, "llama-server", f"not responding on :{port} — "
                                  f"run tools/llama_server.sh start")
        return False
    log = ROOT / "logs" / "llama-server.log"
    if log.exists() and "no usable GPU found" in log.read_text(errors="ignore"):
        line(BAD, "llama-server", "running WITHOUT the GPU (~2x slower) — "
                                  "tools/llama_server.sh restart")
        return False
    line(OK, "llama-server", f":{port} healthy, GPU backend loaded")
    return True


def check_gpu_offload(model):
    """The single most important performance check on this board.

    Jetson shares one pool of memory between CPU and GPU. If too little is free
    when Ollama loads the model, it silently offloads ZERO layers to the GPU and
    runs on CPU instead — same answers, several times slower. `ollama ps` reports
    the split, so we read it rather than assume.
    """
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                             timeout=15).stdout
    except Exception:
        line(WARN, "gpu offload", "could not run `ollama ps`")
        return
    rows = [r for r in out.splitlines()[1:] if r.strip()]
    if not rows:
        line(WARN, "gpu offload", "model not loaded — ask it something first, "
                                  "then re-run this check")
        return
    for r in rows:
        if "%" in r and "/" in r:
            frag = [p for p in r.split() if "%" in p]
            detail = " ".join(frag)
            if "100% GPU" in r:
                line(OK, "gpu offload", f"{detail} — fully on GPU")
            elif "CPU" in detail:
                line(BAD, "gpu offload", f"{detail} — RUNNING ON CPU, "
                                         "free more RAM (see README)")
            else:
                line(WARN, "gpu offload", detail)
            return
    line(WARN, "gpu offload", rows[0][:60])


def check_memory(backend="ollama", server_up=False):
    info = {}
    for ln in open("/proc/meminfo"):
        k, _, v = ln.partition(":")
        info[k] = int(v.split()[0]) // 1024        # MB
    avail, total = info["MemAvailable"], info["MemTotal"]
    # Gemma 3 4B needs ~2.4 GB of weights plus ~1 GB of context/compute buffers
    # to land entirely on the GPU. Below that it degrades to CPU.
    need = 3600
    if backend == "llama_server" and server_up:
        # The model is ALREADY resident inside llama-server, so low free memory is
        # the expected steady state rather than a problem. Flagging it as a failure
        # here was simply wrong.
        line(OK, "memory available", f"{avail} MB of {total} MB "
                                     f"(model already resident in llama-server)")
        return True
    status = OK if avail >= need else (WARN if avail >= 2500 else BAD)
    line(status, "memory available", f"{avail} MB of {total} MB "
                                     f"(need ~{need} MB free for GPU offload)")
    if avail < need:
        try:
            out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                                 capture_output=True, text=True, timeout=10).stdout
            for name in out.split():
                line(WARN, "  docker running", f"{name} — `docker stop {name}` to free RAM")
        except Exception:
            pass
    return avail >= 2500


def check_camera(index, width=1280, height=720):
    devs = sorted(glob.glob("/dev/video*"))
    if not devs:
        line(BAD, "camera", "no /dev/video* devices")
        return False
    line(OK, "video devices", " ".join(devs))
    try:
        import cv2
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            line(BAD, f"camera {index}", "cannot open (in use by another process?)")
            return False
        # Request the same format the app uses, so the reported resolution is
        # what you'll actually get rather than the driver default.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            line(BAD, f"camera {index}", "opened but read failed")
            return False
        h, w = frame.shape[:2]
        line(OK, f"camera {index}", f"{w}x{h} frame captured")
        return True
    except Exception as e:
        line(BAD, f"camera {index}", str(e)[:50])
        return False


def list_audio():
    try:
        import pyaudio
    except Exception:
        print("pyaudio not installed — run: sudo apt install -y portaudio19-dev "
              "&& pip install pyaudio")
        return
    # PortAudio probes every ALSA plugin on init and prints a wall of harmless
    # "Unknown PCM cards.pcm.rear" noise to stderr. Silence it so the device
    # table is actually readable.
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    try:
        pa = pyaudio.PyAudio()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    print("\nPyAudio devices — use these INDEX values in config.yaml")
    print("(these are NOT ALSA card numbers)\n")
    print(f"  {'idx':>3}  {'in':>2} {'out':>3}  {'rate':>6}  name")
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        print(f"  {i:>3}  {int(d['maxInputChannels']):>2} "
              f"{int(d['maxOutputChannels']):>3}  {int(d['defaultSampleRate']):>6}  "
              f"{d['name'][:52]}")
    try:
        print(f"\n  default input : {pa.get_default_input_device_info()['index']}")
        print(f"  default output: {pa.get_default_output_device_info()['index']}")
    except Exception:
        pass
    pa.terminate()


def check_audio_devices():
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True,
                             timeout=10).stdout
        cards = [l for l in out.splitlines() if l.startswith("card")]
        usb = [c for c in cards if "USB" in c or "C960" in c or "ReSpeaker" in c
               or "XVF" in c]
        line(OK if cards else BAD, "capture devices",
             f"{len(cards)} card(s)" + (f", mic: {usb[0][:44]}" if usb else ""))
    except Exception as e:
        line(BAD, "capture devices", str(e)[:40])
    try:
        out = subprocess.run(["pactl", "list", "short", "sinks"],
                             capture_output=True, text=True, timeout=10).stdout
        sinks = [l.split("\t")[1] for l in out.splitlines() if "\t" in l]
        bt = [s for s in sinks if "bluez" in s.lower()]
        line(OK, "output sinks", ", ".join(s[:34] for s in sinks[:3]) or "none")
        line(OK if bt else WARN, "bluetooth speaker",
             bt[0][:44] if bt else "not connected — pair it, then re-run")
    except Exception:
        line(WARN, "output sinks", "pactl unavailable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-audio", action="store_true",
                    help="print PyAudio device indices and exit")
    args = ap.parse_args()

    if args.list_audio:
        list_audio()
        return 0

    from app import config as config_mod
    cfg = config_mod.load()

    print("\n=== python + packages ===")
    ok = check_python()
    ok &= check_imports()
    backend = (cfg.get("llm.backend") or "ollama").strip()
    print(f"\n=== models (backend: {backend}) ===")
    server_up = False
    if backend == "llama_server":
        server_up = check_llama_server(cfg)
        ok &= server_up
        # Ollama is still needed as the model STORE — llama_server reads the GGUF
        # blob that `ollama pull` downloaded.
        ok &= check_ollama(cfg.get("llm.llama_server.model_tag", "gemma3:4b"))
    else:
        ok &= check_ollama(cfg.get("llm.model", "gemma3:4b"))
    voice = cfg.path("tts.models_dir", "models") / f"{cfg.get('tts.voice')}.onnx"
    if voice.exists():
        line(OK, "piper voice", f"{voice.name} ({voice.stat().st_size // 1048576} MB)")
    else:
        line(BAD, "piper voice", f"missing: {voice}")
        ok = False
    print("\n=== hardware ===")
    check_memory(backend, server_up)
    if backend != "llama_server":
        check_gpu_offload(cfg.get("llm.model", "gemma3:4b"))
    check_camera(cfg.get("camera.device", 0),
                 cfg.get("camera.width", 1280), cfg.get("camera.height", 720))
    check_audio_devices()
    print()
    print("all good — run ./run.sh" if ok else
          "fix the FAIL lines above, then re-run")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
