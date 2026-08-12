#!/usr/bin/env bash
# Start / stop the Kokoro TTS sidecar (GPU voice for the companion).
#
#   tools/kokoro_server.sh start|stop|restart|status
#
# Runs tools/kokoro_server.py under the ~/.venvs/tts-lab venv as a transient
# systemd unit inside voice.slice — same containment as llama-server, its own
# memory ceiling, and an OOM kill order between llama-server (1000, dies
# first) and the agent (900, dies last).
#
# THE VENV IS SEPARATE ON PURPOSE. The agent's venv pins onnxruntime 1.23.2
# (CPU) for pipecat + Silero VAD; Kokoro-on-GPU needs the Jetson wheel of
# onnxruntime-gpu 1.24, and the two cannot share a venv. Recreate with:
#
#   python3 -m virtualenv ~/.venvs/tts-lab
#   ~/.venvs/tts-lab/bin/pip install kokoro-onnx
#   ~/.venvs/tts-lab/bin/pip uninstall -y onnxruntime
#   ~/.venvs/tts-lab/bin/pip install \
#       --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 onnxruntime-gpu
#
# (kokoro-onnx pulls the CPU onnxruntime as a dependency; it must be removed
# before installing the GPU build or the two collide in one package dir and
# imports break. The index is pypi.jetson-ai-lab.IO — the .dev mirror is dead.)
#
# Model files (not in the repo — 338 MB):
#   curl -L -o models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
#   curl -L -o models/voices-v1.0.bin  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

set -uo pipefail
cd "$(dirname "$0")/.."

VENV="$HOME/.venvs/tts-lab"
PORT="${KOKORO_PORT:-8092}"
UNIT="${KOKORO_UNIT:-kokoro-tts}"
LOG="logs/kokoro.log"
MODEL="models/kokoro-v1.0.onnx"
VOICES="models/voices-v1.0.bin"

# Measured RSS is ~850 MB (CUDA context + weights + ORT arenas). 1200M is
# headroom without letting a leak eat the agent's share of voice.slice.
MEM_MAX_MB="${KOKORO_MEM_MAX_MB:-1200}"

health() { curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null; }

do_status() {
  local h; h=$(health)
  if [[ "$h" != *ok* ]]; then
    echo "kokoro-tts: not serving on $PORT"
    return 1
  fi
  echo "kokoro-tts: port $PORT | $h"
  [[ "$h" == *'"gpu": true'* || "$h" == *'"gpu":true'* ]] \
    || { echo "  WARNING: running on CPU — ~5x slower (RTF 0.7)"; return 1; }
}

do_stop() {
  systemctl --user stop "$UNIT" 2>/dev/null
  systemctl --user reset-failed "$UNIT" 2>/dev/null
  echo "stopped"
}

do_start() {
  if [[ "$(health)" == *ok* ]]; then echo "already running"; do_status; return 0; fi
  [[ -x "$VENV/bin/python" ]] || { echo "missing venv $VENV — see header" >&2; exit 1; }
  [[ -r "$MODEL" && -r "$VOICES" ]] || { echo "missing model files — see header" >&2; exit 1; }

  mkdir -p logs
  systemctl --user reset-failed "$UNIT" 2>/dev/null
  systemctl --user stop "$UNIT" 2>/dev/null
  # CPUAffinity 0-2: its CPU work (phonemization) is tiny and bursts alongside
  # llama-server's cores, keeping 3-5 clear for whisper during turn overlap.
  systemd-run --user --quiet \
    --unit="$UNIT" \
    --slice=voice.slice \
    --property=MemoryMax="${MEM_MAX_MB}M" \
    --property=MemorySwapMax=0 \
    --property=OOMScoreAdjust=950 \
    --property=CPUAffinity=0-2 \
    --property=LimitCORE=0 \
    --property=Restart=on-failure \
    --property=RestartSec=3 \
    --property=StandardOutput="append:$PWD/$LOG" \
    --property=StandardError="append:$PWD/$LOG" \
    "$VENV/bin/python" "$PWD/tools/kokoro_server.py" \
      --port "$PORT" --model "$PWD/$MODEL" --voices "$PWD/$VOICES" \
    || { echo "systemd-run failed" >&2; exit 1; }

  # Load + CUDA warm-up takes ~15 s the first time.
  for _ in $(seq 1 30); do
    [[ "$(health)" == *ok* ]] && break
    systemctl --user is-active --quiet "$UNIT" \
      || { echo "died on startup; last lines:"; tail -8 "$LOG"; exit 1; }
    sleep 2
  done
  do_status
}

case "${1:-start}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; sleep 1; do_start ;;
  status)  do_status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 1 ;;
esac
