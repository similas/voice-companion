#!/usr/bin/env bash
# Start / stop the llama.cpp server that backs the companion.
#
#   tools/llama_server.sh start
#   tools/llama_server.sh stop
#   tools/llama_server.sh status
#   tools/llama_server.sh restart
#
# WHY THIS EXISTS
# ---------------
# Ollama is a wrapper around llama.cpp and charges ~1.1-1.6 s of per-request
# overhead on this device — measured, and independent of both model size and
# prompt length:
#
#     ollama            TTFT 1872 ms   15.1 tok/s
#     llama-server      TTFT  196 ms   17.4 tok/s     <- same GGUF, same weights
#
# So we run the SAME binary Ollama ships, directly, as a long-lived process with a
# persistent slot. Nothing about the model changes; only the request path.
#
# THREE THINGS THAT WILL SILENTLY COST YOU THE GPU
# ------------------------------------------------
# 1. The Jetson CUDA backend is cuda_jetpack6, NOT cuda_v12. The generic CUDA 12
#    build does not work on Tegra.
# 2. GGML_BACKEND_PATH must be the .so FILE. Given a directory it fails with
#    "cannot read file data: Is a directory".
# 3. Failure is a WARNING, not an error: "no usable GPU found, --gpu-layers option
#    will be ignored" — and it then serves happily on CPU at half the speed. This
#    script greps for that line and refuses to pretend it succeeded.
#
# Vision: gemma3's projector lives inside the same GGUF, so --mmproj points at the
# same file. llama.cpp logs "detected Ollama-format gemma3 GGUF used as mmproj;
# translating" and it works.

set -uo pipefail
cd "$(dirname "$0")/.."

OLLAMA_LIB=/usr/local/lib/ollama
BIN="$OLLAMA_LIB/llama-server"
BACKEND="$OLLAMA_LIB/cuda_jetpack6/libggml-cuda.so"
LOG="logs/llama-server.log"
PORT="${LLAMA_PORT:-8081}"
CTX="${LLAMA_CTX:-2048}"
THREADS="${LLAMA_THREADS:-4}"

# Resolve the model blob from Ollama's store so we reuse whatever `ollama pull`
# already downloaded rather than keeping a second copy of a 3.3 GB file.
MODEL_TAG="${LLAMA_MODEL_TAG:-gemma3:4b}"
find_blob() {
  local tag="${1%%:*}" ver="${1##*:}"
  local man="/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library/$tag/$ver"
  [[ -r "$man" ]] || return 1
  python3 - "$man" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for l in d["layers"]:
    if l["mediaType"].endswith(".model"):
        print("/usr/share/ollama/.ollama/models/blobs/"+l["digest"].replace(":","-"))
        break
PY
}

pids() { pgrep -x llama-server 2>/dev/null; }

do_stop() {
  local found=0
  for p in $(pids); do found=1; kill "$p" 2>/dev/null; done
  [[ $found -eq 1 ]] || { echo "not running"; return 0; }
  for _ in $(seq 1 10); do
    [[ -z "$(pids)" ]] && break
    sleep 1
  done
  for p in $(pids); do kill -9 "$p" 2>/dev/null; done
  echo "stopped"
}

do_status() {
  if [[ -z "$(pids)" ]]; then
    echo "llama-server: not running"
    return 1
  fi
  local health
  health=$(curl -s --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)
  echo "llama-server: pid $(pids | tr '\n' ' ')| port $PORT | health ${health:-unreachable}"
  if grep -q "no usable GPU found" "$LOG" 2>/dev/null; then
    echo "  WARNING: running on CPU — the CUDA backend did not load"
    return 1
  fi
  echo "  GPU backend: loaded"
}

do_start() {
  if [[ -n "$(pids)" ]]; then echo "already running"; do_status; return 0; fi
  [[ -x "$BIN" ]] || { echo "missing $BIN (is ollama installed?)" >&2; exit 1; }
  [[ -r "$BACKEND" ]] || { echo "missing CUDA backend $BACKEND" >&2; exit 1; }

  local blob
  blob="${LLAMA_MODEL_BLOB:-$(find_blob "$MODEL_TAG")}"
  if [[ -z "${blob:-}" || ! -r "$blob" ]]; then
    echo "could not resolve a model blob for $MODEL_TAG" >&2
    echo "  try: ollama pull $MODEL_TAG   (or set LLAMA_MODEL_BLOB)" >&2
    exit 1
  fi

  # Free the GPU first: Ollama holding the same model would double the ~3 GB and
  # push one of the two onto the CPU.
  ollama stop "$MODEL_TAG" >/dev/null 2>&1
  ollama stop gemma3:4b-jetson >/dev/null 2>&1
  sleep 2

  mkdir -p logs
  : > "$LOG"
  echo "starting llama-server: $(basename "$blob") ctx=$CTX threads=$THREADS port=$PORT"
  GGML_BACKEND_PATH="$BACKEND" \
  LD_LIBRARY_PATH="$OLLAMA_LIB/cuda_jetpack6:$OLLAMA_LIB" \
  nohup "$BIN" \
    --model "$blob" \
    --mmproj "$blob" \
    --host 127.0.0.1 --port "$PORT" \
    --ctx-size "$CTX" --n-gpu-layers 99 --threads "$THREADS" \
    --cont-batching \
    >> "$LOG" 2>&1 &
  disown

  for _ in $(seq 1 60); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ok && break
    [[ -z "$(pids)" ]] && { echo "died on startup; last lines:"; tail -12 "$LOG"; exit 1; }
    sleep 2
  done
  do_status
}

case "${1:-start}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; sleep 2; do_start ;;
  status)  do_status ;;
  *) echo "usage: $0 {start|stop|restart|status}" >&2; exit 1 ;;
esac
