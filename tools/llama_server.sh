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
#
# --reasoning off IS ALSO LOAD-BEARING, for a different reason. gemma-4-E2B is a
# reasoning model: by default it emits a whole thinking block into
# `reasoning_content` before the first token of `content`. For a chatbot that is
# fine; for a voice loop it is fatal, because time-to-first-audio waits on the
# entire reasoning pass. Measured on "Say hello in five words": with reasoning on,
# 40 tokens of thinking and content still empty at the cap. Off: 130 ms to first
# token. --reasoning-budget 0 is belt-and-braces for templates that ignore the
# first flag.
#
# --parallel 1 IS LOAD-BEARING ON 8 GB. The default is 4 slots, and each slot gets
# its own KV cache of --ctx-size tokens, so ctx 2048 silently reserves 8192 tokens'
# worth. On unified memory that came out of the same pool the agent needs, and the
# kernel OOM-killed the agent mid-conversation:
#     Out of memory: Killed process 35421 (python)
# This is a single-user voice loop; one slot is all it can ever use. Ollama's own
# runner passes -np 1 for the same reason.

set -uo pipefail
cd "$(dirname "$0")/.."

OLLAMA_LIB=/usr/local/lib/ollama
BIN="$OLLAMA_LIB/llama-server"
BACKEND="$OLLAMA_LIB/cuda_jetpack6/libggml-cuda.so"
LOG="logs/llama-server.log"
PORT="${LLAMA_PORT:-8081}"
CTX="${LLAMA_CTX:-2048}"
# 3 threads on cores 0-2, not 4 floating. Six cores run llama-server, whisper and
# piper AT THE SAME TIME during every turn (the stages overlap by design), and
# letting each size itself for the whole machine oversubscribed it ~10 threads on 6
# cores. Measured cost of that fight: STT 242 ms alone -> 921 ms live, LLM TTFT
# 138 ms -> 850 ms. Partition instead: llama-server cores 0-2, agent cores 3-5.
THREADS="${LLAMA_THREADS:-3}"
UNIT="${LLAMA_UNIT:-llama-server}"

# MEMORY BUDGET. This machine has 7.4 GB shared between CPU and GPU, and running
# out does not degrade gracefully — the kernel OOM killer picks a victim, and it
# picked the agent mid-conversation and then the user's own session:
#     Out of memory: Killed process 35421 (python)
#
# RESERVE_MB is what must remain free AFTER this server has loaded, for the agent
# that starts next (~700 MB with whisper and piper resident). MemAvailable already
# excludes running processes, so this covers only NEW allocation.
#
# This is no longer a safety margin against OOM — voice.slice provides that. It is
# just "leave room for the agent", so 800 is right and the old 2000 was cargo.
#
# MEM_MAX_MB is a hard cgroup ceiling, so if this server ever does run away the
# kernel kills IT inside its own cgroup rather than picking the agent or your shell.
RESERVE_MB="${LLAMA_RESERVE_MB:-800}"

# 3200M, not 4200M. This unit now lives inside voice.slice, which caps the WHOLE
# stack at 4600M. If this one process could take 4200M of that, the agent would be
# squeezed into 400M and the slice would kill something on every turn. Measured
# steady state here is ~2450M, so 3200M is headroom without starving the agent.
MEM_MAX_MB="${LLAMA_MEM_MAX_MB:-3200}"

# Vision costs about 1 GB of GPU memory for the projector. The voice loop does not
# use it, so it is off unless explicitly asked for.
VISION="${LLAMA_VISION:-0}"

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

# Match ONLY our server, by port. Ollama spawns the very same llama-server binary
# for its own runner, so a bare `pgrep -x llama-server` matches Ollama's process
# too — which made `status` report healthy while our server was actually dead.
pids() {
  local out=""
  for p in $(pgrep -x llama-server 2>/dev/null); do
    if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q -- "--port $PORT"; then
      out="$out $p"
    fi
  done
  echo $out
}

do_stop() {
  systemctl --user stop "$UNIT" 2>/dev/null
  systemctl --user reset-failed "$UNIT" 2>/dev/null
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
  local procs health
  procs=$(pids)
  health=$(curl -s --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null)

  # Health is the source of truth, not the presence of a process. A process that
  # is not serving is not "running" for any purpose we care about.
  if [[ "$health" != *ok* ]]; then
    if [[ -n "$procs" ]]; then
      echo "llama-server: pid$procs alive but NOT serving on $PORT — restart it"
    else
      echo "llama-server: not running"
    fi
    return 1
  fi

  echo "llama-server: pid$procs | port $PORT | health $health"
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

  # Prefer an explicit model_path from config.yaml (a local .gguf), then the env
  # override, then whatever `ollama pull` already downloaded.
  local blob cfg_path
  cfg_path=$(python3 - <<'PY' 2>/dev/null
import yaml
try:
    c = yaml.safe_load(open("config.yaml")) or {}
    print(((c.get("llm") or {}).get("llama_server") or {}).get("model_path") or "")
except Exception:
    print("")
PY
)
  blob="${LLAMA_MODEL_BLOB:-${cfg_path:-}}"
  [[ -n "$blob" && ! -r "$blob" ]] && { echo "config model_path not readable: $blob" >&2; exit 1; }
  blob="${blob:-$(find_blob "$MODEL_TAG")}"
  if [[ -z "${blob:-}" || ! -r "$blob" ]]; then
    echo "could not resolve a model blob for $MODEL_TAG" >&2
    echo "  try: ollama pull $MODEL_TAG   (or set LLAMA_MODEL_BLOB)" >&2
    exit 1
  fi
  # MUST be absolute. systemd-run does NOT inherit this shell's working directory,
  # so a relative model_path from config.yaml (models/foo.gguf) resolves against the
  # unit's own cwd and fails with "No such file or directory" — while the identical
  # command run by hand from the project root works fine.
  blob="$(readlink -f "$blob")"

  # DO NOT CALL `ollama stop` HERE. It looks like a way to free memory. It is the
  # opposite, and it froze this machine three times before the persistent journal
  # finally caught it:
  #
  #   16:13:02  POST /api/show        <- `ollama stop gemma3:4b`
  #   16:13:02  POST /api/generate    <- ...which LOADS the model to expire it
  #   16:13:41  ollama: "loading model via llama-server"
  #   16:13:42  ollama: "mmproj worst-case memory usage 942.23 MiB"
  #   16:16:02  ollama: "waiting for llama-server to become available" (x20)
  #   16:17:01  snapd watchdog timeout — box unresponsive, had to be power-cycled
  #
  # `ollama stop` is implemented as a generate request with keep_alive=0, so Ollama
  # has to LOAD the model — with its vision projector — before it can unload it.
  # On 7.6 GB shared memory, next to our own server, that is fatal.
  #
  # /api/ps is read-only and safe, so we look rather than touch, and report instead
  # of acting. The real fix is to not run ollama at all: `sudo systemctl disable
  # --now ollama`, since we drive llama-server directly and never use it.
  if command -v ollama >/dev/null; then
    local held
    held=$(ollama ps 2>/dev/null | tail -n +2 | grep -c . || true)
    if [[ "${held:-0}" -gt 0 ]]; then
      echo "  WARNING: ollama is holding ${held} model(s) — that memory is not available:" >&2
      ollama ps 2>/dev/null | tail -n +2 | sed 's/^/    /' >&2
      echo "    stop it from another shell, then retry:  ollama stop <name>" >&2
      echo "    (do NOT let this script call that — it loads the model to unload it)" >&2
    fi
  fi

  # ---- memory preflight -------------------------------------------------
  # WHAT THIS CHECK IS FOR, because it changed.
  #
  # Originally this was the thing standing between us and a system-wide OOM. It is
  # not any more: voice.slice now caps the whole stack with MemoryMax and denies it
  # swap, so a shortfall is a contained cgroup kill rather than a global event, and
  # sshd is never a candidate. That is the guarantee. This check is now only a
  # civilised early exit — "will this model actually load?" — so it should PREDICT
  # resident use, not pad against catastrophe. The padding is what kept refusing
  # starts that then ran with hundreds of MB to spare.
  #
  # Predicting from file size over-counts, because a q4_0 model is NEON-repacked at
  # load (which disables mmap) into a form smaller than the file. Measured here at
  # ctx 2048:
  #
  #     file on disk   3194 MB
  #     process RSS    2448 MB   <- weights AND kv AND compute buffers, all in
  #     ratio          0.77
  #
  # 0.85 is that ratio with margin, so the estimate stays slightly pessimistic
  # without being absurd. If a future model repacks differently and this is wrong,
  # the cgroup ceiling catches it — which is the right division of labour.
  local model_mb avail_mb need_mb kv_mb file_mb
  file_mb=$(( $(stat -c%s "$blob") / 1048576 ))
  model_mb=$(( file_mb * 85 / 100 ))
  kv_mb=$(( CTX / 8 ))                     # rough; tiny for 1-KV-head models
  need_mb=$(( model_mb + kv_mb ))
  [[ "$VISION" == "1" ]] && need_mb=$(( need_mb + 1000 ))
  avail_mb=$(awk '/MemAvailable/{printf "%d",$2/1024}' /proc/meminfo)

  echo "  memory: need ~${need_mb} MB, ${avail_mb} MB available, "\
       "keeping ${RESERVE_MB} MB for the agent and your session"
  if (( need_mb + RESERVE_MB > avail_mb )); then
    cat >&2 <<EOF
  REFUSING TO START — not enough memory.
    model            ${model_mb} MB   (${file_mb} MB file x 0.85 repack ratio)
    kv               ${kv_mb} MB
    reserve          ${RESERVE_MB} MB   (agent + editor + desktop)
    ---------------- 
    required         $(( need_mb + RESERVE_MB )) MB
    available        ${avail_mb} MB

  Free memory and retry. Common culprits:
    docker ps                      # a container holding a GB
    ollama ps && ollama stop <m>   # Ollama holding a second copy of a model
    LLAMA_CTX=1024 tools/llama_server.sh start     # smaller KV
  Starting anyway would let the kernel pick what to kill, and last time it
  picked the voice agent and then the login session.
EOF
    exit 3
  fi

  mkdir -p logs
  : > "$LOG"
  local mmproj_args=()
  if [[ "$VISION" == "1" ]]; then
    mmproj_args=(--mmproj "$blob")
    echo "  vision: ON (projector costs ~1 GB)"
  else
    echo "  vision: off (LLAMA_VISION=1 to enable)"
  fi

  echo "starting llama-server: $(basename "$blob") ctx=$CTX threads=$THREADS port=$PORT"
  # A unit that exited — cleanly or by cgroup OOM kill — lingers in systemd's state
  # table, and systemd-run then refuses with "Unit llama-server.service already
  # exists". Since MemoryMax makes an OOM exit an EXPECTED outcome here, clearing
  # this is part of a normal start, not error recovery.
  systemctl --user reset-failed "$UNIT" 2>/dev/null
  systemctl --user stop "$UNIT" 2>/dev/null
  # Run as a transient user unit with a HARD memory ceiling. If this process ever
  # runs away it is killed inside its own cgroup, instead of the kernel picking the
  # agent or your shell. MemorySwapMax=0 stops it thrashing swap on the way there.
  systemd-run --user --quiet \
    --unit="$UNIT" \
    --slice=voice.slice \
    --property=MemoryMax="${MEM_MAX_MB}M" \
    --property=MemorySwapMax=0 \
    --property=OOMScoreAdjust=1000 \
    --property=CPUAffinity=0-2 \
    --property=LimitCORE=0 \
    --property=Restart=no \
    --property=StandardOutput="append:$PWD/$LOG" \
    --property=StandardError="append:$PWD/$LOG" \
    --setenv=GGML_BACKEND_PATH="$BACKEND" \
    --setenv=LD_LIBRARY_PATH="$OLLAMA_LIB/cuda_jetpack6:$OLLAMA_LIB" \
    "$BIN" \
      --model "$blob" \
      "${mmproj_args[@]}" \
      --host 127.0.0.1 --port "$PORT" \
      --ctx-size "$CTX" --n-gpu-layers 99 --threads "$THREADS" \
      --parallel 1 \
      --reasoning off --reasoning-budget 0 \
    || { echo "systemd-run failed" >&2; exit 1; }

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
