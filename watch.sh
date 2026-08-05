#!/usr/bin/env bash
# Follow the live conversation: per-turn transcript, reply, and stage latencies.
# Filters out ALSA/PortAudio noise and deprecation warnings so you see only turns.
#
#   ./watch.sh          follow live
#   ./watch.sh -n 50    show the last 50 lines of turn history first
set -uo pipefail
cd "$(dirname "$0")"
LOG=logs/agent.log
[[ -f "$LOG" ]] || { echo "no log yet — is the service running? (./status.sh)"; exit 1; }
N=200; [[ "${1:-}" == "-n" ]] && N="${2:-200}"
tail -n "$N" -f "$LOG" \
  | grep --line-buffered -viE "ALSA lib|snd_|onnxruntime|device_discovery|DeprecationWarning|warnings\.warn|^\s+(stt|tts|llm)?\s*$" \
  | grep --line-buffered -E "turn |you:|bot:|stt |TOTAL|vision:|llm warm|noise gate|ERROR|WARNING|bt_ensure|Speak when ready"
