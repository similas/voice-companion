#!/usr/bin/env bash
# Launch the companion. Keeps you from having to remember the venv path.
set -euo pipefail
cd "$(dirname "$0")"

VENV="$HOME/.venvs/voice-companion"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "venv missing at $VENV — see README.md (Install)" >&2
  exit 1
fi

# The venv deliberately lives OUTSIDE this directory. nltk 3.10 refuses to import
# any module located under the current working directory, so a venv at ./.venv
# would make every dependency unimportable. See README "Why the venv is elsewhere".
# Always tee into logs/agent.log so `./watch.sh` works no matter how the agent was
# started — foreground, nohup, or the systemd unit. Without this, a run started by
# hand logged somewhere else and watch.sh silently followed a stale file.
mkdir -p logs

# SINGLE INSTANCE, ENFORCED BY THE KERNEL.
#
# Two agents were once running at the same time, and the failure mode is genuinely
# confusing rather than obviously broken: each one hears the OTHER through the room
# speaker, transcribes it, and answers it. The log fills with plausible turns whose
# transcripts are the other instance's replies word for word, interleaved turn
# counters from two metrics loggers, and it reads exactly like "the bot is hearing
# itself" — an echo-cancellation bug that does not exist.
#
# It happened because a hand-rolled "kill the old one" loop matched process name
# `python3`, while the venv binary reports `python`. It matched nothing, so every
# restart silently added an instance.
#
# flock removes the class of bug: the lock is held by the kernel for as long as this
# process lives, and released automatically however it dies. No PID file to go
# stale, no pattern to get wrong.
exec 9>"logs/.agent.lock"
if ! flock -n 9; then
  echo "An agent is already running (holding logs/.agent.lock)." >&2
  echo "Two instances talk to EACH OTHER through the speaker — refusing to start." >&2
  echo "  running: $(ps -o pid=,lstart= -p "$(cat logs/.agent.pid 2>/dev/null)" 2>/dev/null || echo unknown)" >&2
  echo "  stop it:  ./stop.sh" >&2
  exit 1
fi
echo $$ > logs/.agent.pid

# Make sure there is a REAL output device before starting. The USB speaker comes up
# with its card profile "off" after a boot, leaving `auto_null` as the only sink —
# and the agent then speaks into a null device with no error anywhere. You talk to
# it, it answers, and you hear nothing. voice-audio.service normally handles this
# at login; this is the belt-and-braces for a session where it did not run.
if [[ -x tools/ensure_audio.sh ]]; then
  tools/ensure_audio.sh || echo "WARNING: no usable audio output — you will not hear replies" >&2
fi

# Run INSIDE voice.slice, which caps the whole stack (this agent + llama-server).
# Without this the agent was uncapped, so a shortfall became a GLOBAL out-of-memory
# event — and a global OOM lets the kernel kill anything, including the sshd
# holding your login session. Contained in the slice, the kill lands in here.
#
# --scope, not --unit: a scope adopts THIS process in THIS terminal, so Ctrl-C,
# the tty, and the tee below all keep working exactly as before. A service unit
# would detach and break both.
#
# If systemd-run is unavailable for any reason, fall back to running uncontained
# rather than refusing to start — but say so, because the protection is off.
if command -v systemd-run >/dev/null && systemctl --user is-active voice.slice >/dev/null 2>&1; then
  # taskset 3-5: llama-server owns cores 0-2 (set in tools/llama_server.sh), the
  # agent (whisper + piper + vad) owns 3-5. Partitioned, they stop evicting each
  # other's caches mid-turn — the 6-core free-for-all measured 3.8-6.2x slowdowns.
  # taskset, not systemd AllowedCPUs: the cpuset controller is not delegated to the
  # user slice on this box (cgroup.controllers: memory pids), so AllowedCPUs would
  # be silently ignored, while an exec-time affinity mask is inherited by every
  # thread the process spawns.
  exec systemd-run --user --scope --quiet --slice=voice.slice \
    -- taskset -c 3-5 "$VENV/bin/python" -u -m app.main "$@" 2>&1 | tee -a logs/agent.log
else
  echo "WARNING: voice.slice unavailable — running WITHOUT a memory ceiling." >&2
  echo "         Start it with: systemctl --user start voice.slice" >&2
  taskset -c 3-5 "$VENV/bin/python" -u -m app.main "$@" 2>&1 | tee -a logs/agent.log
fi
