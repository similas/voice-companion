#!/usr/bin/env bash
# Stop the companion and release the mic + camera.
#
# The bracketed pattern matters: `pkill -f app.main` also matches THIS script's
# own command line and kills the shell running it. `app[.]main` matches the
# literal string "app.main" without matching itself.
#
# Note the process is named `python` (not python3) because run.sh execs the venv
# interpreter directly — `pgrep -x python3` will not find it.
pkill -f "app[.]main" 2>/dev/null
sleep 2
if pgrep -f "app[.]main" >/dev/null; then
  pkill -9 -f "app[.]main"
  sleep 1
fi
if pgrep -f "app[.]main" >/dev/null; then
  echo "still running:"; pgrep -af "app[.]main"; exit 1
fi
echo "stopped; camera $(fuser /dev/video0 2>/dev/null >/dev/null && echo 'STILL HELD' || echo free)"
