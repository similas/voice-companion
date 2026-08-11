#!/usr/bin/env bash
# Listen to exactly what the STT model heard, in a browser.
#
#   tools/listen.sh          serve on :8090
#   tools/listen.sh stop
#
# WHY
# ---
# "It misheard me" has two causes with opposite fixes: the microphone captured
# something unintelligible, or it captured fine and the model got it wrong. Reading
# a transcript cannot tell them apart. Hearing the audio can, in seconds.
#
# These WAVs are the exact arrays handed to whisper (written from inside the STT
# service, after VAD segmentation), so the file boundaries ARE the turn boundaries.
# That makes clipped starts and mid-sentence splits audible, not just inferable.
#
# Playing them through the Jetson's own speaker would be simpler, but the agent is
# usually listening and would transcribe them as new turns — so this serves them to
# YOUR machine instead.

set -uo pipefail
cd "$(dirname "$0")/.."

PORT="${LISTEN_PORT:-8090}"
DIR="logs/turns"
LOG="logs/agent.log"

if [[ "${1:-}" == "stop" ]]; then
  # Match the port, not the word "http.server" — this script's own command line
  # contains that string and would match itself.
  for p in $(pgrep -f "http.server $PORT" 2>/dev/null); do kill "$p" 2>/dev/null; done
  echo "stopped"
  exit 0
fi

[[ -d "$DIR" ]] || { echo "no recordings yet — is stt.save_audio true in config.yaml?" >&2; exit 1; }

# Pair each WAV with the transcript and level the agent logged for it.
python3 - "$DIR" "$LOG" <<'PY'
import html, re, sys
from pathlib import Path
d, logp = Path(sys.argv[1]), Path(sys.argv[2])
meta = {}
if logp.exists():
    for ln in logp.read_text(errors="ignore").splitlines():
        m = re.search(r"stt: saved (\S+) \(([\d.]+)s, rms ([\d.]+)\) -> (.*)$", ln)
        if m:
            meta[Path(m.group(1)).name] = (m.group(2), m.group(3), m.group(4))
rows = []
for w in sorted(d.glob("*.wav")):
    secs, rms, text = meta.get(w.name, ("?", "?", "(no log line)"))
    try:
        lvl = float(rms)
        verdict = ("SILENT — mic problem" if lvl < 0.005 else
                   "quiet" if lvl < 0.02 else "healthy")
    except ValueError:
        verdict = ""
    rows.append(f"""<tr><td class=n>{html.escape(w.name)}</td>
<td><audio controls preload=none src="{html.escape(w.name)}"></audio></td>
<td class=m>{secs}s</td><td class=m>{rms}<span class=v>{verdict}</span></td>
<td class=t>{html.escape(text)}</td></tr>""")
(d / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>What the STT heard</title>
<style>
 :root{{color-scheme:light dark}}
 body{{font:15px/1.5 system-ui,sans-serif;margin:2rem;max-width:60rem}}
 h1{{font-size:1.2rem}} p{{opacity:.75}}
 table{{border-collapse:collapse;width:100%}}
 td,th{{padding:.5rem .6rem;border-bottom:1px solid #8883;vertical-align:middle}}
 .n{{font-family:ui-monospace,monospace;font-size:.85em;white-space:nowrap}}
 .m{{font-family:ui-monospace,monospace;font-size:.85em;text-align:right;white-space:nowrap}}
 .t{{font-weight:600}} .v{{display:block;font-weight:400;opacity:.7;font-size:.8em}}
 audio{{height:2rem;width:15rem}}
</style>
<h1>What the STT model actually heard</h1>
<p>The exact audio handed to whisper, after VAD segmentation. File boundaries are
turn boundaries — a clipped start or a sentence split at a pause is audible here.
rms below ~0.005 means the microphone is the problem; ~0.02+ means it isn't.</p>
<table><tr><th>file</th><th>play</th><th>len</th><th>rms</th><th>transcribed as</th></tr>
{''.join(rows) or '<tr><td colspan=5>no recordings yet</td></tr>'}</table>""")
print(f"  {len(rows)} recording(s)")
PY

for p in $(pgrep -f "http.server $PORT" 2>/dev/null); do kill "$p" 2>/dev/null; done
sleep 1
nohup python3 -m http.server "$PORT" --directory "$DIR" >/dev/null 2>&1 &
disown
sleep 1

TS=$(tailscale ip -4 2>/dev/null | head -1)
LAN=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -vE '^(172\.1[78]\.|127\.|100\.)' | head -1)
echo
[[ -n "$TS"  ]] && echo "  anywhere (Tailscale):  http://$TS:$PORT/"
[[ -n "$LAN" ]] && echo "  at home (WiFi):        http://$LAN:$PORT/"
echo
echo "  re-run this after more turns to refresh the list"
echo "  stop it with:  tools/listen.sh stop"
