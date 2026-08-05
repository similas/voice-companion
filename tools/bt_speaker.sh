#!/usr/bin/env bash
# Pair/connect a Bluetooth speaker and route audio to it, then play a test sound.
#
#   tools/bt_speaker.sh AC:BF:71:FD:F0:B3        # pair + connect + set default + test
#   tools/bt_speaker.sh AC:BF:71:FD:F0:B3 --no-test
#
# The speaker must be IN PAIRING MODE for a first-time pair (hold its Bluetooth
# button until the LED flashes fast). A speaker that is merely powered on
# advertises only over BLE, which cannot carry audio — bluetoothctl will say
# "not available" and that is what it means.
#
# PREREQUISITE, and the reason this script exists: WirePlumber loads its
# Bluetooth support in a SEPARATE daemon instance, and on this Ubuntu image that
# instance ships disabled. Without it BlueZ has no A2DP consumer and connecting
# fails with `br-connection-profile-unavailable`. Enabled here idempotently.

set -uo pipefail

MAC="${1:-}"
TEST=1
[[ "${2:-}" == "--no-test" ]] && TEST=0

if [[ -z "$MAC" ]]; then
  echo "usage: $0 <MAC> [--no-test]" >&2
  echo "hint: find it with  bluetoothctl --timeout 15 scan on" >&2
  exit 1
fi

say() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------------------
# Check 0: can BlueZ do audio at all?
#
# JetPack ships /usr/lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf
# which overrides ExecStart to start bluetoothd with:
#     --noplugin=audio,a2dp,avrcp
# With those plugins disabled BlueZ never exposes org.bluez.Media1, so a speaker
# pairs and reports "Connected: yes" while no audio profile and no sink ever
# appear. It looks like a PipeWire problem and is not one. Detect it up front.
# ---------------------------------------------------------------------------
say "checking BlueZ has audio plugins enabled"
BTD_ARGS="$(tr '\0' ' ' < /proc/"$(pgrep -x bluetoothd | head -1)"/cmdline 2>/dev/null || true)"
if [[ "$BTD_ARGS" == *"noplugin"*a2dp* || "$BTD_ARGS" == *"noplugin"*audio* ]]; then
  cat >&2 <<EOF
   FAIL: bluetoothd is running with audio plugins DISABLED:
     $BTD_ARGS

   This is JetPack's nv-bluetooth-service.conf drop-in. A2DP cannot work until
   it is overridden. Run this once, then re-run this script:

     sudo mkdir -p /etc/systemd/system/bluetooth.service.d
     sudo tee /etc/systemd/system/bluetooth.service.d/zz-enable-audio.conf >/dev/null <<'CONF'
     [Service]
     ExecStart=
     ExecStart=/usr/lib/bluetooth/bluetoothd
     CONF
     sudo systemctl daemon-reload && sudo systemctl restart bluetooth

   (The zz- prefix matters: drop-ins merge in filename order, so it must sort
    after nv-bluetooth-service.conf to take effect.)
EOF
  exit 4
fi
echo "   ok — audio plugins enabled"

say "checking PipeWire Bluetooth support"
# The MAIN wireplumber already loads bluetooth support — /usr/share/wireplumber/
# wireplumber.conf lists `{ name = bluetooth.lua, type = config/lua }` in
# wireplumber.components, which pulls in bluetooth.lua.d/ and bluez_monitor.
#
# Do NOT enable wireplumber@bluetooth on this version. That separate instance
# runs with `wireplumber.export-core = false`, so it grabs BlueZ's media
# application registration and then never exports the resulting device into the
# main graph — the speaker connects and no sink ever appears. Verified the hard
# way. If it is running, stop it.
if grep -q "bluetooth.lua" /usr/share/wireplumber/wireplumber.conf 2>/dev/null; then
  echo "   main wireplumber loads bluetooth.lua — good"
  if systemctl --user is-active --quiet wireplumber@bluetooth.service; then
    echo "   stopping redundant wireplumber@bluetooth (it steals the media app)"
    systemctl --user disable --now wireplumber@bluetooth.service >/dev/null 2>&1
    systemctl --user restart wireplumber.service
    sleep 4
  fi
else
  echo "   main config lacks bluetooth.lua; enabling separate instance"
  systemctl --user enable --now wireplumber@bluetooth.service >/dev/null 2>&1
  sleep 3
fi
printf '   wireplumber: %s\n' "$(systemctl --user is-active wireplumber.service)"

say "powering controller"
bluetoothctl power on >/dev/null 2>&1

# A background scan must be RUNNING for pair/connect to resolve a device that is
# not already in BlueZ's cache.
say "scanning for $MAC"
bluetoothctl scan on >/dev/null 2>&1 &
SCAN_PID=$!
trap 'kill "$SCAN_PID" 2>/dev/null; bluetoothctl scan off >/dev/null 2>&1' EXIT

found=0
for i in $(seq 1 20); do
  if bluetoothctl devices 2>/dev/null | grep -qi "$MAC"; then found=1; break; fi
  sleep 1
done
if [[ $found -eq 0 ]]; then
  echo "   NOT FOUND. The speaker is probably not in pairing mode." >&2
  echo "   Hold its Bluetooth button until the LED flashes fast, then re-run." >&2
  exit 2
fi
echo "   found"

say "pairing"
if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Paired: yes"; then
  echo "   already paired"
else
  timeout 40 bluetoothctl pair "$MAC" 2>&1 | tail -2
fi
bluetoothctl trust "$MAC" >/dev/null 2>&1

say "connecting"
for attempt in 1 2 3; do
  out=$(timeout 30 bluetoothctl connect "$MAC" 2>&1 | tail -2)
  echo "   attempt $attempt: $(echo "$out" | tr '\n' ' ')"
  if bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then break; fi
  sleep 3
done

bluetoothctl info "$MAC" 2>/dev/null | grep -E "Name|Paired|Trusted|Connected" | sed 's/^\s*/   /'

say "waiting for the audio sink to appear"
SINK=""
for i in $(seq 1 15); do
  SINK=$(pactl list short sinks 2>/dev/null | awk '/bluez/ {print $2; exit}')
  [[ -n "$SINK" ]] && break
  sleep 1
done

if [[ -z "$SINK" ]]; then
  echo "   no bluez sink appeared." >&2
  echo "   The device is connected but not offering A2DP. Try:" >&2
  echo "     systemctl --user restart wireplumber@bluetooth" >&2
  echo "     bluetoothctl disconnect $MAC && bluetoothctl connect $MAC" >&2
  exit 3
fi
echo "   sink: $SINK"

say "making it the default output"
pactl set-default-sink "$SINK"
pactl set-sink-volume "$SINK" 65% 2>/dev/null
# Move anything already playing over to the new sink.
for id in $(pactl list short sink-inputs 2>/dev/null | cut -f1); do
  pactl move-sink-input "$id" "$SINK" 2>/dev/null
done
pactl get-default-sink 2>/dev/null | sed 's/^/   default is now: /'

if [[ $TEST -eq 1 ]]; then
  say "playing a test sound"
  WAV="$(dirname "$0")/../logs/bt_test.wav"
  VENV="$HOME/.venvs/voice-companion/bin/python"
  VOICE="$(dirname "$0")/../models/en_US-lessac-medium.onnx"
  if [[ -x "$VENV" && -f "$VOICE" ]]; then
    "$VENV" - "$VOICE" "$WAV" <<'PY'
import sys, wave
from piper import PiperVoice
voice = PiperVoice.load(sys.argv[1])
with wave.open(sys.argv[2], "wb") as wf:
    voice.synthesize_wav(
        "Bluetooth speaker connected. Audio output is working.", wf)
PY
    paplay --device="$SINK" "$WAV" && echo "   played (Piper voice)"
  else
    # Fallback if Piper isn't set up yet.
    for f in /usr/share/sounds/alsa/Front_Center.wav \
             /usr/share/sounds/freedesktop/stereo/complete.oga; do
      [[ -f "$f" ]] && paplay --device="$SINK" "$f" && \
        echo "   played $(basename "$f")" && break
    done
  fi
fi

say "done — set audio.output_device in config.yaml using:"
echo "   python tools/check_env.py --list-audio"
