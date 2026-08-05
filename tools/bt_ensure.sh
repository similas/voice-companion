#!/usr/bin/env bash
# Make sure the Bluetooth speaker is connected and is the default sink.
#
# Light-weight on purpose: this runs before every service start, so it does NOT
# scan or pair. It assumes the speaker is already paired+trusted (done once by
# tools/bt_speaker.sh) and just reconnects it if the link dropped.
#
# Never fails the service start — a missing speaker should degrade to the analog
# output, not stop the agent from running.

MAC="${1:-AC:BF:71:FD:F0:B3}"
VOL="${2:-40}"

if ! bluetoothctl info "$MAC" 2>/dev/null | grep -q "Connected: yes"; then
  bluetoothctl connect "$MAC" >/dev/null 2>&1
  # A2DP takes a moment to negotiate and for the sink to appear in PipeWire.
  for _ in $(seq 1 10); do
    pactl list short sinks 2>/dev/null | grep -q bluez && break
    sleep 1
  done
fi

SINK=$(pactl list short sinks 2>/dev/null | awk '/bluez/ {print $2; exit}')
if [[ -n "$SINK" ]]; then
  pactl set-default-sink "$SINK" 2>/dev/null
  pactl set-sink-volume "$SINK" "${VOL}%" 2>/dev/null
  pactl set-sink-mute "$SINK" 0 2>/dev/null
  echo "bt_ensure: $SINK at ${VOL}%"
else
  echo "bt_ensure: no bluez sink; falling back to the default output"
fi

# Also make the webcam mic the default source, so input_device 30 ("pulse")
# resolves to it rather than the analog line-in.
SRC=$(pactl list short sources 2>/dev/null | awk '/C960/ && !/monitor/ {print $2; exit}')
[[ -n "$SRC" ]] && pactl set-default-source "$SRC" 2>/dev/null && \
  echo "bt_ensure: mic $SRC"

exit 0
