#!/usr/bin/env bash
# Make the USB speaker usable after a boot, and make it the default sink.
#
# WHY THIS EXISTS
# ---------------
# On every boot the speaker comes up with its card profile set to "off", so
# PipeWire creates no sink for it and the only output is `auto_null`. You can talk
# to the agent and hear nothing, with no error anywhere — the agent happily writes
# audio into a null device.
#
# The cause is in the kernel log, at USB enumeration:
#
#     usb 1-2.1: 3:1: cannot get freq at ep 0x84
#
# The device fails a clock-rate query while it is still settling. WirePlumber
# probes its profiles at that moment, the probe fails, and it falls back to "off"
# and never revisits the decision. Restarting WirePlumber later works purely
# because the device has settled by then — which is why this is a retry, not a
# workaround for a config mistake.
#
# So: wait for the card, and if it landed on "off", re-probe and select a real
# output profile. Idempotent — safe to run any time, does nothing if already fine.
#
# THIS SCRIPT NEVER CHANGES THE VOLUME. The volume is the user's setting (70%) and
# raising it has been explicitly ruled out. It only reports what the volume is.

set -uo pipefail

CARD_MATCH="${AUDIO_CARD_MATCH:-alsa_card.usb-Jieli}"
WANT_PROFILE="${AUDIO_PROFILE:-output:analog-stereo}"
TIMEOUT="${AUDIO_TIMEOUT:-45}"

log() { echo "  $*"; }

# Wait for PipeWire itself, then for the card to be enumerated. On a cold boot the
# USB device can take many seconds to appear, so polling beats a fixed sleep.
for _ in $(seq 1 "$TIMEOUT"); do
  pactl info >/dev/null 2>&1 && break
  sleep 1
done
if ! pactl info >/dev/null 2>&1; then
  log "PipeWire not responding after ${TIMEOUT}s — giving up"
  exit 1
fi

card=""
for _ in $(seq 1 "$TIMEOUT"); do
  card=$(pactl list short cards 2>/dev/null | awk -v m="$CARD_MATCH" '$2 ~ m {print $2; exit}')
  [[ -n "$card" ]] && break
  sleep 1
done
if [[ -z "$card" ]]; then
  log "no card matching '$CARD_MATCH' — is the speaker plugged in?"
  exit 1
fi

profile=$(pactl list cards 2>/dev/null | awk -v c="$card" '
  $0 ~ "Name: "c {f=1} f && /Active Profile:/ {print $3; exit}')
log "card $card (profile: $profile)"

if [[ "$profile" == "off" ]]; then
  # The profile list itself is often wrong at this point — a failed probe leaves
  # only "off" and "pro-audio", with the real output profiles missing entirely.
  # Restarting WirePlumber forces a fresh probe against a now-settled device.
  if ! pactl list cards 2>/dev/null | awk -v c="$card" '$0 ~ "Name: "c {f=1} f' | grep -q "$WANT_PROFILE"; then
    log "profile '$WANT_PROFILE' not offered — re-probing (device was not ready at boot)"
    systemctl --user restart wireplumber
    for _ in $(seq 1 20); do pactl info >/dev/null 2>&1 && break; sleep 1; done
    sleep 3
  fi
  log "setting profile -> $WANT_PROFILE"
  pactl set-card-profile "$card" "$WANT_PROFILE" 2>/dev/null \
    || { log "could not set profile"; exit 1; }
  sleep 1
fi

sink=$(pactl list short sinks 2>/dev/null | awk -v m="$CARD_MATCH" '
  { split(m,a,"alsa_card."); } $2 ~ "usb-Jieli" {print $2; exit}')
if [[ -z "$sink" ]]; then
  log "profile set but no sink appeared — check: pactl list short sinks"
  exit 1
fi

pactl set-default-sink "$sink" 2>/dev/null
# Unmute only. Volume is deliberately left exactly where the user put it.
pactl set-sink-mute "$sink" 0 2>/dev/null
vol=$(pactl get-sink-volume "$sink" 2>/dev/null | head -1 | grep -o '[0-9]*%' | head -1)
log "default sink: $sink"
log "volume: ${vol:-unknown} (unchanged by this script)"
