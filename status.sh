#!/usr/bin/env bash
# Is the companion up, and is everything it needs healthy?
cd "$(dirname "$0")"
printf "service      : %s\n" "$(systemctl --user is-active voice-companion 2>/dev/null)"
printf "enabled      : %s\n" "$(systemctl --user is-enabled voice-companion 2>/dev/null)"
printf "pid          : %s\n" "$(systemctl --user show voice-companion -p MainPID --value 2>/dev/null)"
printf "speaker      : %s\n" "$(pactl list short sinks 2>/dev/null | awk '/bluez/{print $2" ("$5")"; f=1} END{if(!f)print "NOT CONNECTED"}')"
printf "default sink : %s\n" "$(pactl get-default-sink 2>/dev/null)"
printf "volume       : %s\n" "$(pactl list sinks 2>/dev/null | grep -A15 bluez_output | grep -m1 'Volume:' | grep -oE '[0-9]+%' | head -1)"
printf "mic          : %s\n" "$(pactl get-default-source 2>/dev/null | sed 's/alsa_input\.//')"
printf "model        : %s\n" "$(ollama ps 2>/dev/null | awk 'NR==2{print $1" "$4" "$5}')"
printf "memory free  : %s MB\n" "$(awk '/MemAvailable/{printf "%d", $2/1024}' /proc/meminfo)"
printf "turns logged : %s\n" "$(grep -c '^  turn' logs/agent.log 2>/dev/null || echo 0)"
printf "latest CSV   : %s\n" "$(ls -t logs/latency_*.csv 2>/dev/null | head -1)"
