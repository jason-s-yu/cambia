#!/bin/bash
# Monitors training log and annotates each iteration with CPU load.
# Usage: ./training_load_monitor.sh runs/rebel-phase1/logs/training.log
LOG="${1:-runs/rebel-phase1/logs/training.log}"
OUT="${LOG%.log}_load.log"
echo "# Training load monitor — $(date -Iseconds)" > "$OUT"
echo "# Format: iter | self_play_s | load_1m | load_5m | cpus_busy" >> "$OUT"
tail -f "$LOG" | while read -r line; do
  if echo "$line" | grep -q '^\[rebel\] iter'; then
    iter=$(echo "$line" | grep -oP 'iter \K[0-9]+')
    sp=$(echo "$line" | grep -oP 'self_play=\K[0-9.]+')
    load=$(cat /proc/loadavg | awk '{print $1, $2}')
    echo "iter=$iter self_play=${sp}s load=${load} $(date +%H:%M:%S)" >> "$OUT"
    echo "iter=$iter self_play=${sp}s load=${load} $(date +%H:%M:%S)"
  fi
done
