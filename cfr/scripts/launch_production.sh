#!/usr/bin/env bash
# Launch production training runs for Cambia Deep CFR.
# Run from workspace root: bash cfr/scripts/launch_production.sh

set -euo pipefail

RUNS=(
    "prod-full-333"
    "prod-full-500"
    "prod-d20-333"
)

# Create run subdirectories
for run in "${RUNS[@]}"; do
    mkdir -p "cfr/runs/${run}/checkpoints" "cfr/runs/${run}/evaluations" "cfr/runs/${run}/logs"
done

echo "Launching training runs..."

# Launch training processes
declare -a TRAIN_PIDS

for run in "${RUNS[@]}"; do
    cambia train deep \
        --config "cfr/runs/${run}/config.yaml" \
        --headless \
        > "cfr/runs/${run}/logs/train.log" 2>&1 &
    pid=$!
    TRAIN_PIDS+=("$pid")
    echo "  ${run}: PID ${pid}"
done

# Launch eval watcher
mkdir -p cfr/runs
python cfr/scripts/eval_watcher.py \
    --run-dirs cfr/runs/prod-full-333 cfr/runs/prod-full-500 cfr/runs/prod-d20-333 \
    --games 5000 \
    --poll-interval 30 \
    > cfr/runs/eval_watcher.log 2>&1 &
EVAL_PID=$!
echo "  eval_watcher: PID ${EVAL_PID}"

echo ""
echo "All PIDs:"
for i in "${!RUNS[@]}"; do
    echo "  ${RUNS[$i]}: ${TRAIN_PIDS[$i]}"
done
echo "  eval_watcher: ${EVAL_PID}"

echo ""
echo "Waiting for training runs to complete..."
for pid in "${TRAIN_PIDS[@]}"; do
    wait "$pid" || echo "  WARNING: process ${pid} exited with non-zero status"
done

echo "All training runs complete. Stopping eval watcher (PID ${EVAL_PID})..."
kill "${EVAL_PID}" 2>/dev/null || true

echo "Done."
