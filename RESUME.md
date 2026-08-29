# cambia-737 relaunch runbook

How to restart the gate-vs-served measurement after a host restart, an OOM kill,
or any other interruption. The driver resumes from disk: every completed shard
is skipped, so a relaunch costs only the work that had not finished.

## What is running

Three independent drivers, one per X2R run directory, each spawning
single-threaded shard workers:

    cfr/scripts/gate_vs_served_gap.py run --run-dir <run> --work-dir <work> ...

Work directory (all state lives here; nothing is ever written into the run dirs):

    /tmp/claude-1000/-home-jasonyu-dev-cambia/140ba070-3e09-4e18-aea6-79f338124402/scratchpad/x2gap

Note it is under `/tmp`. If the host restart clears `/tmp`, everything below is
gone and the relaunch is a cold start (~2700 snapshot forwards). Copy the work
directory somewhere persistent if that matters.

## Relaunch commands

Run from `<worktree>/cfr` with `PYTHONPATH=$PWD`. Set `W` to the work directory
above. The same commands are what launched the run, verbatim: re-issuing one is
always safe, it recomputes nothing that is already on disk.

    cd /home/jasonyu/dev/cambia/.claude/worktrees/agent-a500da34a8197f59b/cfr
    export PYTHONPATH=$PWD
    W=/tmp/claude-1000/-home-jasonyu-dev-cambia/140ba070-3e09-4e18-aea6-79f338124402/scratchpad/x2gap
    PY=/home/jasonyu/dev/cambia/cfr/.venv/bin/python
    RUNS=/home/jasonyu/dev/cambia/runs

C-rep (1000 snapshots; checkpoints 350 / 700 / 940 / 1000, 41 shards):

    $PY scripts/gate_vs_served_gap.py run --run-dir $RUNS/v0.4-x2r-crep-xpu \
      --work-dir $W --checkpoints 350,700,940,1000 --workers 6 --shard-size 25 \
      --verify-forward 256

S2 (1000 snapshots; checkpoints 350 / 700 / 1000, 40 shards):

    $PY scripts/gate_vs_served_gap.py run --run-dir $RUNS/v0.4-x2r-s2-xpu \
      --work-dir $W --checkpoints 350,700,1000 --workers 6 --shard-size 25 \
      --verify-forward 256

C1 (747 snapshots on disk, so the 1000 cell is unavailable and the plan stops at
700; checkpoints 350 / 700, 28 shards). C1's net is larger than the other two
cells (11.8MB snapshots vs 3.1MB), so its shards run ~3.7x longer:

    $PY scripts/gate_vs_served_gap.py run --run-dir $RUNS/v0.4-x2r-c1-xpu \
      --work-dir $W --checkpoints 350,700,1000 --workers 5 --shard-size 25 \
      --verify-forward 256

`--checkpoints 1000` on C1 is dropped automatically with a printed notice; leave
it in so the command is identical if C1 is later mirrored to 1000.

The original launcher that runs all three detached, plus the completion watcher
that touches the `GAP_DONE` sentinel, are in the scratchpad:

    scratchpad/launch.sh     three drivers, detached, then writes logs/DONE
    scratchpad/watcher.sh    waits for logs/DONE, renders the table, touches GAP_DONE

Relaunching `launch.sh` is safe and resumes. It re-materializes the pinned source
caches first (idempotent) so the three drivers never race on a `git archive`.

To confine the whole thing to a core subset (the user's current constraint,
cores 0-7 at nice 19):

    setsid nohup taskset -c 0-7 nice -n 19 bash scratchpad/launch.sh &

## Final report

Once all three drivers finish (`$W/logs/DONE` exists):

    $PY scripts/gate_vs_served_gap.py report --work-dir $W --json-out $W/gap_table.json

## Where partial state lives

    $W/src-cache/<commit>/cfr          pinned source checkout per run commit
    $W/<run-name>/tree.npz             tiny tree structure  (+ .meta.json sidecar)
    $W/<run-name>/shards/shard_LO_HI.npz   one snapshot range  (+ .info.json sidecar)
    $W/<run-name>/results.json         per-run scored checkpoints
    $W/<run-name>/results_prelim.json  hand-run partial scoring, not read by anything
    $W/<run-name>/run_meta.json        commit, recorded gate series, checkpoints
    $W/logs/{crep,s2,c1}.log           driver logs
    $W/logs/DONE                       written after all three drivers exit

A stage counts as complete only when its **sidecar** is present next to its
payload: `tree.npz` + `tree.npz.meta.json`, `shard_X_Y.npz` +
`shard_X_Y.npz.info.json`. Payloads are written to a `*.partial` temp file,
fsynced, then renamed, and the sidecar is written last, so a killed worker can
never leave a payload that reads as finished. The driver sweeps stray `*.partial`
files on startup.

## What is safe to delete

Safe, costs only recomputation:

- any `shard_*.npz` plus its `.info.json` (that range recomputes; ~11 min for
  C-rep/S2, ~40 min for C1 at 25 snapshots per shard)
- `results.json`, `results_prelim.json`, `gap_table.json` (re-derived by `score`
  and `report` from the shards in seconds)
- `$W/logs/*` (logs only; deleting `DONE` just makes the watcher wait again)
- `$W/src-cache/` (re-extracted by `git archive` in seconds)
- `<worktree>/GAP_DONE` (the completion sentinel; untracked, never committed)

Costs a 50-second rebuild:

- `<run>/tree.npz` and its `.meta.json`. Delete BOTH or neither: a tree without
  its sidecar is treated as absent and rebuilt anyway.

Do NOT delete:

- anything under `/home/jasonyu/dev/cambia/runs/` -- the source run directories
  are read-only inputs and hold the only copy of the snapshots.

Deleting a shard without its sidecar (or vice versa) is safe but pointless; the
driver recomputes the range either way.

## Correctness invariants a relaunch must not break

- Each run is scored under the commit pinned in its own `jobspec.json` (C-rep
  `69c3338`, S2 and C1 `0aeefd1`), materialized into `$W/src-cache`. Scoring
  these pre-F1 checkpoints with current master gives silently wrong numbers
  (C-rep iteration 1 reads 1.582023 instead of the recorded 1.417861), so never
  point `--src-root` at the live worktree.
- Shard boundaries must land on every checkpoint. Changing `--shard-size` or
  `--checkpoints` between a first run and a relaunch leaves shards that no
  longer align; the `score` stage refuses to combine them rather than folding a
  wrong snapshot set. If that happens, delete that run's `shards/` and restart it.
- The `score` stage rebuilds the tree and compares a SHA-256 of the infoset
  ordering against the one recorded at prepare time, so a shard set can never be
  applied to a differently ordered tree.
