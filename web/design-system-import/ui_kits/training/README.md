# Training dashboard UI kit

Admin-side CFR training dashboard (local hardware, not public). One screen: runs sidebar + run detail with win-rate chart, summary, checkpoints, eval jobs, live log, and a GPU/CPU/disk resource strip.

Data shapes mirror `web/src/types/training.ts` (Run, ProcessStatus, EvalJob, GPUStat) and `service/internal/training/`. Remote (serving-harness) runs render read-only with a sync badge, per `runnerd/README.md`.
