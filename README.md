# Cambia

A real-time multiplayer card game (also known as Cabo/Cambio) with AI opponents trained via Deep Counterfactual Regret Minimization.

## Components

- **[engine/](engine/)**: Cambia game engine (Go). Core game rules, agent belief state, and tensor encoding. Builds to a shared library for FFI.
- **[service/](service/)**: Multiplayer game server (Go). WebSocket gameplay, REST API, auth, matchmaking.
- **[cfr/](cfr/)**: Deep CFR training pipeline (Python). Neural network training, outcome/external sampling MCCFR.
- **[web/](web/)**: Web client (React/TypeScript). Real-time game UI.

## Quick Start

```bash
# Game server
cd service && docker compose up -d && go run cmd/server/main.go

# Web client
cd web && npm install && npm run dev

# CFR training
cd cfr && pip install -e . && python -m pytest tests/

# Build shared library for training FFI
make libcambia
```

## Testing

```bash
make test           # engine + service + cfr test suites
make parity-gate     # Go vs Python cross-engine lockstep parity
```

`make parity-gate` is the migration acceptance for retiring the Python reference engine: it builds
the shared library, then plays both engines in lockstep over one declared seed set, asserting equal
acting player, legal action set, pile lengths, hands, token stream, and terminal utilities after
every action. Seed breadth is the single constant in `cfr/tests/parity_seeds.py` (default 40);
override it with `CAMBIA_PARITY_SEEDS=200 make parity-gate`. The 4-seat leg is currently a named
skip, blocked on cambia-1419 (the Python engine cannot play above 2 seats); 2-seat parity is strict.
CI runs this gate on changes to `engine/`, `cfr/src/game/`, and `cfr/src/encoding.py`
(`.github/workflows/parity-gate.yml`).

## Serving Harness

Training and evaluation jobs can run on a remote runner host instead of locally. `cambia harness submit` pins the current commit, pushes it to the runner's git mirror, and submits a train or evaluate job description to a bounded queue on the runner daemon (`runnerd`). The runner stages an isolated worktree and Python environment, launches the job, and streams status and logs back over a TLS + Bearer-JWT control plane; artifacts and `run_db` rows reconcile back to the local machine with `cambia harness pull` or `cambia harness watch`.

The client ships as the `harness` pip extra (see [cfr/README.md](cfr/README.md)) and installs the `cambia harness` CLI. See [runnerd/README.md](runnerd/README.md) for running the runner daemon and [docs/serving-harness/architecture.md](docs/serving-harness/architecture.md) for the full design.

## Game Rules

See [RULES.md](RULES.md) for the full Cambia ruleset.
