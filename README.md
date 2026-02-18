# Cambia

A real-time multiplayer card game (also known as Cabo/Cambio) with AI opponents trained via Deep Counterfactual Regret Minimization.

## Components

- **[engine/](engine/)** — Cambia game engine (Go). Core game rules, agent belief state, and tensor encoding. Builds to a shared library for FFI.
- **[service/](service/)** — Multiplayer game server (Go). WebSocket gameplay, REST API, auth, matchmaking.
- **[cfr/](cfr/)** — Deep CFR training pipeline (Python). Neural network training, outcome/external sampling MCCFR.
- **[web/](web/)** — Web client (React/TypeScript). Real-time game UI.

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

## Game Rules

See [RULES.md](RULES.md) for the full Cambia ruleset.
