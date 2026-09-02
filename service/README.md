# cambia-service

> Websocket-based game handler for Cambia

## Table of Contents

- [cambia-service](#cambia-service)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Running the Server](#running-the-server)
  - [Architecture Note](#architecture-note)
  - [Historian](#historian)
    - [Who owns the `games` row](#who-owns-the-games-row)
    - [Retries and the dead-letter list](#retries-and-the-dead-letter-list)
  - [License](#license)

## Getting Started

### Prerequisites

- **Go** version 1.20 or higher installed on your machine.

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/jason-s-yu/cambia.git
   cd cambia
   ```

2. **Install Dependencies**

   Ensure you have Go modules enabled:

   ```bash
   go mod tidy
   ```

### Running the Server

Run the server using the `go run` command:

```bash
go run main.go
```

The server will start and listen on `http://localhost:8080`.

Alternatively, using Air for hot-reloading:

```bash
air
```

## Architecture Note

`internal/game/game.go` wraps `engine.GameState` from the Go engine package. All game rule logic is authoritative from the engine (the service does not duplicate it). The `CambiaGame` struct holds an `Engine engine.GameState` field and dispatches WebSocket actions through `applyEngineAction`, mapping between service player UUIDs and engine player indices.

## Historian

`cmd/db/historian.go` builds `cambia-historian`, a separate process from the game server. It pops
game action records off a Redis list (`HISTORIAN_QUEUE_NAME`, default `cambia_actions`), batches
them, and writes them to `game_actions`. It is the only writer of that table.

### Who owns the `games` row

The game server owns it. `database.UpsertInitialGameState` creates it at game start with the
`lobby_id`, the round index and the initial deck and hands, and it is the only place a `games` row
is created.

The historian does not create one, does not update one back to `in_progress`, and has no fallback
for a missing one: `game_actions.game_id` is a foreign key it reads, nothing more. Its two writes to
`games` both advance an existing row's status and are conditioned on it still being `in_progress`:
the terminal `action_end_game` record moves it to `completed` and sets `end_time`, and the
inactivity loop moves it to `abandoned` after `GAME_INACTIVITY_TIMEOUT_SEC`. A row that is absent or
already closed is left alone, and in the terminal case the action still lands.

The upsert the historian used to run was a fallback from before the server wrote the row. Migration
5 (`5_add_lobby_persistence.sql`) made `games.lobby_id` `NOT NULL` with no default, and the upsert
never supplied it, so every flush failed with SQLSTATE 23502 and no game action was persisted at all
between that migration and cambia-1881. It failed even when the row already existed, because
Postgres validates `NOT NULL` against the proposed tuple before it resolves `ON CONFLICT`.

### Retries and the dead-letter list

The server writes the `games` row from a background goroutine and queues the game's first actions in
parallel, so an action can reach the historian before its row commits. That shows up as SQLSTATE
23503 on `game_actions_game_id_fkey` and clears on its own, so the flush waits it out.

A batch is first written in one transaction. If that fails, the historian retries record by record,
so one action that cannot land does not hold the rest of its batch out of the table. The retry runs
`HISTORIAN_RETRY_ATTEMPTS` passes (default 5) with `HISTORIAN_RETRY_BASE_MS` (default 50) doubling
between them: 50, 100, 200 and 400 ms, 750 ms of waiting in total, capped by a 5s ceiling on any one
flush's retrying so a `SIGTERM` drain stays inside the container's 15s `stop_grace_period`.

Whatever still fails after those passes is pushed onto a Redis list named
`cambia_actions_dead_letter` (`HISTORIAN_DEAD_LETTER_QUEUE_NAME` overrides it) and logged at ERROR
with its game id, action index and SQLSTATE. Nothing pops that list; it is there so a failed write is
inspectable rather than lost. Each entry is a JSON object holding the original record under
`record`, plus `reason`, `sql_state`, `attempts` and `failed_at`, so an entry can be replayed by
pushing its `record` field back onto the live action queue. If the push to Redis itself fails, the
entry is written to the log in full instead.

Environment variables, all optional:

| Variable | Default | Meaning |
|-|-|-|
| `HISTORIAN_QUEUE_NAME` | `cambia_actions` | Redis list the game server publishes actions to |
| `HISTORIAN_DEAD_LETTER_QUEUE_NAME` | `cambia_actions_dead_letter` | Redis list for records that failed every write attempt |
| `HISTORIAN_BATCH_SIZE` | `20` | Records accumulated before a size-triggered flush |
| `HISTORIAN_FLUSH_MS` | `500` | Ticker interval for the time-triggered flush |
| `HISTORIAN_RETRY_ATTEMPTS` | `5` | Per-record write passes before a record is dead-lettered |
| `HISTORIAN_RETRY_BASE_MS` | `50` | First backoff between those passes, doubling each time |
| `GAME_INACTIVITY_TIMEOUT_SEC` | `600` | Idle time after which a game is marked `abandoned` |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
