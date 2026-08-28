# Lobby WebSocket Actions (docs/lobby_actions.md)

This document describes the JSON payloads used for WebSocket communication on the `/lobby/ws/{lobby_id}` endpoint, using the `lobby` subprotocol.

**Conventions:**

* Every payload is a JSON object with a mandatory top-level `type` key (string).
* UUIDs are strings (e.g., `"f47ac10b-58cc-4372-a567-0e02b2c3d479"`).
* Timestamps (`ts`) are UNIX seconds (integer).
* User identification uses `user_id` (string) or `userID` (string) in client->server messages, and `user_id` (string) or structured objects like `{"id": "{uuid}"}` in server->client messages, often nested under keys like `user_join` or `user_left`. Consistency varies slightly.

## Client → Server Commands

| Action                   | `type` String    | Payload                                                                                                | Handler Location                 | Notes                                                      |
| :----------------------- | :--------------- | :----------------------------------------------------------------------------------------------------- | :------------------------------- | :--------------------------------------------------------- |
| Mark Ready               | `ready`          | *(None)* | `internal/handlers/lobby_ws.go`  | Marks sender as ready. May trigger countdown if autoStart. |
| Mark Unready             | `unready`        | *(None)* | `internal/handlers/lobby_ws.go`  | Marks sender as unready. Cancels any active countdown.     |
| Invite User              | `invite`         | `{ "userID": "{uuid}" }`                                                                               | `internal/handlers/lobby_ws.go`  | Invites another user to a private lobby.                   |
| Leave Lobby              | *(not a WS message)* | *(None)* | `internal/handlers/lobby.go` | `POST /lobby/{id}/leave`. Leaving releases membership, which nothing a lost socket can also trigger may do, so it is an HTTP call rather than a frame (cambia-807). Refused with 409 while the lobby's game is in progress. |
| Send Chat Message        | `chat`           | `{ "msg": "Your message here" }`                                                                       | `internal/handlers/lobby_ws.go`  | Sends a chat message to the lobby.                         |
| Update Rules (Host Only) | `update_rules`   | `{ "rules": { ... partial HouseRules object ... } }` (See `internal/game/rules.go` for fields)       | `internal/handlers/lobby_ws.go`  | Host updates lobby's house rules or circuit settings.    |
| Force Start (Host Only)  | `start_game`     | *(None)* | `internal/handlers/lobby_ws.go`  | Host attempts to start the game manually (if all ready).   |

## Server → Client Events

These messages are typically broadcast to all users in the lobby unless specified otherwise.

| Event Description             | `type` String             | Payload Example / Key Fields                                                                                                                                                                | Emitter Location           | Notes                                                                                             |
| :---------------------------- | :------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------- | :------------------------------------------------------------------------------------------------ |
| User Joined / Left            | `lobby_update`            | `{ "user_join": "{uuid}", "is_host": bool, "lobby_status": { ... } }` OR `{ "user_left": "{uuid}", "lobby_status": { ... } }`                                                                | `internal/game/lobby.go`   | Sent when a user connects or disconnects. Includes updated `lobby_status`.                        |
| Full Lobby State (Private)    | `lobby_state`             | `{ "lobby_id", "host_id", "your_id", "your_is_host", "lobby_type", "game_mode", "in_game", "game_id", "house_rules": {...}, "circuit": {...}, "settings": {...}, "lobby_status": { ... } }` | `internal/game/lobby.go`   | Sent privately to a user upon joining/connecting.                                                 |
| User Ready State Change       | `ready_update`            | `{ "user_id": "{uuid}", "is_ready": bool }`                                                                                                                                                  | `internal/game/lobby.go`   | Sent when a user's ready state changes.                                                           |
| User Invited                  | `lobby_invite`            | `{ "invitedID": "{uuid}" }`                                                                                                                                                                  | `internal/game/lobby.go`   | Sent when a user is invited via the `invite` command.                                             |
| Countdown Started             | `lobby_countdown_start`   | `{ "seconds": int }`                                                                                                                                                                        | `internal/game/lobby.go`   | Sent when the auto-start countdown begins.                                                        |
| Countdown Canceled            | `lobby_countdown_cancel`  | *(None)* | `internal/game/lobby.go`   | Sent if the countdown is stopped (e.g., user leaves or becomes unready).                          |
| Rules Updated                 | `lobby_rules_updated`     | `{ "house_rules": { ... full HouseRules object ... }, "circuit": { ... full Circuit object ... } }`                                                                                         | `internal/game/lobby.go`   | Sent when the host successfully updates rules via `update_rules`.                                 |
| Chat Message Received         | `chat`                    | `{ "user_id": "{uuid}", "msg": "The message", "ts": int }`                                                                                                                                  | `internal/game/lobby.go`   | Echoes a chat message sent by a user.                                                             |
| Game Started                  | `game_start`              | `{ "game_id": "{uuid}" }`                                                                                                                                                                   | `internal/handlers/lobby_ws.go` (via callback) | Sent when the game instance is created and starts. Clients should connect to `/game/ws/{game_id}`. |
| Error Occurred (Private)      | `error`                   | `{ "message": "Error description text" }`                                                                                                                                                   | `internal/game/lobby.go`   | Sent privately to the user who caused an error (e.g., invalid action, not host).                 |

**`lobby_status` Object Structure (within `lobby_update` and `lobby_state`):**

```json
{
  "users": [
    {
      "id": "{uuid}",
      "is_host": bool,
      "is_ready": bool
    }
    // ... more users
  ]
  // Potentially other status fields could be added here
}

**`HouseRules` Object Structure (within `lobby_state`, `lobby_rules_updated`, used by `update_rules`):**
(See `internal/game/rules.go` for field definitions)

```json
{
  "allowDrawFromDiscardPile": bool,
  "allowReplaceAbilities": bool,
  "allowOpponentSnapping": bool,
  "snapRace": bool,
  "lockCallerHand": bool,
  "forfeitOnDisconnect": bool,
  "disconnectGraceSec": int,
  "penaltyDrawCount": int,
  "turnTimerSec": int,
  "maxGameTurns": int,
  "cardsPerPlayer": int,
  "cambiaAllowedRound": int,
  "numJokers": int,
  "numDecks": int,
  "initialViewCount": int
}
```

Numeric rules are range-checked server-side and an out-of-range value rejects the whole
`update_rules` message, leaving the lobby's rules unchanged. The accepted ranges follow the
engine's own limits: `penaltyDrawCount` 0-6, `turnTimerSec` 0-86400, `maxGameTurns` 0-65535
(0 = unlimited), `cardsPerPlayer` 1-6, `cambiaAllowedRound` 0-255, `numJokers` 0-2,
`numDecks` 1-4, `initialViewCount` 0-6, `disconnectGraceSec` 0-3600. `initialViewCount` additionally may not exceed
`cardsPerPlayer`: the pregame peek cannot cover more cards than the hand holds, and the pair is
checked after the whole update is applied, so both keys may move in one message.

**`Circuit` Object Structure (within `lobby_state`, `lobby_rules_updated`, used by `update_rules`):**
(See `internal/game/game.go` for field definitions)

```json
{
    "enabled": bool,
    "mode": string, // e.g., "circuit_4p"
    "rules": {
        "targetScore": int,
        "winBonus": int,
        "falseCambiaPenalty": int,
        "freezeUserOnDisconnect": bool
    }
}
```

**`LobbySettings` Object Structure (within `lobby_state`, used by `update_rules`):**
(See `internal/game/lobby.go` for field definitions)

```json
{
  "autoStart": bool
}
```

**Reconnect grace (`disconnectGraceSec`)**

`disconnectGraceSec` is how long a dropped socket keeps its seat before `forfeitOnDisconnect`
takes it (default 60, range 0-3600, 0 forfeits on the drop itself). It is read only when
`forfeitOnDisconnect` is on, and it is what makes a page reload survivable: the seat is held, the
table keeps playing, and a reconnect inside the window restores the player through the usual
`private_sync_state` (see `game_actions.md`, "Disconnect grace"). Matchmaking queues do not take
this from the lobby: each queue carries its own value (`internal/matchmaking/validation.go`),
applied when the game is built, since a queued lobby has no host setting rules.
