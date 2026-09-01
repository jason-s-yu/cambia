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
| Leave Lobby              | *(not a WS message)* | *(None)* | `internal/handlers/lobby.go` | `POST /lobby/{id}/leave`, optional body `{ "forfeit": true }`. Leaving releases membership, which nothing a lost socket can also trigger may do, so it is an HTTP call rather than a frame (cambia-807). Mid-game the answer turns on the caller's seat, not on the lobby's `inGame` flag: a caller holding no live seat (never dealt in, already forfeited, or the game has finished) leaves normally, a live seat is refused with 409, and a live seat that sent `forfeit: true` is forfeited on the spot and then leaves (cambia-1520). The flag is an opt-in a client sends only after the player has agreed to give the seat up; without it the seat sat on the table until `disconnectGraceSec` ran out. |
| Send Chat Message        | `chat`           | `{ "msg": "Your message here" }`                                                                       | `internal/handlers/lobby_ws.go`  | Sends a chat message to the lobby.                         |
| Update Rules (Host Only) | `update_rules`   | `{ "rules": { "presetId": "h2h_rapid", ... partial HouseRules object ... } }` (See `internal/game/rules.go` for fields)       | `internal/handlers/lobby_ws.go`  | Host updates lobby's house rules or circuit settings. Refused outright for a ranked or matchmaking lobby: see "Host role". An optional `presetId` inside `rules` names a whole ruleset from `GET /lobby/presets` and is expanded server-side before the field-by-field keys, so a `houseRules` object in the same message lands on top of it; an unknown id rejects the whole message (cambia-1088). The id is recorded on the lobby and echoed back as `lobby_state.preset_id`; sending one with the expanded sheet is how a client keeps the lobby's ruleset named (see "Ruleset identity"). A preset that fixes a player count fixes `game_mode` with it, and one that seats fewer players than the lobby already has is refused, changing nothing. The auto-start block is read from `settings`, with `lobbySettings` accepted as an alias (see `rest_api.md`). An accepted edit broadcasts `lobby_state` to every connected seat; a refused one broadcasts nothing. |
| Force Start (Host Only)  | `start_game`     | *(None)* | `internal/handlers/lobby_ws.go`  | Host attempts to start the game manually (if all ready). Refused for a system-hosted lobby, which starts on its ready check. |
| Cancel Search (Host Only) | `cancel_search` | *(None)* | `internal/hub/hub.go` | Party leader takes the lobby back out of its queue. Valid only in the `searching` phase, which is the phase a party still has a leader in. |
| Return To Lobby (Host Only) | `return_to_lobby` | *(None)* | `internal/hub/hub.go` | Closes the results screen and reopens the lobby without waiting out the results timer. Valid only in the `post_game` and `match_end` phases; ignored in every other phase. Host-gated, widened to any seated player where no player holds the role: see "Post-game exit". |

## Host role

Every host-gated action compares the sender against the lobby's `HostUserID`, read live rather
than cached on the socket, so the role moves with `RemoveUser`'s migration (cambia-835).

A **matchmade lobby has no player host**. The moment the matchmaker seats a match in a lobby,
that lobby's host role is handed to the system: `HostUserID` becomes the reserved sentinel
`lobby.SystemHostUserID` (the nil UUID, which no authenticated user can hold), and it never
returns to a player for the rest of the lobby's life (cambia-1087). Every quick play queue is
ranked, so the match runs on its queue's settings and there is nothing for a host to decide;
leaving one party's leader in charge of everybody else's rated game is the privilege this
removes. What follows from it:

* `your_is_host` is false in every seat's `lobby_state`, and no entry in `lobby_status.users`
  carries `is_host`. `host_id` is the nil UUID, and `system_host` is true - that flag is what
  separates "somebody else hosts this lobby" from "nobody does", which read the same off
  `your_is_host` alone.
* `update_rules` is refused with *the queue sets the rules for this match; they cannot be
  changed*. That check runs before the host check, so the refusal names the lock rather than a
  role nobody holds. It covers the auto-start setting too, which travels in the same message.
* `start_game` is refused with *this match starts on its own once every player is ready*. The
  ready check is what starts a matchmade game: auto-start is on for every lobby the create
  handler builds, and a queue-backed lobby cannot turn it off (see `rest_api.md`, `POST
  /lobby/create`), so the last ready seat begins the countdown.
* There is no rename action, on this socket or anywhere else: a lobby's `name` is set once at
  create time and never edited.

The handover happens at **match formation, not at lobby creation**. A quick play lobby is a party
before it is a match, and its party leader keeps the one power a party needs: pulling itself back
out of the queue, through `cancel_search` here or `DELETE /lobby/{id}/search`. Once a match
exists the lobby is the queue's, so both are refused with a 403 or the host-gate error from then
on.

The sentinel is never written to the database. `lobbies.host_user_id` is `NOT NULL` with an FK to
`users`, so a system-hosted lobby persists its `CreatorUserID` (whoever called
`POST /lobby/create`, stamped once and never reassigned) in that column instead.

## Post-game exit

A finished game leaves the hub in `post_game` (a casual single game) or `match_end` (a ranked
circuit), showing results. `return_to_lobby` is how a table leaves that screen: it drops the
finished game, clears the lobby's in-game flags, unreadies every seat and broadcasts
`phase_change` with `open` plus a refreshed `lobby_state`, which is the same reset the results
timer runs (`PostGameDuration`, 10s by default). Sending it is the client's only way out ahead of
the timer, and in `match_end` it is the only way out at all: nothing arms a timer for that phase.

* **Host-gated.** The exit closes the results for everyone at the table, so a seat that does not
  hold the host role cannot take them away from the rest. Refused with *only the host can close
  the results and reopen the lobby*.
* **Fallback for a system-hosted lobby: any seated player.** A matchmade lobby has no player host
  at all (see "Host role": `HostUserID` is the system sentinel and `your_is_host` is false in
  every seat), so gating on the role would leave every seat of a finished match stuck on the
  results screen. Membership is what the fallback reads, not the socket: a connection whose user
  never joined the lobby is refused with the same message.
* **Idempotent.** Only the first exit transitions. A repeat is ignored on the phase check, and the
  reset the timer armed is dropped when it lands after its own results screen was closed, so it
  cannot cut short a later game's results (the hub is in `post_game` for those too).
* **Client-only.** The transition itself is internal to the hub, and a frame naming an internal
  message type (`_return_to_lobby` and the other underscore-prefixed types) is refused: those run
  phase transitions with no check on the sender, which would be a way around this gate.
* A phase that is not `post_game` or `match_end` ignores the message, sending nothing back. There
  is nothing to close, and a live game is not left this way (leaving is `POST /lobby/{id}/leave`).

The ranked half stops there. `RoundsPlayed`, `CumulativeScores` and `RoundHistory` are left where
the match left them, since what becomes of a finished ranked match's lobby is the unratified half
of the round-lifecycle design (cambia-466).

## Ruleset identity

`lobby_state.preset_id` names the ruleset the lobby carries, and the service records it rather
than leaving a client to work it out from `house_rules` (cambia-1123). Values cannot answer the
question: MATCHMAKING.md 5.2 fixes one ruleset for every ranked queue, so all six queue presets
in `GET /lobby/presets` hold byte-identical house rules and differ only in player and round
count. A client matching rules against that list names whichever preset is returned first, which
is how a lobby created from H2H Rapid came back reading H2H Quick.

* Set when a `presetId` is accepted, on `POST /lobby/create` or in an `update_rules` message,
  and only when the sheet that lands is still that preset: explicit `houseRules` in the same
  message land on top of the preset, and a sheet that departs from it in the call that named it
  is not on it.
* Cleared by the first later `update_rules` that moves a house rule or the auto-start setting
  without naming a preset. Circuit settings are not part of a preset, so changing them alone
  leaves the id where it is.
* A preset carries a lobby shape as well as a ruleset: one that fixes a player count fixes
  `game_mode` too, in `update_rules` as on create, so a lobby never records a 4-player ruleset
  while still calling itself `head_to_head`. A preset that seats fewer players than the lobby
  already has is refused and nothing is written, not even the `houseRules` alongside it: the seats
  are taken, and the service does not empty them to fit a ruleset (cambia-1099).
* A queue-backed lobby carries its queue's preset from creation, and takes it again at match
  formation. Its `house_rules` are the queue's from that moment, which is what makes the
  read-only rule sheet a matchmade lobby shows the ruleset its game is actually built from.
* Empty means the sheet is nobody's preset. A client may fall back to matching values then, but
  the match has to include the game mode, or a 4-player preset names a 2-player lobby's rules.

`lobby_state.rules_locked` sits beside `preset_id` and answers the other half: whether that
ruleset can still move. It is the same expression `update_rules` refuses on
(`Lobby.RulesLockedUnsafe`: a `matchmaking` lobby, or any lobby whose mode is `ranked`), computed
server-side so the refusal and the sheet a client renders cannot drift apart. Sent because no
client can derive it from this payload: a public or private lobby that queued its party into a
ranked queue is locked by its mode alone (cambia-966), and the snapshot carries no mode, so the
rule sheet used to hand that lobby's host controls every Save would refuse (cambia-1099). Always
present, on `sync_state` as on `lobby_state`, since both are built from the one snapshot.

## Server → Client Events

These messages are typically broadcast to all users in the lobby unless specified otherwise.

| Event Description             | `type` String             | Payload Example / Key Fields                                                                                                                                                                | Emitter Location           | Notes                                                                                             |
| :---------------------------- | :------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------- | :------------------------------------------------------------------------------------------------ |
| User Joined / Left            | `lobby_update`            | `{ "user_join": "{uuid}", "is_host": bool, "lobby_status": { ... } }` OR `{ "user_left": "{uuid}", "lobby_status": { ... } }`                                                                | `internal/game/lobby.go`   | Sent when a user connects or disconnects. Includes updated `lobby_status`.                        |
| Full Lobby State (Private)    | `lobby_state`             | `{ "lobby_id", "host_id", "your_id", "your_is_host", "system_host", "lobby_type", "game_mode", "in_game", "game_id", "house_rules": {...}, "preset_id": "h2h_rapid", "rules_locked": false, "circuit": {...}, "settings": {...}, "lobby_status": { ... } }` | `internal/game/lobby.go`   | Sent privately to a user upon joining/connecting, and rebroadcast to everyone when the roster or the host role changes. `system_host` marks a lobby the queue runs (see "Host role"). `preset_id` names the ruleset `house_rules` came from, always present and empty for a sheet that is nobody's preset, and `rules_locked` says whether it can still change (see "Ruleset identity"). |
| User Ready State Change       | `ready_update`            | `{ "user_id": "{uuid}", "is_ready": bool }`                                                                                                                                                  | `internal/game/lobby.go`   | Sent when a user's ready state changes.                                                           |
| User Invited                  | `lobby_invite`            | `{ "invitedID": "{uuid}" }`                                                                                                                                                                  | `internal/game/lobby.go`   | Sent when a user is invited via the `invite` command.                                             |
| Countdown Started             | `lobby_countdown_start`   | `{ "seconds": int }`                                                                                                                                                                        | `internal/game/lobby.go`   | Sent when the auto-start countdown begins.                                                        |
| Countdown Canceled            | `lobby_countdown_cancel`  | *(None)* | `internal/game/lobby.go`   | Sent if the countdown is stopped (e.g., user leaves or becomes unready).                          |
| Rules Updated                 | `lobby_state`             | The `lobby_state` payload above, carrying the new `house_rules`, `circuit`, `settings`, `preset_id` and `game_mode`                                                                          | `internal/hub/hub.go`      | An accepted `update_rules` rebroadcasts the full snapshot to every connected seat. Everyone at the table plays by these rules, so everyone is told: before cambia-1099 only the host who sent the edit knew, and every other seat rendered the old rule sheet until it reloaded. There is no separate rules event: one snapshot shape means a client renders the sheet the same way however it moved. |
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

**`HouseRules` Object Structure (within `lobby_state`, used by `update_rules`):**
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

**`Circuit` Object Structure (within `lobby_state`, used by `update_rules`):**
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
