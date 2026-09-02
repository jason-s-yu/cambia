# Lobby WebSocket Actions (docs/lobby_actions.md)

This document describes the JSON payloads used for WebSocket communication on the unified `/ws/{lobby_id}` endpoint (see `rest_api.md`, "WebSocket Endpoints"), using the `cambia` subprotocol, while the hub is in a lobby phase (`open`, `ready_check`, `countdown`, or - for `chat` and `return_to_lobby` - `post_game`/`match_end`). The same socket carries the connection into and out of an in-game phase; see `game_actions.md` for what it accepts there.

**Conventions:**

* Every payload is a JSON object with a mandatory top-level `type` key (string).
* UUIDs are strings (e.g., `"f47ac10b-58cc-4372-a567-0e02b2c3d479"`).
* Timestamps (`ts`) are UNIX seconds (integer).
* User identification uses `user_id` (string) or `userID` (string) in client->server messages, and `user_id` (string) or structured objects like `{"id": "{uuid}"}` in server->client messages, often nested under keys like `user_join` or `user_left`. Consistency varies slightly.

## Client → Server Commands

| Action                   | `type` String    | Payload                                                                                                | Handler Location                 | Notes                                                      |
| :----------------------- | :--------------- | :----------------------------------------------------------------------------------------------------- | :------------------------------- | :--------------------------------------------------------- |
| Mark Ready               | `ready`          | *(None)* | `internal/hub/hub.go`  | Marks sender as ready. May trigger countdown if autoStart. |
| Mark Unready             | `unready`        | *(None)* | `internal/hub/hub.go`  | Marks sender as unready. Cancels any active countdown.     |
| Invite User              | `invite`         | `{ "userID": "{uuid}" }`                                                                               | `internal/hub/hub.go`  | Invites another user to a private lobby. Silent on success: no event is sent back to the inviter and nothing is broadcast, since an invite is not membership and does not appear in `lobby_status` until the invited user joins. |
| Leave Lobby              | *(not a WS message)* | *(None)* | `internal/handlers/lobby.go` | `POST /lobby/{id}/leave`, optional body `{ "forfeit": true }`. Leaving releases membership, which nothing a lost socket can also trigger may do, so it is an HTTP call rather than a frame (cambia-807). Mid-game the answer turns on the caller's seat, not on the lobby's `inGame` flag: a caller holding no live seat (never dealt in, already forfeited, or the game has finished) leaves normally, a live seat is refused with 409, and a live seat that sent `forfeit: true` is forfeited on the spot and then leaves (cambia-1520). The flag is an opt-in a client sends only after the player has agreed to give the seat up; without it the seat sat on the table until `disconnectGraceSec` ran out. |
| Send Chat Message        | `chat`           | `{ "msg": "Your message here" }`                                                                       | `internal/hub/hub.go`  | Sends a chat message to the lobby.                         |
| Update Rules (Host Only) | `update_rules`   | `{ "rules": { "presetId": "h2h_rapid", ... partial HouseRules object ... } }` (See `internal/game/rules.go` for fields)       | `internal/hub/hub.go`  | Host updates lobby's house rules or circuit settings. Refused outright for a ranked or matchmaking lobby: see "Host role". An optional `presetId` inside `rules` names a whole ruleset from `GET /lobby/presets` and is expanded server-side before the field-by-field keys, so a `houseRules` object in the same message lands on top of it; an unknown id rejects the whole message (cambia-1088). The id is recorded on the lobby and echoed back as `lobby_state.preset_id`; sending one with the expanded sheet is how a client keeps the lobby's ruleset named (see "Ruleset identity"). A preset that fixes a player count fixes `game_mode` with it, and one that seats fewer players than the lobby already has is refused, changing nothing. The auto-start block is read from `settings`, with `lobbySettings` accepted as an alias (see `rest_api.md`). An accepted edit broadcasts `lobby_state` to every connected seat; a refused one broadcasts nothing. |
| Force Start (Host Only)  | `start_game`     | *(None)* | `internal/hub/hub.go`  | Host attempts to start the game manually (if all ready). Refused for a system-hosted lobby, which starts on its ready check. |
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

Every row below names an event `internal/hub/hub.go` emits at HEAD, traced to its `Emit`
(broadcast to all connections) or `EmitTo`/`emitToWithSeq` (one connection) call site. A design
that predated the hub unification sent finer-grained events for a joined/left user, a ready-state
flip, an invite, and a countdown starting or being called off; none of those are emitted any more.
A roster, ready-state, host-role or rule change is now conveyed by a fresh full `lobby_state`
snapshot, and a countdown starting or ending is a `phase_change` like any other phase transition -
see those two rows.

| Event Description                | `type` String  | Payload Example / Key Fields                                                                                                                                                                | Emitter Location                                              | Notes                                                                                             |
| :-------------------------------- | :------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------- | :------------------------------------------------------------------------------------------------ |
| Full Lobby State (Private + Broadcast) | `lobby_state` | `{ "lobby_id", "host_id", "your_id", "your_is_host", "system_host", "phase", "lobby_type", "game_mode", "in_game", "game_id", "house_rules": {...}, "preset_id": "h2h_rapid", "rules_locked": false, "circuit": {...}, "settings": {...}, "lobby_status": { ... }, "match_state": {...} }` | `internal/hub/hub.go` (`buildLobbySnapshot`, sent by `sendLobbyState`/`broadcastLobbyUpdate`) | Sent privately to a user on connect (`sendLobbyState`), then rebroadcast to everyone (`broadcastLobbyUpdate`) on the same connect, on a leave/disconnect, on `ready`/`unready`, on an accepted `update_rules`, on match formation for the hosting lobby, and on a closed results screen. `system_host` marks a lobby the queue runs (see "Host role"). `preset_id`/`rules_locked` are covered in "Ruleset identity". `match_state` (`queue_id`, `is_ranked`, `total_rounds`, `current_round`, `cumulative_scores`, `round_history`, `dealer_seat`) is present only for a ranked lobby past its first round (`IsRanked && TotalRounds > 1`). There is no incremental event for any of the above; a client diffs this snapshot against its own state. |
| Desync Recovery Snapshot (Private) | `sync_state`  | The `lobby_state` payload above, plus `"seq": <uint64>`                                                                                                                                     | `internal/hub/hub.go` (`sendSyncState`)                        | Sent instead of applying an inbound frame whose `last_seq` trails the hub's own `seq` (`dispatch`'s staleness gate); the frame itself is dropped. `seq` lets the client catch its own counter up before retrying. |
| Chat Message Received             | `chat`         | `{ "userID": "{uuid}", "username": "...", "msg": "The message" }`                                                                                                                          | `internal/hub/hub.go` (`handleLobbyMsg`, `chat` case)          | Broadcast to everyone, including the sender. No `ts` field; a client stamps one on receipt if it wants one. Reachable from every lobby phase, `searching`, and `post_game`/`match_end` - each forwards into this one case. |
| Phase Changed                     | `phase_change` | `{ "phase": "open" \| "ready_check" \| "countdown" \| "searching" \| "in_game" \| "round_end" \| "post_game" \| "match_end" }`, plus `"seconds": <int>` only when `phase` is `"countdown"` | `internal/hub/hub.go` (every phase transition; canonical set is `LobbyPhase.String()`) | Sent on every phase transition, including a countdown starting (`countdown`, with `seconds`) and being called off (`open`) - the two events a pre-unification design sent separately. A roster or rule change that does not cross a phase boundary sends no `phase_change`, only a fresh `lobby_state`. |
| Search Status                     | `search_status` | `{ "searching": bool }`, plus `"queue_id": "<id>"` only when entering                                                                                                                     | `internal/hub/hub.go` (`applySearchState`, and `cancel_search` in `handleSearchingMsg`) | Sent alongside a `phase_change` to/from `searching` on the same transition. |
| Match Found                       | `match_found`  | `{ "lobby_id": "{uuid}", "queue_id": "<id>", "total_rounds": int, "is_ranked": bool, "players": [ { "UserID": "{uuid}", "Username": "...", "IsHost": bool } ] }`                          | `internal/hub/hub.go` (`handleMatchFound`)                     | Sent to every hub in the group the instant the matchmaker forms a match; `lobby_id` is where it is played, so a non-hosting party's clients use it to navigate there. `players` entries serialize their Go struct field names verbatim (`MatchedPlayer` carries no JSON tags) - the one departure from this API's `camelCase`/`snake_case` convention. Followed by a `phase_change` (`ready_check` for the hosting lobby, `open` for the rest) and, for the hosting lobby, a `lobby_state` rebroadcast, since its host role has just moved to the system (cambia-1087). |
| Game Started                      | `game_started` | `{ "game_id": "{uuid}", "players": ["{uuid}", ...] }`                                                                                                                                       | `internal/hub/hub.go` (`createAndStartGame`)                   | Sent to every connected participant when the game instance is created, before the pre-game reveal begins. `players` is host-first, then the rest in a stable order. No new connection is needed: the same socket that was in the lobby continues into the game (see `rest_api.md`, "WebSocket Endpoints"). |
| Round Started                     | `round_start`  | `{ "round": int, "total_rounds": int, "dealer_seat": int }`                                                                                                                                | `internal/hub/hub.go` (`startNextRound`)                       | Sent when a ranked circuit's next round begins, preceded by a `phase_change` to `in_game`. `round` is the 1-based round about to start. |
| Round Ended                       | `round_end`    | `{ "round": int, "total_rounds": int, "round_scores": {...}, "cumulative_scores": {...}, "subsidies": {...}, "finalHands": [...] }`                                                       | `internal/hub/hub.go` (`HandleRoundEnd`, non-final branch)     | Sent when a ranked round ends with rounds remaining, preceded by a `phase_change` to `round_end`. `round_scores`/`cumulative_scores` are keyed by user id; `subsidies` is the aggression-subsidy award, keyed by user id string. `finalHands` is the same reveal `game_actions.md`'s "Round-end reveal" section documents for `game_end`/`game_results`. The next round auto-starts after 10 seconds. |
| Match Ended                       | `match_end`    | `{ "round_scores": {...}, "cumulative_scores": {...}, "round_history": [...], "subsidies": {...}, "finalHands": [...], "final": true }`                                                   | `internal/hub/hub.go` (`HandleRoundEnd`, final branch)         | Sent when a ranked round ends and it was the circuit's last, preceded by a `phase_change` to `match_end`. `round_history` is every round's `round_scores`, oldest first. See "Post-game exit" for how a table leaves this screen. |
| Action Refused (Private)          | `error`        | `{ "error": "<message>" }`                                                                                                                                                                  | `internal/hub/hub.go` (`errEnvelope`, sent via `conn.SendEnvelope` by the refusing handler) | Sent only to the connection whose frame was rejected, never broadcast: a validation failure or host-gate refusal on `invite`, `update_rules`, `start_game`, `cancel_search`, or `return_to_lobby`. The message text is the whole explanation a client shows. |
| Hub Fatal Error (Broadcast)       | `error`        | `{ "code": "hub_fatal", "message": "This game hit an internal error and has ended. Other games are unaffected.", "fatal": true }`                                                        | `internal/hub/hub.go` (`reportFatalPanic`)                     | Broadcast to every connection just before the hub dissolves, on recovery from a panic inside the hub's event loop. Distinguished from the private refusal above by shape: this one carries `code`/`fatal`, the other only `error`. |

**`lobby_status` Object Structure (within `lobby_state` and `sync_state`):**

```json
{
  "users": [
    {
      "id": "{uuid}",
      "is_host": bool,
      "is_ready": bool,
      "username": "..." // present only for a user currently holding a connection
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

A `numDecks` the host never sends (the lobby is still carrying `DefaultHouseRules`'s value)
resolves at game creation from the seated player count instead, per MATCHMAKING.md 1.1: one
deck for 2-4 seated players, two for 5-8. The rule sheet itself keeps reporting whatever it last
held (1 by default), since the resolved count is a property of the game that gets dealt, not a
value written back to the lobby; a game created from a 5-8 seat casual lobby that never touched
`numDecks` deals from two decks even though `lobby_state.house_rules.numDecks` still reads 1. An
explicit `numDecks` from the host, including a value equal to the default, is honored as sent at
every seat count and is never upgraded or downgraded; there is no separate warning for an
under-provisioned choice.

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
(See `internal/lobby/lobby.go` for field definitions)

```json
{
  "autoStart": bool
}
```

**Reconnect grace (`disconnectGraceSec`)**

`disconnectGraceSec` is how long a dropped socket keeps its seat before something acts on the drop
(default 90, range 0-3600, 0 acts on the drop itself). Under `forfeitOnDisconnect` that something
is the forfeit; in a circuit round, which is created with `forfeitOnDisconnect` off, it is the
seat's takeover by the turn clock instead (cambia-1233). Either way it is what makes a page reload
survivable: the seat is held, the table keeps playing, and a reconnect inside the window restores
the player through the usual `private_sync_state` (see `game_actions.md`, "Disconnect grace").
Matchmaking queues do not take this from the lobby: each queue carries its own value
(`internal/matchmaking/validation.go`), applied when the game is built, since a queued lobby has no
host setting rules.
