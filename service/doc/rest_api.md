# Cambia Service API (docs/api.md)

This document outlines the available HTTP REST endpoints and WebSocket connections for the Cambia game service.

## Authentication

Most endpoints require authentication via an `auth_token` JWT cookie sent in the `Cookie` header.

* **Obtaining a Token:** Use `POST /user/login`. The token is returned in the response body and set as an `HttpOnly` cookie. The default expiration time is configurable via the `TOKEN_EXPIRE_TIME` environment variable (e.g., "72h", "0" or "never" for no expiration).
* **Ephemeral Guests:** Connecting to a WebSocket endpoint (`/lobby/ws/*` or `/game/ws/*`) *without* a valid `auth_token` cookie will automatically create a temporary guest user, set the `auth_token` cookie, and return the user's ephemeral ID.
* **Claiming Guests:** Guests can call `POST /user/claim` (`ClaimEphemeralHandler`) to convert an ephemeral guest account into a persistent one by adding email/username/password.
* **Token Verification:** The server uses an Ed25519 key pair (generated at runtime by default) to sign and verify JWTs.
* **Duplicate cookies:** A request can carry more than one `auth_token` cookie (localhost cookies are shared across ports/apps on a multi-app host, and a dev-server restart that rotates the signing key can leave a stale cookie in the jar alongside a fresh one). All cookie-based auth in this service - `middleware.RequireAuth` (gating `/training/*` and `/ws/training/*`) and the REST handlers that authenticate directly off the `Cookie` header (`GET /user/me`, `POST /user/claim`, `EnsureEphemeralUser` guest bootstrap, `/friend/*`) - go through the shared `auth.ResolveAuthTokenCookie` helper. It checks every `auth_token` cookie on the request and accepts the first one that verifies, rather than only the first cookie in the header. Any invalid `auth_token` cookie seen along the way gets an expiring `Set-Cookie` in the response so the browser drops it instead of resending it on every request.

## HTTP REST Endpoints

These endpoints handle user management, friends, and lobby setup. They require the `auth_token` cookie unless otherwise specified.

*(Note: All handlers are registered in `cmd/server/main.go`.)*

---

### User Endpoints

Handled by `internal/handlers/user.go`.

#### `POST /user/create`

* **Description:** Creates a new persistent user account. Returns an error if the email already exists. Cannot be used to claim an existing ephemeral user.
* **Authentication:** None required.
* **Request Body:** `application/json`
    ```json
    {
      "email": "user@example.com", // string, required, must be unique
      "password": "securepassword", // string, required
      "username": "preferred_username" // string, required
    }
    ```
* **Response (Success: 201 Created):** `application/json`
    ```json
    {
      "id": "...",         // string (UUID)
      "email": "...",      // string
      // Password omitted
      "username": "...",   // string
      "is_ephemeral": false, // boolean
      "is_admin": false,   // boolean
      "elo_1v1": 1500,     // integer
      "elo_4p": 1500,      // integer
      "elo_7p8p": 1500,    // integer
      "phi_1v1": 350.0,    // float64
      "sigma_1v1": 0.06    // float64
    }
    ```
* **Response (Error):**
    * `400 Bad Request`: Invalid payload.
    * `409 Conflict`: Email already exists.
    * `500 Internal Server Error`: Database error.

#### `POST /user/login`

* **Description:** Authenticates a user with email and password. Returns a JWT token in the body and sets the `auth_token` cookie.
* **Authentication:** None required.
* **Request Body:** `application/json`
    ```json
    {
      "email": "user@example.com", // string, required
      "password": "securepassword" // string, required
    }
    ```
* **Response (Success: 200 OK):** `application/json`
    * **Headers:** `Set-Cookie: auth_token={jwt}; Path=/; HttpOnly; Max-Age={seconds}`
    ```json
    {
      "token": "{jwt}" // string
    }
    ```
* **Response (Error):**
    * `400 Bad Request`: Invalid payload.
    * `403 Forbidden`: Authentication failed (wrong email/password, or user not found).
    * `500 Internal Server Error`: Failed to create JWT or write response.

#### `GET /user/me`

* **Description:** Retrieves the authenticated user's basic information (non-sensitive fields).
* **Authentication:** `auth_token` cookie required.
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json`
    ```json
    {
      "id": "...",         // string (UUID)
      "username": "...",   // string
      "is_ephemeral": bool, // boolean
      "is_admin": bool    // boolean
      // Other non-sensitive fields like Elo might be added here
    }
    ```
* **Response (Error):**
    * `403 Forbidden`: Invalid or missing token.
    * `404 Not Found`: User ID from token not found in database.
    * `500 Internal Server Error`: Failed to write response.

---

### Friends Endpoints

Handled by `internal/handlers/friend.go`. Require `auth_token` cookie.

#### `POST /friends/add`

* **Description:** Sends a friend request from the authenticated user to the user specified in the payload. Creates a `friends` record with `status='pending'`.
* **Request Body:** `application/json`
    ```json
    {
      "friend_id": "{uuid}" // string (UUID), required - ID of the user to send request to
    }
    ```
* **Response (Success: 201 Created):** `text/plain` - "friend request sent"
* **Response (Error):** `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`.

#### `POST /friends/accept`

* **Description:** Accepts a pending friend request *sent by* the user specified in the payload *to* the authenticated user. Updates the `friends` record status to `'accepted'`.
* **Request Body:** `application/json`
    ```json
    {
      "friend_id": "{uuid}" // string (UUID), required - ID of the user whose request is being accepted
    }
    ```
* **Response (Success: 200 OK):** `text/plain` - "friend request accepted"
* **Response (Error):** `400 Bad Request` (e.g., no pending request found), `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`.

#### `GET /friends/list`

* **Description:** Returns a list of all friend relationships (pending or accepted) involving the authenticated user.
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json`
    ```json
    [
      {
        "user1_id": "{uuid}", // string (UUID)
        "user2_id": "{uuid}", // string (UUID)
        "status": "pending" | "accepted" // string
      }
      // ... more relationships
    ]
    ```
* **Response (Error):** `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`.

#### `POST /friends/remove`

* **Description:** Deletes the friend relationship between the authenticated user and the user specified in the payload.
* **Request Body:** `application/json`
    ```json
    {
      "friend_id": "{uuid}" // string (UUID), required - ID of the user to unfriend
    }
    ```
* **Response (Success: 200 OK):** `text/plain` - "friend removed"
* **Response (Error):** `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`.

---

### Lobby Endpoints

Handled by `internal/handlers/lobby.go`. These manage *ephemeral* in-memory lobbies. Every endpoint below requires the `auth_token` cookie except `GET /lobby/list`, which carries no identity requirement (see that entry).

#### `POST /lobby/create`

* **Description:** Creates a new ephemeral game lobby in memory, hosted by the authenticated user. Lobby is automatically deleted when the last user leaves.
* **Request Body:** `application/json` (Optional - defaults apply if omitted)
    ```json
    {
      "type": "private" | "public" | "matchmaking", // string, optional (default: "private")
      "gameMode": "head_to_head" | "group_of_4" | ..., // string, optional (default: "head_to_head")
      "queueID": "h2h_quickplay", // string, required for type "matchmaking", optional otherwise
      // Partial houseRules, circuit, or lobbySettings objects can be included
      "houseRules": { "turnTimerSec": 30 }, // optional
      "lobbySettings": { "autoStart": false } // optional
    }
    ```
    **Matchmaking lobbies** are defined by their queue, not by a game mode: send
    `{"type":"matchmaking","queueID":"<queue id>"}` and the handler derives the rest from that
    queue's config (`internal/matchmaking/validation.go`, the same one `POST /lobby/{id}/search`
    reads later, so the two never disagree):
    * `gameMode`: `head_to_head` for a 2-player queue, `group_of_4` for a 4-player one. A
      multi-round queue keeps the player-count mode until the round lifecycle lands (cambia-466).
    * `mode`: `ranked` for a ranked queue, otherwise `casual`.
    * `queueID`: echoed back, and it is what the search endpoint queues the lobby into. Round
      count and ranked-ness are not stored a second time on the lobby; the hub reads them from
      the queue config at search time.

    The host is a joined member of a matchmaking lobby from creation (a party of one), so
    `POST /lobby/{id}/search` succeeds without a WebSocket connection in between. That host role
    lasts until the matchmaker seats a match in the lobby, and no further: from match formation
    on the lobby is system-hosted and no player holds host powers over it (see
    `lobby_actions.md`, "Host role"). `hostUserID` in this response is therefore the creator, and
    stays the creator in the persisted `lobbies` row even after the handover.

    A `queueID` on a `public` or `private` lobby is accepted and validated: the search endpoint
    gates on host, `searching` and `queueID` alone, so a standing lobby can queue its party
    without being typed `matchmaking`.

    **A queue sets its own rules.** Whenever a create request resolves to a configured queue - by
    `type: "matchmaking"`, by a `queueID` on any lobby type, or through the transitional
    `gameMode` shape below - a rule-carrying key in the body is refused with a 400 rather than
    applied. The queue config is the whole rule set for the matches it forms, and it is what the
    matchmaker paired the players on; the WebSocket rules lock alone left `POST /lobby/create` as
    a way around it, so a hand-written body could seat a rated match of a public queue on rules
    of the caller's choosing (cambia-1089). The refused keys are `houseRules`, `circuit`,
    `settings` and `lobbySettings`. A caller who wants their own rules creates a lobby with no
    `queueID`.

    Transitional shape (remove after 2026-10-01): `{"type":"matchmaking","gameMode":"<queue id>"}`
    with no `queueID`, which is what web bundles cached from before cambia-933 send, is read as
    that queue id and logged as deprecated.
* **Matchmaking 400 bodies:**
    * `Matchmaking lobby requires queueID` - type `matchmaking` with no `queueID` (and no queue id
      in `gameMode`). There is no default queue.
    * `Unknown matchmaking queue: <id>` - the `queueID` (on any lobby type) names no configured queue.
    * `Matchmaking queue <id> has an unsupported player count: <n>` - the queue config asks for a
      player count no game mode covers.
    * `A matchmaking queue sets its own rules: remove <key> or create a lobby without a queueID` -
      the body carried `houseRules`, `circuit`, `settings` or `lobbySettings` alongside a queue id.
* **Response (Success: 200 OK):** `application/json` - Returns the full state of the created lobby.
    ```json
    {
        "id": "{uuid}",
        "hostUserID": "{uuid}",
        "type": "private",
        "gameMode": "head_to_head",
        "name": "",
        "gameId": "00000000-0000-0000-0000-000000000000",
        "inGame": false,
        "createdAt": "{RFC3339 timestamp}",
        "houseRules": { ... }, // Full HouseRules object
        "circuit": { ... }, // Full Circuit object
        "lobbySettings": { ... }, // Full LobbySettings object
        "mode": "casual",
        "searching": false
    }
    ```
    `name` is the host-supplied display name (empty string when omitted from the request), and
    `mode`/`searching` always serialize. `gameId` carries no `omitempty` tag: `GameID` is a
    `uuid.UUID` (a fixed-size byte array), and Go's `encoding/json` only treats a pointer, slice,
    map, or string as "empty" for that tag, never a fixed-size array - the tag would have been a
    no-op, so a lobby with no game yet always reports the nil UUID rather than omitting the key.
    `queueID` is the one field that genuinely omits: it is a plain string, empty until a queue is
    selected.
* **Response (Error):** `400 Bad Request` (invalid type/mode/payload), `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`.

#### `POST /lobby/{id}/search` and `DELETE /lobby/{id}/search`

* **Description:** Puts the lobby's party into its `queueID` queue, or takes it back out. Host
    only, and a lobby that has already held a match has no player host to satisfy that: both
    verbs answer `403` on a system-hosted lobby, since a formed match is the queue's and not a
    party's to requeue (`lobby_actions.md`, "Host role").
* **Response (Success: 200 OK):** `{"status":"searching","queue_id":"<id>"}` for the POST,
    `{"status":"cancelled"}` for the DELETE.
* **Response (Error):** `400 Bad Request` (`No queue selected for this lobby`, `Unknown queue ID`,
    or a matchmaker rejection such as an empty party), `403 Forbidden` (not the host),
    `404 Not Found`, `409 Conflict` (already searching).
* **Who gets matched:** a queued party is only paired while at least one of its members holds a
    WebSocket connection. A party whose members all closed their tabs keeps its place in the queue,
    since membership survives a dropped socket by design and a page refresh must not cost a place
    in line, but it is passed over until somebody reconnects and is released by the lobby's idle
    window if nobody does. Pairing such a party would drop the connected side into a ready check
    the absent side can never answer (cambia-933). A match that turns out to be short when it is
    consolidated is abandoned: no client is told anything, and the still-connected parties go back
    in the queue with their original queue time, so their search simply continues.

#### `GET /lobby/list`

* **Description:** Lists public lobbies a caller can currently join. Requires no authentication: the handler reads no identity, so the list is the same for every caller. Three filters apply beyond simple membership:
    * **Type:** only lobbies of type `"public"` appear. A private lobby's id, host id, host-typed name and house rules are never included here regardless of who asks; its own members reach it through `GET /lobby/active` instead.
    * **Presence:** a lobby with no game in progress and no live WebSocket connection is left out entirely, unless it is still within `lobbyListingCreationGrace` (30 seconds) of its own creation - the gap between `POST /lobby/create` and the host's first WebSocket upgrade, plus room for a brief reconnect blip.
    * **In-game exemption:** a lobby with a game in progress stays listed even with nobody currently connected, mirroring the idle reaper's own exemption.
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json` - Returns a map where keys are lobby UUIDs and values wrap the lobby object with player counts.
    ```json
    {
      "{lobby_uuid_1}": {
        "lobby": {
          "id": "{uuid}",
          "hostUserID": "{uuid}",
          "type": "public",
          "gameMode": "head_to_head",
          "name": "Friday Night Cambia",
          "gameId": "00000000-0000-0000-0000-000000000000",
          "inGame": false,
          "createdAt": "{RFC3339 timestamp}",
          "houseRules": { ... },
          "circuit": { ... },
          "lobbySettings": { ... },
          "mode": "casual",
          "searching": false
        },
        "playerCount": 1,
        "maxPlayers": 2
      }
    }
    ```
    Same fields as the `POST /lobby/create` response above, since `ListLobbiesResponse.Lobby` is
    the same struct; see that entry's note on `name`, `gameId`, and `queueID`.
* **Response (Error):** `500 Internal Server Error`.

#### `GET /lobby/active`

* **Description:** Returns the single lobby or in-progress game the authenticated caller should be offered to resume - the home screen's answer to "what was I doing" after a refresh or a lost tab. Scans every lobby the caller has joined whose hub is still alive (a hub whose last connection left has dissolved its `Run` loop even though it stays registered, so it is excluded rather than offered as a dead resume target) and returns the highest-ranked candidate: a game in progress outranks a matchmaking search, which outranks an idle open lobby; ties break on the lower lobby UUID for a stable answer across calls. A private lobby is included here - `GET /lobby/list` excludes it from the public list, but a caller resuming their own membership is not the "uninvited caller" that filter guards against.
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json` - `active` is `null` when the caller belongs to no live lobby, otherwise the resumable session. The bodies below model two players seated in a `head_to_head` game; their shape is asserted against the live handler by `internal/handlers/active_session_test.go`'s `TestActiveSessionDocSample`, which fails if the field set here drifts from what the endpoint emits. The uuids are illustrative.
    ```json
    {
      "active": {
        "lobbyId": "c1f77934-17a4-473f-a324-2925546b4e1c",
        "lobbyType": "public",
        "gameMode": "head_to_head",
        "phase": "in_game",
        "gameId": "aabb1965-6ef7-400e-8fff-2d16b430d137",
        "seated": true,
        "playerCount": 2
      }
    }
    ```
    The empty case, for a token with no lobby membership (asserted byte for byte by the same test):
    ```json
    { "active": null }
    ```
    `name` (the lobby's display name) and `gameId` both carry `omitempty` here - unlike `Lobby.GameID` above, this `GameID` is a plain `string`, empty until a live game is attached, so the tag actually omits the key rather than serializing a nil UUID. `phase` is one of `"open"`, `"searching"`, or `"in_game"`, derived from lobby/game state rather than read off the hub's own `Phase` field (which mutates only inside the hub's `Run` goroutine and would race here). `seated` is `false` for a lobby member who holds no seat in a game already dealt (joined after the deal); `playerCount` is the seated count while in game, otherwise the joined lobby member count.
* **Response (Error):** `401 Unauthorized` (no `auth_token` cookie present) - body `Missing authentication token`, also asserted by `TestActiveSessionDocSample`. `403 Forbidden` (cookie present but invalid, expired, or carrying an unparseable user id). `405 Method Not Allowed` for anything but `GET`.

---

### Matchmaking Endpoints

Handled by `internal/handlers/lobby.go` (`ListQueuesHandler`), registered at `/matchmaking/queues` in `cmd/server/main.go`.

#### `GET /matchmaking/queues`

* **Description:** Lists all configured matchmaking queues with live stats. Requires no authentication: the handler reads no identity, so the list is the same for every caller.
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json` - an array with one entry per queue in `matchmaking.QueueConfigs` (`internal/matchmaking/validation.go`):
    ```json
    [
      {
        "queueId": "h2h_quickplay",
        "name": "H2H Quick",
        "players": 2,
        "rounds": 1,
        "ratingPool": "h2h_qp",
        "ranked": true,
        "hiddenRating": true,
        "playerCount": 0,
        "avgWaitSec": 0
      }
    ]
    ```
    `playerCount` and `avgWaitSec` come from the matchmaker's live in-memory queue state, not the queue config.
    The array order is fixed by each queue's `QueueConfig.Order` (ties broken by `queueId`), currently `h2h_quickplay`,
    `h2h_blitz`, `h2h_rapid`, `h2h_classical`, `ffa4_standard`, `ffa4_classical` - not by ranging over the config map
    directly, since Go re-randomizes map iteration order on every range statement and the response previously
    reordered itself between calls with nothing actually changed (cambia-957).
* **Response (Error):** `405 Method Not Allowed` for anything but `GET`.

---

### Game Endpoints (Legacy/Debug)

Handled by `internal/handlers/game.go`.

#### `POST /game/create`

* **Description:** (Debug/Legacy) Creates a game instance directly in memory without going through a lobby. Does not handle player joining or auth via HTTP. Use lobby flow instead.
* **Authentication:** None (intended for debug).
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json`
    ```json
    {
      "game_id": "{uuid}"
    }
    ```
* **Response (Error):** `500 Internal Server Error`.

#### `GET /game/reconnect/{game_id}`

* **Description:** (Deprecated) Acknowledges a reconnect attempt via HTTP but cannot fully re-establish WebSocket state. Use the WebSocket endpoint `/game/ws/{game_id}` for actual reconnection.
* **Authentication:** Requires `auth_token` cookie (logic commented out in handler but intended).
* **Response (Success: 200 OK):** `text/plain` - "Reconnect acknowledged via HTTP. Please establish a WebSocket connection..."
* **Response (Error):** `400 Bad Request` (invalid game_id), `403 Forbidden` (invalid token), `404 Not Found` (game not found).

## WebSocket Endpoints

These endpoints handle real-time communication for lobbies and active games. They require the `auth_token` cookie (or trigger guest creation) and specific subprotocols.

*(Note: WebSocket handlers upgrade HTTP connections initiated at these paths. See `cmd/server/main.go` for registration.)*

| Path                    | Subprotocol | Description                                                               | Handler Location               | Payload Details Reference     |
| :---------------------- | :---------- | :------------------------------------------------------------------------ | :----------------------------- | :-------------------------- |
| `/lobby/ws/{lobby_id}`  | `lobby`     | Handles joining, leaving, chat, readiness, and game start orchestration.  | `internal/handlers/lobby_ws.go` | `docs/lobby_actions.md` |
| `/game/ws/{game_id}`    | `game`      | Handles all in-game actions (drawing, discarding, special abilities, etc.). | `internal/handlers/game_ws.go` | `docs/game_actions.md`  |

---

*(Refer to `docs/lobby_actions.md` and `docs/game_actions.md` for detailed WebSocket message payloads.)*
