# Cambia Service API (docs/api.md)

This document outlines the available HTTP REST endpoints and WebSocket connections for the Cambia game service.

## Authentication

Most endpoints require authentication by a JWT, carried by an `auth_token` cookie or sent explicitly by the caller.

* **Carriers and precedence:** every authenticated path resolves the caller through `auth.ResolveAuthToken`, which reads, in order: (1) `Authorization: Bearer <jwt>` (the scheme is case-insensitive); (2) a `Sec-WebSocket-Protocol` entry prefixed `cambia-token.` (the WebSocket handshake, where a browser client cannot set headers); (3) the `auth_token` cookie(s). A caller that sends an explicit token has opted out of the cookie for that request: if that token does not verify the request fails, and the cookie is neither read nor expired. An `Authorization` header with another scheme, or an empty Bearer value, offers nothing and falls through to the cookie.
* **Tab-scoped sessions:** the request header `X-Cambia-Session: tab` asks for a session held by one browser tab instead of the origin's shared cookie jar. `POST /user/guest` then mints a fresh guest, returns its token in the body, and sets no cookie; `POST /user/login` returns its usual body and skips the `Set-Cookie`. The client keeps the token in `sessionStorage` and replays it on the carriers above. Accepting tokens is always on; only the minting endpoints below are dev-gated.
* **Obtaining a Token:** Use `POST /user/login`. The token is returned in the response body and set as an `HttpOnly` cookie. The default expiration time is configurable via the `TOKEN_EXPIRE_TIME` environment variable (e.g., "72h", "0" or "never" for no expiration).
* **Ephemeral Guests:** Connecting to a WebSocket endpoint (`/lobby/ws/*` or `/game/ws/*`) *without* a valid `auth_token` cookie will automatically create a temporary guest user, set the `auth_token` cookie, and return the user's ephemeral ID.
* **Claiming Guests:** Guests can call `POST /user/claim` (`ClaimEphemeralHandler`) to convert an ephemeral guest account into a persistent one by adding email/username/password.
* **Token Verification:** The server uses an Ed25519 key pair (generated at runtime by default) to sign and verify JWTs.
* **Duplicate cookies:** A request can carry more than one `auth_token` cookie (localhost cookies are shared across ports/apps on a multi-app host, and a dev-server restart that rotates the signing key can leave a stale cookie in the jar alongside a fresh one). All auth in this service - `middleware.RequireAuth` (gating `/training/*` and `/ws/training/*`) and the REST handlers that authenticate directly off the request (`GET /user/me`, `POST /user/claim`, `GET /user/history`, `GET /user/ratings`, `GET /leaderboard`, `EnsureEphemeralUser` guest bootstrap, `/friend/*`) - goes through the shared `auth.ResolveAuthToken` helper, whose cookie step is `auth.ResolveAuthTokenCookie`. It checks every `auth_token` cookie on the request and accepts the first one that verifies, rather than only the first cookie in the header. Any invalid `auth_token` cookie seen along the way gets an expiring `Set-Cookie` in the response so the browser drops it instead of resending it on every request. An invalid explicit token never triggers that expiry: one tab's stale token must not clear the jar the other tabs are using.

## HTTP REST Endpoints

These endpoints handle user management, friends, and lobby setup. They require the `auth_token` cookie unless otherwise specified.

*(Note: All handlers are registered in `cmd/server/main.go`.)*

---

### Health Check

#### `GET /healthz`

* **Description:** Liveness/readiness probe. Reports service status plus whether the database and Redis are currently reachable (each pinged with a 2s timeout; either can be `false` without failing the request - Redis is non-fatal to the service, and the probe still answers `200` while a dependency is down).
* **Authentication:** None required.
* **Request Body:** None. `HEAD` is also accepted.
* **Response (Success: 200 OK):** `application/json`
    ```json
    {
      "status": "ok",
      "db": true,
      "redis": true
    }
    ```
* **Response (Error):** `405 Method Not Allowed` for anything but `GET`/`HEAD`.

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
* **Tab mode:** with `X-Cambia-Session: tab` the response body is unchanged and no `Set-Cookie` is written, so the caller pins the returned token to one tab and the shared cookie identity is left as it was.

#### `POST /user/logout`

* **Description:** Clears the `auth_token` cookie (`Max-Age=-1`). Unconditional: it expires whatever cookie is present without verifying it, and does not read a request body or check method.
* **Authentication:** None required.
* **Request Body:** None.
* **Response (Success: 200 OK):** No body.

#### `POST /user/guest`

* **Description:** Provisions an ephemeral guest session without a WebSocket. Without the tab header it returns the caller's existing identity when a valid credential is present, and otherwise creates a guest and sets the `auth_token` cookie.
* **Authentication:** None required.
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json`
    * **Headers:** `Set-Cookie: auth_token={jwt}; ...`
    ```json
    {
      "id": "..." // string (UUID)
    }
    ```
* **Tab mode:** with `X-Cambia-Session: tab` a *fresh* guest is always minted (never the cookie's identity: the caller asked for an identity only this tab holds), no cookie is set, and the body carries the token:
    ```json
    {
      "id": "...",    // string (UUID)
      "token": "{jwt}" // string
    }
    ```
* **Response (Error):** `500 Internal Server Error`: Failed to create the guest user.

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

#### `POST /user/claim`

* **Description:** Converts the caller's ephemeral guest account into a persistent one by attaching email, password, and (optionally) a new username. The account already exists; this updates it in place rather than creating a new user, so the caller's id and history carry over.
* **Authentication:** `auth_token` cookie or explicit token required.
* **Request Body:** `application/json`
    ```json
    {
      "email": "user@example.com", // string, required
      "password": "securepassword", // string, required
      "username": "preferred_username" // string, optional; keeps the current username if omitted
    }
    ```
* **Response (Success: 200 OK):** `text/plain` - "Account claimed successfully."
* **Response (Error):**
    * `400 Bad Request`: Invalid payload, missing email/password, or the account is not ephemeral (already claimed).
    * `403 Forbidden`: Invalid or missing token.
    * `404 Not Found`: User ID from token not found in database.
    * `409 Conflict`: Email already in use by another account.
    * `500 Internal Server Error`: Database error.

#### `GET /user/history`

* **Description:** Returns the authenticated caller's own recent games, newest first: each game's opponents, the caller's score/outcome, the lobby context it was played under, and the rating change it produced when rated. Never exposes another user's history; the caller is read from their token, not a request parameter. A caller with no games gets `200 OK` with an empty `games` array.
* **Authentication:** `auth_token` cookie or explicit token required.
* **Request Body:** None. Query parameters: `limit` (optional, default 20, capped at 100), `offset` (optional, default 0).
* **Response (Success: 200 OK):** `application/json`
    ```json
    {
      "games": [
        {
          "gameId": "{uuid}",
          "playedAt": "{RFC3339 timestamp}",
          "status": "...",          // string, e.g. "completed"
          "roundIndex": 0,          // int16, see note below
          "lobbyType": "private",   // string
          "mode": "casual",         // string
          "rated": false,           // boolean
          "playerCount": 2,         // integer
          "score": 12,              // int, nullable
          "didWin": true,           // boolean, nullable
          "ranking": 1,             // int16, nullable (finishing place)
          "rating": {               // object, null when the game was not rated
            "pool": "1v1",          // string
            "old": 1500,            // integer
            "new": 1512,            // integer
            "delta": 12             // integer, new - old
          },
          "opponents": [
            {
              "userId": "{uuid}",
              "username": "...",
              "score": 8,           // int, nullable
              "didWin": false,      // boolean, nullable
              "ranking": 2          // int16, nullable
            }
          ]
        }
      ],
      "limit": 20,
      "offset": 0,
      "total": 37 // total games the caller has, for paging without a second request
    }
    ```
    **`roundIndex`** is `games.round_index`: `0` for a game played outside a circuit (a casual single game, or a non-circuit ranked match), and otherwise the 1-based round the game was within its circuit - `1` for a fresh circuit's first game, `N+1` once `N` rounds have been recorded. It is written once, from `CambiaGame.RoundIndex`, when the game's row is created, and is never revised afterward even if the circuit that produced it is later abandoned (cambia-1240).
* **Response (Error):**
    * `400 Bad Request`: `limit` or `offset` present and not a valid non-negative integer (`limit` must be positive).
    * `401 Unauthorized`: No credential presented.
    * `403 Forbidden`: Credential present but invalid.
    * `500 Internal Server Error`: Database error.

#### `GET /user/ratings`

* **Description:** Returns the authenticated caller's current Glicko-2 standing (rating, deviation, volatility, game/win counts, peak rating) in every rating pool, the OpenSkill mu/sigma pair used for circuit play, and the caller's lifetime win/loss record. Every pool is always present, at baseline values for a pool the caller has never played, so the response shape does not change with play history.
* **Authentication:** `auth_token` cookie or explicit token required.
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json`
    ```json
    {
      "pools": [
        {
          "pool": "1v1",       // string: "1v1" | "4p" | "7p8p"
          "rating": 1500,      // integer
          "rd": 350.0,         // float64
          "volatility": 0.06,  // float64
          "games": 12,         // integer, rated games played in this pool
          "wins": 7,           // integer
          "peak": 1540         // integer, highest rating ever reached in this pool
        }
      ],
      "openSkill": { "mu": 25.0, "sigma": 8.333 },
      "record": { "games": 40, "wins": 21 } // lifetime, across all pools
    }
    ```
* **Response (Error):**
    * `401 Unauthorized`: No credential presented.
    * `403 Forbidden`: Credential present but invalid.
    * `500 Internal Server Error`: Database error.

---

### Dev Identity Endpoints

Handled by `internal/handlers/dev_session.go`. Registered only when `CAMBIA_DEV_ACCOUNTS=1` (or `true`); with the variable unset the routes do not exist and `/dev/session` returns `404 Not Found`, indistinguishable from an unknown path. They mint tab-scoped tokens for the dev identity switcher (cambia-1149) and never set a cookie.

#### `POST /dev/session`

* **Description:** Upserts a named dev account and returns a token for it. The account is keyed by email `<name>@dev.cambia.local`, has `is_ephemeral=false`, and is created with a random password nobody holds, so `POST /user/login` cannot reach it. Idempotent per name: repeat calls return the same user id. An absent or empty `name` mints a fresh ephemeral guest instead, in the same response shape.
* **Authentication:** None required (the flag is the gate).
* **Request Body:** `application/json`
    ```json
    {
      "name": "alice" // string, optional, [a-z0-9_-]{1,32}
    }
    ```
* **Response (Success: 200 OK):** `application/json`, no `Set-Cookie`
    ```json
    {
      "token": "{jwt}",
      "user": {
        "id": "...",          // string (UUID)
        "username": "alice",  // string
        "is_ephemeral": false, // boolean
        "is_admin": false     // boolean
      }
    }
    ```
* **Response (Error):**
    * `400 Bad Request`: Malformed body, or a name outside `[a-z0-9_-]{1,32}`.
    * `405 Method Not Allowed`: Any method other than GET or POST.
    * `500 Internal Server Error`: Database or JWT failure.

#### `GET /dev/session`

* **Description:** Lists the dev accounts that exist, so the switcher can offer them.
* **Response (Success: 200 OK):** `application/json`
    ```json
    {
      "enabled": true,
      "accounts": [
        { "name": "alice", "id": "..." }
      ]
    }
    ```

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
      "presetId": "h2h_rapid", // string, optional; a ruleset from GET /lobby/presets
      // Partial houseRules, circuit, or settings objects can be included
      "houseRules": { "turnTimerSec": 30 }, // optional
      "circuit": { "enabled": true }, // optional
      "settings": { "autoStart": false } // optional; "lobbySettings" is accepted as an alias
    }
    ```
    The auto-start block is read from **`settings`**. `lobbySettings` is accepted as an alias for
    it, because that is the key the same block serializes back out as in the response and in
    `lobby_state`, so a client can echo a lobby payload it was handed without its auto-start
    silently going missing. Both spellings are refused for a queue-backed lobby.
    **`presetId`** names a whole ruleset from `GET /lobby/presets` instead of sending the sheet
    field by field. The preset's house rules and lobby settings are applied first and an explicit
    `houseRules` object in the same request lands on top of them, so a host can depart from a
    preset in one call. A preset that fixes a player count also fixes `gameMode`
    (`head_to_head` for a 2-player preset, `group_of_4` for a 4-player one) and overrides any
    `gameMode` sent alongside it; the `default` preset fixes none, so `gameMode` is the caller's.
    A preset's `rounds` figure describes the queue it came from and is not applied to the lobby:
    a custom lobby has no round count until the round lifecycle lands (cambia-466).
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
    * `Unknown ruleset preset: <id>` - the `presetId` names no preset in `GET /lobby/presets`.
    * `Ruleset presets are not accepted for a ranked matchmaking lobby` - a `presetId` alongside
      a resolved queue id. A queue owns its lobby's rules, the same rule the WebSocket
      `update_rules` lock enforces (cambia-966).
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
        "presetId": "h2h_rapid", // string, omitted when the lobby is on no preset
        "mode": "casual",
        "searching": false
    }
    ```
    `name` is the host-supplied display name (empty string when omitted from the request), and
    `mode`/`searching` always serialize. `gameId` carries no `omitempty` tag: `GameID` is a
    `uuid.UUID` (a fixed-size byte array), and Go's `encoding/json` only treats a pointer, slice,
    map, or string as "empty" for that tag, never a fixed-size array - the tag would have been a
    no-op, so a lobby with no game yet always reports the nil UUID rather than omitting the key.
    `queueID` and `presetId` are the two fields that genuinely omit: both are plain strings,
    empty until a queue or a preset is selected.

    **`presetId`** is which ruleset the lobby carries, recorded by the service rather than
    derived from `houseRules` (cambia-1123). It has to be recorded: MATCHMAKING.md 5.2 fixes one
    ruleset for every ranked queue, so all six queue presets hold byte-identical `houseRules` and
    a client matching values against `GET /lobby/presets` would name whichever preset that list
    happens to return first. Set when a `presetId` is accepted on create and when `update_rules`
    carries one, and cleared by the first later edit that moves a house rule or lobby setting
    without naming a preset. Circuit settings are not part of a preset, so changing them leaves
    it alone.

    A preset that fixes a player count fixes `gameMode` with it wherever it is accepted, on create
    and in `update_rules` alike, so a lobby never records a 4-player ruleset while still calling
    itself `head_to_head`. An `update_rules` naming a preset that seats fewer players than the
    lobby already has is refused outright and changes nothing (see `lobby_actions.md`).

    A queue-backed lobby (`type: "matchmaking"`, or any lobby with a `queueID`) is created
    carrying its queue's preset: `houseRules`, `lobbySettings` and `presetId` are the queue's
    from the moment of creation, and the games it produces are built from the lobby. Before
    cambia-1123 the queue's rules were only read at game creation, so a matchmade lobby reported
    the defaults it was constructed with while its game ran the queue's ruleset.
* **Response (Error):** `400 Bad Request` (invalid type/mode/payload), `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`.

#### `GET /lobby/presets`

* **Description:** Lists the selectable rulesets a custom lobby can adopt: the default one first, then one per matchmaking queue in the same order `GET /matchmaking/queues` returns them. Defined once in `internal/lobby/presets.go`; the New lobby dialog, the lobby rule sheet, `POST /lobby/create` and the games a queue produces all read that one definition. Requires no authentication: the handler reads no identity, so the list is the same for every caller.
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json` - an array of presets.
    ```json
    [
      {
        "id": "default",
        "name": "Default",
        "description": "The rules a new lobby starts with.",
        "gameMode": "",
        "players": 0,
        "rounds": 1,
        "ranked": false,
        "houseRules": { ... },  // Full HouseRules object
        "settings": { "autoStart": true }
      },
      {
        "id": "h2h_rapid",
        "name": "H2H Rapid",
        "description": "Ranked queue rules. A custom lobby plays a single round.",
        "gameMode": "head_to_head",
        "players": 2,
        "rounds": 8,
        "ranked": true,
        "houseRules": { ... },
        "settings": { "autoStart": true }
      }
    ]
    ```
    A queue preset's `id` is that queue's id, and its `houseRules` are the fixed ranked configuration MATCHMAKING.md 5.2 specifies (`allowDrawFromDiscardPile` and `allowReplaceAbilities` on, `lockCallerHand` off for the T1C fix, `snapRace` on, a full 54-card deck) with the queue's own reconnect grace. Those are the rules the queue's own games are built with (`NewCambiaGameFromLobby`), so the preset a client is shown for a queue is the ruleset that queue plays.

    `gameMode` is empty for the default preset, which fixes no player count. `rounds` describes the queue and is not applied to a lobby: custom lobbies have no round count until the round lifecycle lands (cambia-466). Circuit settings are deliberately absent from a preset, since it cannot express a round count and would otherwise reset a host's circuit configuration to say something it never meant.

    Its own endpoint rather than extra fields on `GET /matchmaking/queues`: the default ruleset is not a queue and has no place in a list the dashboard renders as queue cards, the queue list is polled for live stats while this one is static, and the lobby rule sheet needs the rulesets without the queue stats.
* **Response (Error):** `405 Method Not Allowed` for anything but `GET`.

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

#### `POST /lobby/{id}/join`

* **Description:** Joins a public lobby, or a private one the caller was invited to. `200 OK` with `{"lobby_id":"<id>"}` on success; `400 Bad Request` (invalid lobby id), `403 Forbidden` (private, not invited), `404 Not Found`. See `lobby_actions.md` for the full contract.

#### `POST /lobby/{id}/leave`

* **Description:** Leaves the lobby; not a WebSocket message (`lobby_actions.md`, "Leave Lobby"). `200 OK` (`{"status":"left",...}`) for a caller with no live seat, or a caller not a member at all; `409 Conflict` for a live seat's plain leave; `{"forfeit": true}` forfeits a live seat and then leaves it, also `200 OK`; `404 Not Found` for an unknown lobby. See `lobby_actions.md`, "Leave Lobby" for the row that states this contract in full.

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

### Leaderboard Endpoint

Handled by `internal/handlers/leaderboard.go`.

#### `GET /leaderboard`

* **Description:** Returns the top-rated users in one rating pool, plus the authenticated caller's own row: present with the caller's true global rank even when it falls outside the returned page, `null` when the caller has no rated games in that pool.
* **Authentication:** `auth_token` cookie or explicit token required.
* **Request Body:** None. Query parameters: `pool` (required, one of `1v1`, `4p`, `7p8p`), `limit` (optional, default 50, capped at 100).
* **Response (Success: 200 OK):** `application/json`
    ```json
    {
      "pool": "1v1",
      "rows": [
        {
          "rank": 1,
          "userId": "{uuid}",
          "username": "...",
          "rating": 1912,
          "rd": 45.2,
          "games": 130
        }
      ],
      "you": {
        "rank": 57,
        "userId": "{uuid}",
        "username": "...",
        "rating": 1502,
        "rd": 210.0,
        "games": 4
      }
    }
    ```
* **Response (Error):**
    * `400 Bad Request`: `pool` missing or not one of `1v1`, `4p`, `7p8p`.
    * `401 Unauthorized`: No credential presented.
    * `403 Forbidden`: Credential present but invalid.
    * `500 Internal Server Error`: Database error.

## WebSocket Endpoints

These endpoints handle real-time communication for lobbies and active games. They require the `auth_token` cookie (or trigger guest creation) and the `cambia` subprotocol.

**Token carrier and subprotocol selection.** A browser cannot set headers on `new WebSocket`, so a tab-held token rides the handshake: the client offers `['cambia', 'cambia-token.<jwt>']`, and the server resolves the second entry as an explicit token (see Authentication above). Every socket in the service - the gameplay socket at `/ws/{lobby_id}` and the training sockets at `/ws/training/*` - accepts with `wsopts.AcceptOptions(wsopts.Subprotocol)` and therefore always selects `cambia`; the `cambia-token.` entry is never selected. A client that offers subprotocols and is handed none back must fail the handshake per RFC 6455, which is why the training sockets select `cambia` too. A client that offers no subprotocol still connects, with none negotiated. `?token=` in the URL is not supported: it would put tokens in proxy and access logs.

*(Note: the WebSocket handler upgrades HTTP connections initiated at this path. See `cmd/server/main.go` for registration.)*

There is one gameplay accept site, not one per phase: separate lobby and game endpoints, each with their own handler file, predate the hub unification and no longer exist. A single socket carries a lobby through joining, readiness, an in-progress game, and back to the lobby, since a hub's `Phase` - not the URL - decides whether an incoming frame is lobby-handled or game-handled (`internal/hub/hub.go`, `dispatch`).

| Path             | Subprotocol | Description                                                               | Handler Location                           | Payload Details Reference                       |
| :---------------- | :---------- | :--------------------------------------------------------------------------- | :------------------------------------------- | :------------------------------------------------ |
| `/ws/{lobby_id}`  | `cambia`    | Unified socket for a lobby: joining, readiness, chat, rule edits, in-game actions, and results, routed by the hub's current phase. | `internal/handlers/ws.go` (`HubWSHandler`) | `docs/lobby_actions.md`, `docs/game_actions.md` |

---

*(Refer to `docs/lobby_actions.md` and `docs/game_actions.md` for detailed WebSocket message payloads.)*
