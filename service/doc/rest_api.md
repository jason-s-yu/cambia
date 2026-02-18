# Cambia Service API (docs/api.md)

This document outlines the available HTTP REST endpoints and WebSocket connections for the Cambia game service.

## Authentication

Most endpoints require authentication via an `auth_token` JWT cookie sent in the `Cookie` header.

* **Obtaining a Token:** Use `POST /user/login`. The token is returned in the response body and set as an `HttpOnly` cookie. The default expiration time is configurable via the `TOKEN_EXPIRE_TIME` environment variable (e.g., "72h", "0" or "never" for no expiration).
* **Ephemeral Guests:** Connecting to a WebSocket endpoint (`/lobby/ws/*` or `/game/ws/*`) *without* a valid `auth_token` cookie will automatically create a temporary guest user, set the `auth_token` cookie, and return the user's ephemeral ID.
* **Claiming Guests:** Guests can optionally call `POST /user/create` (currently this creates a *new* user, an endpoint like `/user/claim` would be needed to convert the ephemeral user, though `ClaimEphemeralHandler` exists it's not wired to a route).
* **Token Verification:** The server uses an Ed25519 key pair (generated at runtime by default) to sign and verify JWTs.

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

Handled by `internal/handlers/lobby.go`. Require `auth_token` cookie. These manage *ephemeral* in-memory lobbies.

#### `POST /lobby/create`

* **Description:** Creates a new ephemeral game lobby in memory, hosted by the authenticated user. Lobby is automatically deleted when the last user leaves.
* **Request Body:** `application/json` (Optional - defaults apply if omitted)
    ```json
    {
      "type": "private" | "public" | "matchmaking", // string, optional (default: "private")
      "gameMode": "head_to_head" | "group_of_4" | ..., // string, optional (default: "head_to_head")
      // Partial houseRules, circuit, or lobbySettings objects can be included
      "houseRules": { "turnTimerSec": 30 }, // optional
      "lobbySettings": { "autoStart": false } // optional
    }
    ```
* **Response (Success: 200 OK):** `application/json` - Returns the full state of the created lobby.
    ```json
    {
        "id": "{uuid}",
        "hostUserID": "{uuid}",
        "type": "private",
        "gameMode": "head_to_head",
        "inGame": false,
        "houseRules": { ... }, // Full HouseRules object
        "circuit": { ... }, // Full Circuit object
        "lobbySettings": { ... } // Full LobbySettings object
    }
    ```
* **Response (Error):** `400 Bad Request` (invalid type/mode/payload), `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`.

#### `GET /lobby/list`

* **Description:** Lists all currently active ephemeral lobbies stored in memory. (Primarily for debugging).
* **Request Body:** None.
* **Response (Success: 200 OK):** `application/json` - Returns a map where keys are lobby UUIDs and values are lobby objects.
    ```json
    {
      "{lobby_uuid_1}": { ... lobby object ... },
      "{lobby_uuid_2}": { ... lobby object ... }
    }
    ```
* **Response (Error):** `401 Unauthorized`, `403 Forbidden`, `500 Internal Server Error`.

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
