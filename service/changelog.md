<!-- markdownlint-disable MD024 -->

# Changelog

## [0.2.3] - 2025-04-10

### Added

- Added `cambia-historian` in `cmd/db/historian.go`, which asynchronously pulls messages from the Redis cache and persists it to the database.
- Added Redis service in docker compose.

## [0.2.2] - 2025-04-10

### Added

- Add a callback-based `OnEmpty` mechanism to the `Lobby` struct.

### Changed

- Trigger the `OnEmpty` callback automatically when the last user leaves a lobby, thereby deleting it from the in-memory store.
- Refine the lobby removal flow to support a robust and clean ephemeral lifecycle without external references or global variables.
- Preserve existing ephemeral features from v0.2.1 (including chat, invitations, and game launch) while implementing a modular removal process.

## [0.2.1] - 2025-04-10

### Removed

- Remove all database persistence for lobbies and participants.

### Added

- Introduce fully in-memory lobby management to support ephemeral lobbies.
- Create tests demonstrating ephemeral lobby creation, listing, and invites.

### Changed

- Maintain database usage solely for user data and friend relationships.
- Refactor lobby logic to rely on an ephemeral `LobbyStore` for handling invites, join/ready states, chat, and game launch.
- Establish a lightweight approach for returning users to the lobby after a game ends.
