-- migrations/5_add_lobby_persistence.sql
-- cambia-450: RecordGameAndResults (internal/database/game.go) had zero production callers,
-- so real games never updated ratings. Wiring it in required creating the games row at
-- game-start with its lobby_id, since games.lobby_id is NOT NULL with no default. But the
-- shared dev database already carries a lobbies table, a lobby_participants table, and a
-- games.lobby_id column (with FK to lobbies) that were never captured in any committed
-- migration -- they exist only in that one hand-evolved database. This migration brings the
-- migration history to parity with what the application code has actually depended on, so a
-- fresh database (CI, a new dev machine) bootstraps into the same shape instead of missing
-- the lobbies table and failing every game-start insert.
--
-- Also adds games.final_game_state, which 0_init.sql declares but the shared dev database is
-- missing (a second, unrelated instance of the same migrations/live-schema drift, discovered
-- via a live-DB test while fixing this ticket). Fixed in the same migration since it is the
-- same table and the same underlying problem (committed migrations lagging the real schema).

CREATE TYPE lobby_type AS ENUM ('private', 'public', 'matchmaking');

CREATE TABLE IF NOT EXISTS lobbies (
    id                                 UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    host_user_id                       UUID NOT NULL REFERENCES users(id),
    type                                lobby_type NOT NULL,
    mode                                TEXT,
    house_rule_freeze_disconnect       BOOLEAN NOT NULL DEFAULT FALSE,
    house_rule_forfeit_disconnect      BOOLEAN NOT NULL DEFAULT FALSE,
    house_rule_missed_round_threshold  SMALLINT NOT NULL DEFAULT 2,
    allow_draw_from_discard_pile       BOOLEAN NOT NULL DEFAULT FALSE,
    penalty_card_count                 SMALLINT NOT NULL DEFAULT 2,
    allow_replaced_discard_abilities   BOOLEAN NOT NULL DEFAULT FALSE,
    disconnection_threshold            SMALLINT NOT NULL DEFAULT 2,
    circuit_mode                       BOOLEAN NOT NULL DEFAULT FALSE,
    circuit_elimination_score          INTEGER,
    circuit_num_rounds                 SMALLINT,
    false_cambia_penalty               SMALLINT DEFAULT 0,
    win_bonus                          SMALLINT DEFAULT 0,
    turn_timeout_sec                   INTEGER NOT NULL DEFAULT 15,
    auto_start                         BOOLEAN NOT NULL DEFAULT TRUE,
    ranked                             BOOLEAN NOT NULL DEFAULT FALSE,
    ranking_mode                       TEXT,
    start_time                         TIMESTAMP,
    end_time                           TIMESTAMP,
    created_at                         TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at                         TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS lobby_participants (
    lobby_id      UUID NOT NULL REFERENCES lobbies(id) ON DELETE CASCADE,
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    is_ready      BOOLEAN NOT NULL DEFAULT FALSE,
    seat_position SMALLINT,
    updated_at    TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (lobby_id, user_id)
);

CREATE TRIGGER set_updated_at_lobbies
BEFORE UPDATE ON lobbies
FOR EACH ROW
EXECUTE PROCEDURE set_updated_at();

CREATE TRIGGER set_updated_at_lobby_participants
BEFORE UPDATE ON lobby_participants
FOR EACH ROW
EXECUTE PROCEDURE set_updated_at();

-- games.lobby_id: dev-only database (no live players, see migration 2's precedent), safe to
-- add NOT NULL directly since no games row is ever created without one going forward.
ALTER TABLE games ADD COLUMN IF NOT EXISTS lobby_id UUID REFERENCES lobbies(id) ON DELETE CASCADE;
ALTER TABLE games ALTER COLUMN lobby_id SET NOT NULL;

ALTER TABLE games ADD COLUMN IF NOT EXISTS final_game_state JSONB;
