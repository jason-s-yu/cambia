-- migrations/4_add_game_results_unique_constraint.sql
-- RecordGameAndResults (internal/database/game.go) upserts game_results via
-- ON CONFLICT (game_id, player_id) DO UPDATE, but no unique constraint on that pair was
-- ever added in 0_init.sql, so Postgres rejects the statement at plan time (42P10: no
-- unique or exclusion constraint matching the ON CONFLICT specification). Found while
-- adding a live-DB test for cambia-381 (RecordGameAndResults has no production callers
-- yet, so this was never exercised). Add the constraint the application code already
-- assumes exists.

ALTER TABLE game_results ADD CONSTRAINT game_results_game_id_player_id_key UNIQUE (game_id, player_id);
