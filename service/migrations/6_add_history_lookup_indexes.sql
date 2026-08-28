-- migrations/6_add_history_lookup_indexes.sql
-- cambia-784: the post-game history and rating-summary endpoints
-- (internal/database/history.go) read game_results and ratings by the caller's user id,
-- an access path neither table has an index for. game_results carries only its primary
-- key and the (game_id, player_id) unique constraint from migration 4, whose leading
-- column is game_id, so a lookup keyed on player_id alone cannot use it. ratings has no
-- index at all beyond its primary key, and both the per-pool aggregate (user_id,
-- rating_mode) and the per-game delta join (game_id, user_id) filter on columns it does
-- not cover. Add the three indexes those queries need; no table shape changes.

CREATE INDEX IF NOT EXISTS game_results_player_id_idx ON game_results (player_id);
CREATE INDEX IF NOT EXISTS ratings_user_id_rating_mode_idx ON ratings (user_id, rating_mode);
CREATE INDEX IF NOT EXISTS ratings_game_id_user_id_idx ON ratings (game_id, user_id);
