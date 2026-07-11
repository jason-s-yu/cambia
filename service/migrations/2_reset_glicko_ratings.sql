-- migrations/2_reset_glicko_ratings.sql
-- cambia-243: FinalizeRatings previously re-applied each match result through a
-- 10-iteration Glicko-2 loop (MultiIterationGlicko2), overshooting the true
-- single-rating-period update by roughly 2x. All stored ratings computed under
-- that bug are invalid. This is a dev-only database (no live players), so the
-- fix is a plain reset rather than a recompute.

-- Reset denormalized Elo-style columns to their Glicko-2 / baseline defaults.
UPDATE users SET
    elo_1v1   = 1500,
    elo_4p    = 1500,
    elo_7p8p  = 1500,
    phi_1v1   = 350.0,
    sigma_1v1 = 0.06;

-- Historical per-game rating deltas in this table were computed under the
-- 10x-overshoot bug; discard them along with the ratings they recorded.
TRUNCATE TABLE ratings;
