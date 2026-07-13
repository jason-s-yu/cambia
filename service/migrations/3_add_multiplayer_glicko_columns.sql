-- migrations/3_add_multiplayer_glicko_columns.sql
-- cambia-381: elo_4p and elo_7p8p have no Glicko-2 deviation/volatility tracking,
-- unlike elo_1v1 (phi_1v1/sigma_1v1 from 0_init.sql). Add matching columns so
-- rating deviation for the 4p and 7p8p pools can converge with play instead of
-- resetting to the baseline default every game.

ALTER TABLE users ADD COLUMN IF NOT EXISTS phi_4p FLOAT NOT NULL DEFAULT 350.0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS sigma_4p FLOAT NOT NULL DEFAULT 0.06;
ALTER TABLE users ADD COLUMN IF NOT EXISTS phi_7p8p FLOAT NOT NULL DEFAULT 350.0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS sigma_7p8p FLOAT NOT NULL DEFAULT 0.06;
