-- migrations/1_add_openskill.sql
-- Add OpenSkill (Plackett-Luce) rating columns for circuit/tournament mode.

ALTER TABLE users ADD COLUMN IF NOT EXISTS open_skill_mu FLOAT NOT NULL DEFAULT 25.0;
ALTER TABLE users ADD COLUMN IF NOT EXISTS open_skill_sigma FLOAT NOT NULL DEFAULT 8.333;
