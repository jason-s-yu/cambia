package database

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
)

// CircuitRatingUpdate holds pre/post OpenSkill values for a single player.
type CircuitRatingUpdate struct {
	UserID   uuid.UUID
	OldMu    float64
	OldSigma float64
	NewMu    float64
	NewSigma float64
}

// UpdateUser1v1Rating updates the user's elo_1v1
func UpdateUser1v1Rating(ctx context.Context, userID uuid.UUID, newRating int) error {
	q := `UPDATE users SET elo_1v1 = $1 WHERE id = $2`
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, q, newRating, userID)
		return err
	})
}

// InsertRatingRecord logs a rating change in the 'ratings' table
func InsertRatingRecord(ctx context.Context, userID, gameID uuid.UUID, oldRating, newRating int, mode string) error {
	q := `
		INSERT INTO ratings (user_id, game_id, old_rating, new_rating, rating_mode)
		VALUES ($1, $2, $3, $4, $5)
	`
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, q, userID, gameID, oldRating, newRating, mode)
		return err
	})
}

// SaveUserOpenSkill persists updated OpenSkill ratings for a user.
func SaveUserOpenSkill(ctx context.Context, userID uuid.UUID, mu, sigma float64) error {
	q := `UPDATE users SET open_skill_mu = $1, open_skill_sigma = $2 WHERE id = $3`
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, q, mu, sigma, userID)
		return err
	})
}

// CommitCircuitRatings persists OpenSkill rating updates for all circuit participants.
func CommitCircuitRatings(ctx context.Context, gameID uuid.UUID, updates []CircuitRatingUpdate) error {
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for _, u := range updates {
			if _, err := tx.Exec(ctx,
				`UPDATE users SET open_skill_mu = $1, open_skill_sigma = $2 WHERE id = $3`,
				u.NewMu, u.NewSigma, u.UserID,
			); err != nil {
				return err
			}
			if _, err := tx.Exec(ctx,
				`INSERT INTO ratings (user_id, game_id, old_rating, new_rating, rating_mode) VALUES ($1, $2, $3, $4, $5)`,
				u.UserID, gameID, int(u.OldMu*100), int(u.NewMu*100), "circuit_openskill",
			); err != nil {
				return err
			}
		}
		return nil
	})
}

// Example function that updates 1v1 rating after game completion
func Commit1v1MatchResults(ctx context.Context, winnerID, loserID, gameID uuid.UUID, oldWRating, oldLRating, newWRating, newLRating int) error {
	err := pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		if _, e1 := tx.Exec(ctx, `UPDATE users SET elo_1v1 = $1 WHERE id = $2`, newWRating, winnerID); e1 != nil {
			return e1
		}
		if _, e2 := tx.Exec(ctx, `UPDATE users SET elo_1v1 = $1 WHERE id = $2`, newLRating, loserID); e2 != nil {
			return e2
		}
		_, e3 := tx.Exec(ctx, `
			INSERT INTO ratings (user_id, game_id, old_rating, new_rating, rating_mode)
			VALUES ($1, $2, $3, $4, $5), ($6, $7, $8, $9, $10)
		`,
			winnerID, gameID, oldWRating, newWRating, "1v1",
			loserID, gameID, oldLRating, newLRating, "1v1",
		)
		return e3
	})
	if err != nil {
		return fmt.Errorf("failed to commit 1v1 match results: %w", err)
	}
	return nil
}
