// internal/database/game.go
package database

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/jason-s-yu/cambia/service/internal/rating"
)

// RecordGameAndResults persists the final outcome of a game, plus updates rating (1v1, 4p, 7p/8p).
// We do a basic approach: if players == 2 => "1v1", if 4 => "4p", if 7 or 8 => "7p8p" else no rating update.
func RecordGameAndResults(ctx context.Context, gameID uuid.UUID, players []*models.Player, finalScores map[uuid.UUID]int, winners []uuid.UUID) error {
	err := pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// Mark the game row completed. This assumes the row already exists (created at
		// game-start with a lobby_id, which this function has no way to supply): games.lobby_id
		// is NOT NULL with no default, and Postgres validates that on the proposed row before
		// ON CONFLICT resolution even runs, so an INSERT ... ON CONFLICT DO UPDATE here always
		// fails with a not-null violation regardless of whether a conflicting row exists
		// (verified directly against the schema; not a hypothetical). A plain UPDATE matches
		// this function's actual invariant.
		updGame := `UPDATE games SET status = 'completed' WHERE id = $1`
		ct, e := tx.Exec(ctx, updGame, gameID)
		if e != nil {
			return e
		}
		if ct.RowsAffected() == 0 {
			log.Printf("RecordGameAndResults: no existing games row for game %v; game_results insert will fail its FK", gameID)
		}

		// Insert game_results
		for _, pl := range players {
			score := finalScores[pl.ID]
			didWin := false
			for _, w := range winners {
				if w == pl.ID {
					didWin = true
					break
				}
			}
			q := `
				INSERT INTO game_results (game_id, player_id, score, did_win)
				VALUES ($1, $2, $3, $4)
				ON CONFLICT (game_id, player_id)
				DO UPDATE SET score=$3, did_win=$4
			`
			if _, e2 := tx.Exec(ctx, q, gameID, pl.ID, score, didWin); e2 != nil {
				return e2
			}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("tx upsert game or results: %w", err)
	}

	// figure out rating mode
	var ratingMode rating.RatingMode
	switch len(players) {
	case 2:
		ratingMode = rating.Mode1v1
	case 4:
		ratingMode = rating.Mode4p
	case 7, 8:
		ratingMode = rating.Mode7p8p
	default:
		ratingMode = ""
	}

	if ratingMode == "" {
		log.Printf("No rating update for %d-player game.\n", len(players))
		return nil
	}

	// load user objects from DB for rating
	var userList []models.User
	for _, p := range players {
		u, err := GetUserByID(ctx, p.ID)
		if err != nil {
			log.Printf("user not found for rating: %v\n", p.ID)
			continue
		}
		userList = append(userList, *u)
	}

	// build finalScores => userID => score
	smap := make(map[uuid.UUID]int)
	for _, p := range players {
		smap[p.ID] = finalScores[p.ID]
	}

	// finalize rating
	updated := rating.FinalizeRatings(userList, smap, ratingMode)

	// store updated rating (elo, phi, sigma) for each user + rating record
	err = pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for i, uNew := range updated {
			uOld := userList[i]
			oldElo, _, _ := rating.PoolFields(uOld, ratingMode)
			newElo, newPhi, newSigma := rating.PoolFields(uNew, ratingMode)

			// update user row: elo/phi/sigma columns for the pool this game contributed to
			var updQ string
			switch ratingMode {
			case rating.Mode4p:
				updQ = `UPDATE users SET elo_4p=$1, phi_4p=$2, sigma_4p=$3 WHERE id=$4`
			case rating.Mode7p8p:
				updQ = `UPDATE users SET elo_7p8p=$1, phi_7p8p=$2, sigma_7p8p=$3 WHERE id=$4`
			default:
				updQ = `UPDATE users SET elo_1v1=$1, phi_1v1=$2, sigma_1v1=$3 WHERE id=$4`
			}
			if _, e := tx.Exec(ctx, updQ, newElo, newPhi, newSigma, uNew.ID); e != nil {
				return e
			}
			// insert rating record
			insQ := `
				INSERT INTO ratings (user_id, game_id, old_rating, new_rating, rating_mode)
				VALUES ($1, $2, $3, $4, $5)
			`
			if _, e2 := tx.Exec(ctx, insQ, uNew.ID, gameID, oldElo, newElo, string(ratingMode)); e2 != nil {
				return e2
			}
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("tx rating update: %w", err)
	}

	return nil
}

// StoreFinalGameStateInDB updates the games.final_game_state column with JSON containing
// each player's final hand (rank/suit/value) plus the winner userIDs.
func StoreFinalGameStateInDB(ctx context.Context, gameID uuid.UUID, finalSnapshot map[string]interface{}) error {
	jsonData, err := json.Marshal(finalSnapshot)
	if err != nil {
		return fmt.Errorf("failed to marshal final snapshot: %w", err)
	}
	query := `
		UPDATE games
		SET final_game_state = $1
		WHERE id = $2
	`
	err = pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, e := tx.Exec(ctx, query, jsonData, gameID)
		return e
	})
	if err != nil {
		return fmt.Errorf("storing final game state in DB: %w", err)
	}
	return nil
}

// StoreInitialGameStateInDB sets the games.initial_game_state column with any JSON data
// we want for reconstructing the start of the game (deck order, dealt hands, etc.).
func StoreInitialGameStateInDB(ctx context.Context, gameID uuid.UUID, initSnapshot map[string]interface{}) error {
	js, err := json.Marshal(initSnapshot)
	if err != nil {
		return err
	}
	q := `
		UPDATE games
		SET initial_game_state = $1, status = 'in_progress', start_time = NOW()
		WHERE id = $2
	`
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, e := tx.Exec(ctx, q, js, gameID)
		return e
	})
}

// UpsertInitialGameState stores 'snap' of the deck + initial player hands into games.initial_game_state.
func UpsertInitialGameState(gameID uuid.UUID, initialData interface{}) {
	ctx := context.Background()
	dataBytes, err := json.Marshal(initialData)
	if err != nil {
		log.Printf("failed to marshal initial game state for game %v: %v", gameID, err)
		return
	}
	_ = pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		q := `
			INSERT INTO games (id, status, initial_game_state, start_time)
			VALUES ($1, 'in_progress', $2, NOW())
			ON CONFLICT (id)
			DO UPDATE SET initial_game_state = EXCLUDED.initial_game_state, status='in_progress'
		`
		_, e := tx.Exec(ctx, q, gameID, dataBytes)
		return e
	})
}
