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
		// upsert the game row if not exist, set status=completed
		upsertGame := `
			INSERT INTO games (id, status)
			VALUES ($1, 'completed')
			ON CONFLICT (id) DO UPDATE SET status = 'completed'
		`
		if _, e := tx.Exec(ctx, upsertGame, gameID); e != nil {
			return e
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
	var ratingMode string
	switch len(players) {
	case 2:
		ratingMode = "1v1"
	case 4:
		ratingMode = "4p"
	case 7, 8:
		ratingMode = "7p8p"
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
	updated := rating.FinalizeRatings(userList, smap)

	// store updated rating for each user + rating record
	err = pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		for i, uNew := range updated {
			uOld := userList[i]
			oldElo := uOld.Elo1v1
			newElo := uNew.Elo1v1

			// update user row
			updQ := `UPDATE users SET elo_1v1=$1 WHERE id=$2`
			if _, e := tx.Exec(ctx, updQ, newElo, uNew.ID); e != nil {
				return e
			}
			// insert rating record
			insQ := `
				INSERT INTO ratings (user_id, game_id, old_rating, new_rating, rating_mode)
				VALUES ($1, $2, $3, $4, $5)
			`
			if _, e2 := tx.Exec(ctx, insQ, uNew.ID, gameID, oldElo, newElo, ratingMode); e2 != nil {
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
