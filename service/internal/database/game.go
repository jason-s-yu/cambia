// internal/database/game.go
package database

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/jason-s-yu/cambia/service/internal/rating"
)

// RecordGameAndResults persists the final outcome of a game, plus updates rating (1v1, 4p, 7p/8p)
// when rated is true. Rating is further gated on supported player counts: 2 => "1v1", 4 => "4p",
// 7 or 8 => "7p8p", anything else => no rating update. game_results rows are always written
// regardless of rated, so score history survives for casual games too.
//
// Idempotent: the completion UPDATE is conditioned on status != 'completed'. If the game was
// already recorded (duplicate end-event from a reconnect/replay/timer race), this is a no-op
// that returns nil rather than re-inserting results or re-applying rating deltas.
func RecordGameAndResults(ctx context.Context, gameID uuid.UUID, players []*models.Player, finalScores map[uuid.UUID]int, winners []uuid.UUID, rated bool) error {
	var alreadyRecorded bool

	err := pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// Mark the game row completed. This assumes the row already exists (created at
		// game-start via UpsertInitialGameState, which supplies lobby_id): games.lobby_id is NOT NULL
		// with no default, so an INSERT here would always fail a not-null violation; a plain
		// conditional UPDATE matches this function's actual invariant and doubles as the
		// idempotency guard (status != 'completed' skips duplicate end-events).
		updGame := `UPDATE games SET status = 'completed' WHERE id = $1 AND status != 'completed'`
		ct, e := tx.Exec(ctx, updGame, gameID)
		if e != nil {
			return e
		}
		if ct.RowsAffected() == 0 {
			var existingStatus string
			lookupErr := tx.QueryRow(ctx, `SELECT status FROM games WHERE id = $1`, gameID).Scan(&existingStatus)
			switch {
			case errors.Is(lookupErr, pgx.ErrNoRows):
				return fmt.Errorf("no games row for %v: game-start invariant violated (UpsertInitialGameState was not called)", gameID)
			case lookupErr != nil:
				return lookupErr
			default:
				log.Printf("RecordGameAndResults: game %v already completed (status=%s); skipping duplicate record", gameID, existingStatus)
				alreadyRecorded = true
				return nil
			}
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
	if alreadyRecorded {
		return nil
	}

	if !rated {
		log.Printf("Game %v: unrated, skipping rating update for %d players.", gameID, len(players))
		return nil
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

// UpsertInitialGameState creates the games row for gameID and stores 'snap' of the deck +
// initial player hands into games.initial_game_state. This is the sole place a games row gets
// created, and games.lobby_id is NOT NULL with an FK to lobbies(id): the ephemeral in-memory
// lobby (internal/lobby.Lobby) is never itself persisted, so the referenced lobbies row is
// upserted here first, in the same transaction, using the minimal fields needed to satisfy the
// FK and to record whether the game is rated (cambia-450). lobbyType must be a valid lobby_type
// enum value ("private", "public", "matchmaking"); an empty/invalid value fails the insert and
// the error is returned to the caller rather than discarded.
func UpsertInitialGameState(ctx context.Context, gameID, lobbyID, hostUserID uuid.UUID, lobbyType string, rated bool, initialData interface{}) error {
	dataBytes, err := json.Marshal(initialData)
	if err != nil {
		return fmt.Errorf("marshal initial game state for game %v: %w", gameID, err)
	}

	lobbyMode := "casual"
	if rated {
		lobbyMode = "ranked"
	}

	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		lobQ := `
			INSERT INTO lobbies (id, host_user_id, type, mode, ranked)
			VALUES ($1, $2, $3, $4, $5)
			ON CONFLICT (id) DO NOTHING
		`
		if _, e := tx.Exec(ctx, lobQ, lobbyID, hostUserID, lobbyType, lobbyMode, rated); e != nil {
			return fmt.Errorf("upsert lobbies row for game %v: %w", gameID, e)
		}

		gameQ := `
			INSERT INTO games (id, lobby_id, status, initial_game_state, start_time)
			VALUES ($1, $2, 'in_progress', $3, NOW())
			ON CONFLICT (id)
			DO UPDATE SET initial_game_state = EXCLUDED.initial_game_state, status = 'in_progress'
		`
		if _, e := tx.Exec(ctx, gameQ, gameID, lobbyID, dataBytes); e != nil {
			return fmt.Errorf("upsert games row for game %v: %w", gameID, e)
		}
		return nil
	})
}
