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
	"github.com/jackc/pgx/v5/pgconn"
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
	// Every roster entry carries its own score or nothing is written. Indexing the map for a
	// player it has no entry for yields a 0, which this stored in game_results and handed to the
	// rating sort as the best score at the table; the caller that produced such a map was
	// dropping forfeited seats from it (cambia-1541). Refusing here keeps the roster and the
	// score map one argument in practice, so no future caller can reintroduce the same silence.
	for _, pl := range players {
		if _, ok := finalScores[pl.ID]; !ok {
			return fmt.Errorf("record game %v: no final score for roster player %v", gameID, pl.ID)
		}
	}

	var alreadyRecorded bool

	err := pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// Mark the game row completed and stamp when it ended. This assumes the row already
		// exists (created at game-start via UpsertInitialGameState, which supplies lobby_id):
		// games.lobby_id is NOT NULL with no default, so an INSERT here would always fail a
		// not-null violation; a plain conditional UPDATE matches this function's actual
		// invariant and doubles as the idempotency guard (status != 'completed' skips duplicate
		// end-events), which is also what keeps end_time at the first completion rather than
		// the latest duplicate end-event.
		//
		// end_time moved here in cambia-1904. It had been written only by the historian, on an
		// action name nothing emitted, so it was NULL for every game ever played; the server is
		// the single writer of games rows since cambia-1881, and this transaction is where a
		// game becomes finished.
		updGame := `UPDATE games SET status = 'completed', end_time = NOW() WHERE id = $1 AND status != 'completed'`
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

	playerIDs := make([]uuid.UUID, len(players))
	for i, p := range players {
		playerIDs[i] = p.ID
	}
	return applyRatingUpdate(ctx, gameID, playerIDs, finalScores)
}

// RecordCircuitRatings applies the one rating update a circuit tournament produces, from its
// final cumulative scores (subsidies included). RULES.md T6 and MATCHMAKING.md 6.2/6.3 rate a
// multi-round format strictly once, at its conclusion: the rounds themselves are recorded as
// games and displayed as results but never rated (see game.persistFinalGameState), and this is
// the update that stands in for all of them.
//
// gameID is the circuit's final round, the game whose completion triggers this; it is what the
// ratings rows are attributed to, since ratings.game_id references an existing games row and a
// circuit has no row of its own. playerIDs is the circuit roster (its length selects the rating
// pool, as it does per game) and cumulativeScores maps each of them to its final cumulative
// total, lower being better.
//
// Cumulative totals within CircuitTieMargin points of each other are recorded as a tie
// (rating.CircuitRankScores); everything past that is the same pool-aware path a per-game rating
// takes, writing the same users columns and ratings rows.
func RecordCircuitRatings(ctx context.Context, gameID uuid.UUID, playerIDs []uuid.UUID, cumulativeScores map[uuid.UUID]int) error {
	if len(playerIDs) == 0 {
		return nil
	}
	return applyRatingUpdate(ctx, gameID, playerIDs, rating.CircuitRankScores(playerIDs, cumulativeScores))
}

// applyRatingUpdate runs one pool-aware Glicko-2 update for playerIDs from scores (lower is
// better) and persists it: the users elo/phi/sigma columns for the pool the roster size selects,
// plus one ratings row per player attributed to gameID.
//
// Rating is gated on supported roster sizes: 2 => "1v1", 4 => "4p", 7 or 8 => "7p8p", anything
// else => no rating update. A player whose user row cannot be loaded is dropped from the update
// rather than failing it.
//
// A player with no score is refused instead. Rating on an implicit map miss is what cambia-1541
// fixed: FinalizeRatings sorts ascending and lower is better in Cambia, so the zero value a
// missing key produces is a first-place finish. Two fresh 1500 ratings settled at 1662 and 1338
// that way, with the seat that quit taking the gain.
func applyRatingUpdate(ctx context.Context, gameID uuid.UUID, playerIDs []uuid.UUID, scores map[uuid.UUID]int) error {
	for _, id := range playerIDs {
		if _, ok := scores[id]; !ok {
			return fmt.Errorf("rating update for game %v: no score for roster player %v", gameID, id)
		}
	}

	// figure out rating mode
	var ratingMode rating.RatingMode
	switch len(playerIDs) {
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
		log.Printf("No rating update for %d-player game.\n", len(playerIDs))
		return nil
	}

	// load user objects from DB for rating
	var userList []models.User
	for _, id := range playerIDs {
		u, err := GetUserByID(ctx, id)
		if err != nil {
			log.Printf("user not found for rating: %v\n", id)
			continue
		}
		userList = append(userList, *u)
	}

	// build scores => userID => score
	smap := make(map[uuid.UUID]int)
	for _, id := range playerIDs {
		smap[id] = scores[id]
	}

	// finalize rating
	updated := rating.FinalizeRatings(userList, smap, ratingMode)

	// store updated rating (elo, phi, sigma) for each user + rating record
	err := pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
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
//
// roundIndex is the 1-based circuit round this game belongs to (game.CambiaGame.RoundIndex),
// or 0 for a non-circuit game; it is only ever set at creation and is left alone on the
// ON CONFLICT branch (cambia-1240).
func UpsertInitialGameState(ctx context.Context, gameID, lobbyID, hostUserID uuid.UUID, lobbyType string, rated bool, roundIndex int16, initialData interface{}) error {
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
			INSERT INTO games (id, lobby_id, round_index, status, initial_game_state, start_time)
			VALUES ($1, $2, $3, 'in_progress', $4, NOW())
			ON CONFLICT (id)
			DO UPDATE SET initial_game_state = EXCLUDED.initial_game_state, status = 'in_progress'
		`
		if _, e := tx.Exec(ctx, gameQ, gameID, lobbyID, roundIndex, dataBytes); e != nil {
			return fmt.Errorf("upsert games row for game %v: %w", gameID, e)
		}
		return nil
	})
}

// AbandonStaleGames marks every games row still 'in_progress' as abandoned and stamps its
// end_time, reporting how many rows it closed. It is the server's boot sweep: ConnectDBAsync runs
// it once, after the first successful connection and after migrations, before the process serves
// play.
//
// A game exists only in the server's memory. game.CambiaGame is held by the handlers' in-memory
// game store and nothing rehydrates one from the database, so a row still 'in_progress' when the
// process starts belongs to a game whose process is gone: a crash, a kill, or a deploy that
// landed mid-game. No server path will ever end it, because every path that does end a game (a
// rulebook ending, a forfeit that empties the table, the disconnect grace, the panic guard) runs
// through game.endGame on the object that no longer exists.
//
// Until cambia-1881 the historian covered this from the outside, marking a game abandoned off a
// ten-minute inactivity timer kept in its own process. That went with the historian's games
// writes, and this replaces it exactly rather than on a delay: at boot there is no such thing as
// a legitimately in-progress game, so no activity signal is needed to tell the two apart.
//
// One case this closes only at the next restart rather than promptly: a table where every player
// has dropped, under a lobby that set forfeitOnDisconnect off and turnTimerSec to 0. Nothing then
// forfeits the seats and no turn clock plays them, so the game sits in progress with nobody in it
// until the process ends. Under the default rules (forfeit on, a 15s turn clock) it cannot arise;
// the game.endGame paths close every other game while the server is up.
//
// This assumes one server process per database, which is what the deployment runs
// (deploy/hawking/docker-compose.yml defines a single cambia-server container) and what the
// in-memory game and lobby state already require. A second instance sharing one database would
// abandon the first instance's live games.
// db is the pool in production. It is a parameter rather than the package-level DB because this
// is the one query here that rewrites rows it was not handed the ids of: its test passes a
// transaction and rolls it back, since the dev Postgres is shared by every checkout on the
// machine and a test that really abandoned every in-progress game would reach into other
// packages' fixtures and other people's runs.
func AbandonStaleGames(ctx context.Context, db gamesExecer) (int64, error) {
	q := `
		UPDATE games
		SET status = 'abandoned', end_time = NOW()
		WHERE status = 'in_progress'
	`
	ct, err := db.Exec(ctx, q)
	if err != nil {
		return 0, fmt.Errorf("abandon stale in-progress games: %w", err)
	}
	return ct.RowsAffected(), nil
}

// gamesExecer is the exec surface AbandonStaleGames needs, satisfied by both *pgxpool.Pool and
// pgx.Tx.
type gamesExecer interface {
	Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
}
