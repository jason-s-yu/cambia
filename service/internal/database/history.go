// internal/database/history.go
package database

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// GameHistoryOpponent is one other participant in a game the caller played, as recorded
// in game_results. Score/DidWin/Ranking are pointers because the columns are nullable:
// RecordGameAndResults always writes score and did_win, but ranking has never had a
// writer, so it reads back NULL for every row currently in the table.
type GameHistoryOpponent struct {
	UserID   uuid.UUID
	Username string
	Score    *int
	DidWin   *bool
	Ranking  *int16
}

// GameHistoryEntry is one completed game from the caller's point of view: the caller's
// own game_results row, the lobby context the game was played under, the per-game rating
// change when the game was rated, and every other seat at the table.
//
// RatingMode/OldRating/NewRating come from the ratings row written for this (user, game)
// pair and are nil for an unrated game, or for a rated game whose rating rows predate the
// truncate in migration 2.
type GameHistoryEntry struct {
	GameID      uuid.UUID
	RoundIndex  int16
	Status      string
	PlayedAt    time.Time
	LobbyType   string
	LobbyMode   string
	Rated       bool
	PlayerCount int
	Score       *int
	DidWin      *bool
	Ranking     *int16
	RatingMode  *string
	OldRating   *int
	NewRating   *int
	Opponents   []GameHistoryOpponent
}

// GetUserGameHistory returns one page of the games userID has a recorded result for,
// newest first, plus the total number of such games so a caller can paginate.
//
// A game's timestamp is COALESCE(end_time, updated_at, created_at): end_time is set only
// by the historian's action_end_game path (cmd/db/historian.go), while a game finished
// through RecordGameAndResults gets its completion recorded via the updated_at trigger,
// so neither column alone covers every row. created_at is NOT NULL and backstops both.
//
// game_results rows are only ever written by RecordGameAndResults, which marks the game
// 'completed' in the same transaction, so this is a completed-game list without needing a
// status filter; status is returned anyway rather than assumed.
func GetUserGameHistory(ctx context.Context, userID uuid.UUID, limit, offset int) (entries []GameHistoryEntry, total int, err error) {
	if err = DB.QueryRow(ctx,
		`SELECT COUNT(*) FROM game_results WHERE player_id = $1`, userID,
	).Scan(&total); err != nil {
		return nil, 0, fmt.Errorf("count game history for user %v: %w", userID, err)
	}
	if total == 0 {
		return nil, 0, nil
	}

	// lobbies is LEFT JOINed rather than inner JOINed: games.lobby_id is NOT NULL with an
	// FK to lobbies (migration 5), so the row is expected to be there, but a read endpoint
	// dropping a player's game from their own history is a worse failure than showing that
	// game with blank lobby context.
	//
	// The rating row is picked through a LATERAL rather than a plain LEFT JOIN because a
	// single (user, game) pair can carry more than one ratings row: CommitCircuitRatings
	// writes a 'circuit_openskill' row alongside the Glicko row RecordGameAndResults
	// writes. Ordering the Glicko row first keeps the join to one deterministic row instead
	// of duplicating the game across the page.
	const q = `
		SELECT
			g.id,
			g.round_index,
			g.status,
			COALESCE(g.end_time, g.updated_at, g.created_at) AS played_at,
			COALESCE(l.type::text, ''),
			COALESCE(l.mode, ''),
			COALESCE(l.ranked, FALSE),
			(SELECT COUNT(*) FROM game_results gr_all WHERE gr_all.game_id = g.id),
			gr.score,
			gr.did_win,
			gr.ranking,
			r.rating_mode,
			r.old_rating,
			r.new_rating
		FROM game_results gr
		JOIN games g ON g.id = gr.game_id
		LEFT JOIN lobbies l ON l.id = g.lobby_id
		LEFT JOIN LATERAL (
			SELECT rt.rating_mode, rt.old_rating, rt.new_rating
			FROM ratings rt
			WHERE rt.game_id = g.id AND rt.user_id = gr.player_id
			ORDER BY (rt.rating_mode = 'circuit_openskill'), rt.created_at DESC
			LIMIT 1
		) r ON TRUE
		WHERE gr.player_id = $1
		ORDER BY played_at DESC, g.id DESC
		LIMIT $2 OFFSET $3
	`

	rows, qErr := DB.Query(ctx, q, userID, limit, offset)
	if qErr != nil {
		return nil, 0, fmt.Errorf("query game history for user %v: %w", userID, qErr)
	}
	defer rows.Close()

	gameIDs := make([]uuid.UUID, 0, limit)
	for rows.Next() {
		var e GameHistoryEntry
		if scanErr := rows.Scan(
			&e.GameID, &e.RoundIndex, &e.Status, &e.PlayedAt,
			&e.LobbyType, &e.LobbyMode, &e.Rated, &e.PlayerCount,
			&e.Score, &e.DidWin, &e.Ranking,
			&e.RatingMode, &e.OldRating, &e.NewRating,
		); scanErr != nil {
			return nil, 0, fmt.Errorf("scan game history row for user %v: %w", userID, scanErr)
		}
		entries = append(entries, e)
		gameIDs = append(gameIDs, e.GameID)
	}
	if iterErr := rows.Err(); iterErr != nil {
		return nil, 0, fmt.Errorf("iterate game history rows for user %v: %w", userID, iterErr)
	}
	if len(entries) == 0 {
		return nil, total, nil
	}

	opponents, oppErr := gameOpponents(ctx, gameIDs, userID)
	if oppErr != nil {
		return nil, 0, oppErr
	}
	for i := range entries {
		entries[i].Opponents = opponents[entries[i].GameID]
	}

	return entries, total, nil
}

// gameOpponents loads every game_results row for the given games except excludeUserID's
// own, keyed by game id. Fetched as one query over the page's game ids rather than one
// query per game.
func gameOpponents(ctx context.Context, gameIDs []uuid.UUID, excludeUserID uuid.UUID) (map[uuid.UUID][]GameHistoryOpponent, error) {
	const q = `
		SELECT gr.game_id, gr.player_id, COALESCE(u.username, ''), gr.score, gr.did_win, gr.ranking
		FROM game_results gr
		LEFT JOIN users u ON u.id = gr.player_id
		WHERE gr.game_id = ANY($1::uuid[]) AND gr.player_id <> $2
		ORDER BY gr.game_id, gr.ranking NULLS LAST, gr.score NULLS LAST, gr.player_id
	`
	rows, err := DB.Query(ctx, q, gameIDs, excludeUserID)
	if err != nil {
		return nil, fmt.Errorf("query game opponents: %w", err)
	}
	defer rows.Close()

	out := make(map[uuid.UUID][]GameHistoryOpponent, len(gameIDs))
	for rows.Next() {
		var gameID uuid.UUID
		var o GameHistoryOpponent
		if scanErr := rows.Scan(&gameID, &o.UserID, &o.Username, &o.Score, &o.DidWin, &o.Ranking); scanErr != nil {
			return nil, fmt.Errorf("scan game opponent row: %w", scanErr)
		}
		out[gameID] = append(out[gameID], o)
	}
	if iterErr := rows.Err(); iterErr != nil {
		return nil, fmt.Errorf("iterate game opponent rows: %w", iterErr)
	}
	return out, nil
}

// PoolRatingSummary is a user's standing in one rating pool.
//
// Rating/RD/Volatility come from the users table's elo_*/phi_*/sigma_* columns, the same
// source GetLeaderboard ranks on: current rating has one authoritative store, and is never
// reconstructed by walking the ratings history. Games/Wins/Peak are aggregates over the
// ratings history, which answers a different question (what has happened in this pool)
// than the users row (where the player stands now).
type PoolRatingSummary struct {
	Pool       string
	Rating     int
	RD         float64
	Volatility float64
	Games      int
	Wins       int
	Peak       int
}

// RatingSummary is a user's full rating picture: one entry per Glicko-2 pool, the
// OpenSkill pair used for circuit play, and their lifetime recorded win/loss record
// across every game (rated or not).
type RatingSummary struct {
	Pools          []PoolRatingSummary
	OpenSkillMu    float64
	OpenSkillSigma float64
	TotalGames     int
	TotalWins      int
}

// ratingPools lists the Glicko-2 pools in display order. Each is backed by an
// elo_*/phi_*/sigma_* column triple on users (see rating.RatingMode and
// migrations/3_add_multiplayer_glicko_columns.sql) and by rows in ratings carrying the
// same string as their rating_mode.
var ratingPools = []string{"1v1", "4p", "7p8p"}

// GetUserRatingSummary returns userID's current rating in every pool plus the history
// aggregates that give those numbers context.
//
// Games counts the user's rows in the ratings table for the pool, matching the definition
// GetLeaderboard reports so the two surfaces never disagree about how many rated games a
// player has. Wins counts those whose game the user won.
//
// Peak is the highest rating the user has held in the pool. Each ratings row records both
// sides of a change, so the maximum is taken over old_rating and new_rating together: a
// player who has only ever lost still peaked at the rating they started from, which
// MAX(new_rating) alone would miss. It is then floored at the current rating, since
// migration 2 truncated the ratings table while leaving the users columns in place, and a
// current rating above every recorded row would otherwise display as if the player had
// never reached it.
func GetUserRatingSummary(ctx context.Context, userID uuid.UUID) (*RatingSummary, error) {
	var summary RatingSummary

	// Current standing: one row off users, every pool's columns at once.
	const userQ = `
		SELECT elo_1v1, phi_1v1, sigma_1v1,
		       elo_4p, phi_4p, sigma_4p,
		       elo_7p8p, phi_7p8p, sigma_7p8p,
		       open_skill_mu, open_skill_sigma
		FROM users WHERE id = $1
	`
	standing := make([]PoolRatingSummary, len(ratingPools))
	if err := DB.QueryRow(ctx, userQ, userID).Scan(
		&standing[0].Rating, &standing[0].RD, &standing[0].Volatility,
		&standing[1].Rating, &standing[1].RD, &standing[1].Volatility,
		&standing[2].Rating, &standing[2].RD, &standing[2].Volatility,
		&summary.OpenSkillMu, &summary.OpenSkillSigma,
	); err != nil {
		return nil, fmt.Errorf("query rating summary for user %v: %w", userID, err)
	}

	// History aggregates: one grouped pass over the user's ratings rows, joined to the
	// game's result row for the win count. Pools with no history are absent here and fall
	// back to zeroed aggregates below.
	const histQ = `
		SELECT r.rating_mode,
		       COUNT(*),
		       COUNT(*) FILTER (WHERE gr.did_win),
		       GREATEST(MAX(r.new_rating), MAX(r.old_rating))
		FROM ratings r
		LEFT JOIN game_results gr ON gr.game_id = r.game_id AND gr.player_id = r.user_id
		WHERE r.user_id = $1
		GROUP BY r.rating_mode
	`
	rows, err := DB.Query(ctx, histQ, userID)
	if err != nil {
		return nil, fmt.Errorf("query rating history for user %v: %w", userID, err)
	}
	defer rows.Close()

	type poolHistory struct {
		games int
		wins  int
		peak  *int
	}
	history := make(map[string]poolHistory, len(ratingPools))
	for rows.Next() {
		var mode string
		var h poolHistory
		if scanErr := rows.Scan(&mode, &h.games, &h.wins, &h.peak); scanErr != nil {
			return nil, fmt.Errorf("scan rating history row for user %v: %w", userID, scanErr)
		}
		history[mode] = h
	}
	if iterErr := rows.Err(); iterErr != nil {
		return nil, fmt.Errorf("iterate rating history rows for user %v: %w", userID, iterErr)
	}

	summary.Pools = make([]PoolRatingSummary, 0, len(ratingPools))
	for i, pool := range ratingPools {
		p := standing[i]
		p.Pool = pool
		p.Peak = p.Rating
		if h, ok := history[pool]; ok {
			p.Games, p.Wins = h.games, h.wins
			if h.peak != nil && *h.peak > p.Peak {
				p.Peak = *h.peak
			}
		}
		summary.Pools = append(summary.Pools, p)
	}

	// Lifetime record over every recorded game, rated or not: game_results is written for
	// casual games too, so this is a wider count than the per-pool rated totals above.
	const recordQ = `
		SELECT COUNT(*), COUNT(*) FILTER (WHERE did_win)
		FROM game_results WHERE player_id = $1
	`
	if err := DB.QueryRow(ctx, recordQ, userID).Scan(&summary.TotalGames, &summary.TotalWins); err != nil {
		return nil, fmt.Errorf("query lifetime record for user %v: %w", userID, err)
	}

	return &summary, nil
}
