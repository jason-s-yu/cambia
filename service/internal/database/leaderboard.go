// internal/database/leaderboard.go
package database

import (
	"context"
	"fmt"

	"github.com/google/uuid"
)

// LeaderboardRow is a single ranked row returned by GetLeaderboard: a user's global
// rank, rating, and rated-game count for one rating pool.
type LeaderboardRow struct {
	Rank     int
	UserID   uuid.UUID
	Username string
	Rating   int
	RD       float64
	Games    int
}

// leaderboardColumns maps a rating pool name to its Elo and RD (Glicko-2 phi) column
// names on the users table (see rating.RatingMode / migrations/3_add_multiplayer_glicko_columns.sql).
// Callers must validate pool against a known set before calling GetLeaderboard: pool
// is interpolated directly into the query text since pgx cannot parameterize column
// identifiers, and an unrecognized value here returns ok=false rather than building
// an invalid query.
func leaderboardColumns(pool string) (eloCol, rdCol string, ok bool) {
	switch pool {
	case "1v1":
		return "elo_1v1", "phi_1v1", true
	case "4p":
		return "elo_4p", "phi_4p", true
	case "7p8p":
		return "elo_7p8p", "phi_7p8p", true
	}
	return "", "", false
}

// GetLeaderboard returns the top `limit` users ranked by rating (descending) in the
// given pool, plus the caller's own row. The caller's row is included even when it
// falls outside the top `limit` (its Rank reflects true global position among all
// users); it is nil when the caller has zero rated games recorded in this pool
// ("unrated"). Games is a COUNT of each user's rows in the ratings table for this
// pool: a plain aggregate over the existing ratings history, not a new counter.
//
// Ties in rating are broken by user id for a stable, deterministic order across
// calls (rating alone is not unique).
func GetLeaderboard(ctx context.Context, pool string, limit int, callerID uuid.UUID) (rows []LeaderboardRow, you *LeaderboardRow, err error) {
	eloCol, rdCol, ok := leaderboardColumns(pool)
	if !ok {
		return nil, nil, fmt.Errorf("unknown rating pool %q", pool)
	}

	q := fmt.Sprintf(`
		WITH ranked AS (
			SELECT
				u.id,
				u.username,
				u.%s AS rating,
				u.%s AS rd,
				ROW_NUMBER() OVER (ORDER BY u.%s DESC, u.id ASC) AS rank,
				(SELECT COUNT(*) FROM ratings r WHERE r.user_id = u.id AND r.rating_mode = $1) AS games
			FROM users u
		)
		SELECT rank, id, username, rating, rd, games, (id = $3) AS is_caller
		FROM ranked
		WHERE rank <= $2 OR id = $3
		ORDER BY rank
	`, eloCol, rdCol, eloCol)

	pgRows, qErr := DB.Query(ctx, q, pool, limit, callerID)
	if qErr != nil {
		return nil, nil, fmt.Errorf("query leaderboard for pool %s: %w", pool, qErr)
	}
	defer pgRows.Close()

	for pgRows.Next() {
		var row LeaderboardRow
		var isCaller bool
		if scanErr := pgRows.Scan(&row.Rank, &row.UserID, &row.Username, &row.Rating, &row.RD, &row.Games, &isCaller); scanErr != nil {
			return nil, nil, fmt.Errorf("scan leaderboard row for pool %s: %w", pool, scanErr)
		}
		if row.Rank <= limit {
			rows = append(rows, row)
		}
		if isCaller && row.Games > 0 {
			youCopy := row
			you = &youCopy
		}
	}
	if iterErr := pgRows.Err(); iterErr != nil {
		return nil, nil, fmt.Errorf("iterate leaderboard rows for pool %s: %w", pool, iterErr)
	}

	return rows, you, nil
}
