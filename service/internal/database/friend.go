// internal/database/friend.go

package database

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// InsertFriendRequest inserts a row into the friends table with status='pending'.
func InsertFriendRequest(ctx context.Context, user1, user2 uuid.UUID) error {
	// insert relation in row with status=pending
	q := `
		INSERT INTO friends (user1_id, user2_id, status)
		VALUES ($1, $2, 'pending')
		ON CONFLICT (user1_id, user2_id) 
		DO UPDATE SET status='pending', updated_at=NOW()
	`
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, q, user1, user2)
		return err
	})
}

// AcceptFriend sets status='accepted' for (user1_id, user2_id).
func AcceptFriend(ctx context.Context, user1, user2 uuid.UUID) error {
	// assume user 1 requests user 2, accepting simply sets status=accepted
	q := `
		UPDATE friends
		SET status='accepted', updated_at=NOW()
		WHERE user1_id=$1 AND user2_id=$2 AND status='pending'
	`
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		ct, err := tx.Exec(ctx, q, user1, user2)
		if err != nil {
			return err
		}
		if ct.RowsAffected() == 0 {
			return fmt.Errorf("no pending friend request found from %v to %v", user1, user2)
		}
		// Optionally, we might also insert the reverse row with 'accepted'
		// if you want mutual entries. For now, let's keep one row approach.
		return nil
	})
}

// ListFriends returns all friend relationships for a user, including both accepted & pending.
func ListFriends(ctx context.Context, userID uuid.UUID) ([]models.Friend, error) {
	// return rows matching (user1_id=userID or user2_id=userID), including pending or accepted
	q := `
		SELECT user1_id, user2_id, status
		FROM friends
		WHERE user1_id=$1 OR user2_id=$1
	`
	rows, err := DB.Query(ctx, q, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var fs []models.Friend
	for rows.Next() {
		var f models.Friend
		err := rows.Scan(&f.User1ID, &f.User2ID, &f.Status)
		if err != nil {
			return nil, err
		}
		fs = append(fs, f)
	}
	return fs, nil
}

// FriendListEntry is a friend relationship resolved from one side's perspective:
// CounterpartID is always "the other user" relative to the userID passed to
// ListFriendsEnriched, never the caller themselves.
type FriendListEntry struct {
	CounterpartID uuid.UUID
	Username      string
	Status        string
}

// ListFriendsEnriched returns userID's friend relationships (pending or accepted)
// resolved from userID's perspective: CounterpartID is always the other party, with
// their username joined in from the users table.
func ListFriendsEnriched(ctx context.Context, userID uuid.UUID) ([]FriendListEntry, error) {
	q := `
		SELECT
			CASE WHEN f.user1_id = $1 THEN f.user2_id ELSE f.user1_id END AS counterpart_id,
			u.username,
			f.status
		FROM friends f
		JOIN users u ON u.id = CASE WHEN f.user1_id = $1 THEN f.user2_id ELSE f.user1_id END
		WHERE f.user1_id = $1 OR f.user2_id = $1
	`
	rows, err := DB.Query(ctx, q, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []FriendListEntry
	for rows.Next() {
		var e FriendListEntry
		if err := rows.Scan(&e.CounterpartID, &e.Username, &e.Status); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// RemoveFriend hard deletes the friend relation
func RemoveFriend(ctx context.Context, user1, user2 uuid.UUID) error {
	q := `
		DELETE FROM friends
		WHERE (user1_id=$1 AND user2_id=$2)
		   OR (user1_id=$2 AND user2_id=$1)
	`
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, q, user1, user2)
		return err
	})
}
