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
		SELECT user1_id, user2_id, status, updated_at
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
