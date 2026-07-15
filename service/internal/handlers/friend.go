// internal/handlers/friend.go
package handlers

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/database"
)

// authenticateAndGetUser performs JWT authentication and retrieves the user UUID.
// It handles common authentication errors and returns the UUID or writes an HTTP error.
func authenticateAndGetUser(w http.ResponseWriter, r *http.Request) (uuid.UUID, bool) {
	userIDStr, sawAny, ok := auth.ResolveAuthTokenCookie(w, r)
	if !ok {
		if !sawAny {
			http.Error(w, "Missing authentication token", http.StatusUnauthorized)
		} else {
			http.Error(w, "Invalid or expired authentication token", http.StatusForbidden)
		}
		return uuid.Nil, false
	}

	userUUID, err := uuid.Parse(userIDStr)
	if err != nil {
		// This indicates an issue with the JWT generation or a corrupted token.
		log.Printf("Error parsing user ID from valid token (%s): %v", userIDStr, err)
		http.Error(w, "Invalid user ID format in token", http.StatusForbidden)
		return uuid.Nil, false
	}

	return userUUID, true
}

// AddFriendHandler handles a user sending a friend request to another user.
// It creates a pending friend relationship in the database.
// Expects JSON: { "friend_id": "uuid-string" }
func AddFriendHandler(w http.ResponseWriter, r *http.Request) {
	userUUID, ok := authenticateAndGetUser(w, r)
	if !ok {
		return // Error already written by authenticateAndGetUser.
	}

	var req struct {
		FriendID string `json:"friend_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}
	friendUUID, err := uuid.Parse(req.FriendID)
	if err != nil {
		http.Error(w, "Invalid friend_id format", http.StatusBadRequest)
		return
	}

	if userUUID == friendUUID {
		http.Error(w, "Cannot add yourself as a friend", http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	err = database.InsertFriendRequest(ctx, userUUID, friendUUID)
	if err != nil {
		var pgErr *pgconn.PgError
		// Handle potential foreign key constraint violation if friend_id doesn't exist.
		if errors.As(err, &pgErr) && pgErr.Code == "23503" { // foreign_key_violation
			http.Error(w, "Target user does not exist", http.StatusNotFound)
			return
		}
		// Handle potential unique constraint violation if request already exists (even if pending).
		if errors.As(err, &pgErr) && pgErr.Code == "23505" { // unique_violation
			http.Error(w, "Friend request already sent or relationship exists", http.StatusConflict)
			return
		}
		log.Printf("Failed to insert friend request from %s to %s: %v", userUUID, friendUUID, err)
		http.Error(w, "Failed to send friend request", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusCreated)
	fmt.Fprintln(w, "Friend request sent successfully.")
}

// AcceptFriendHandler handles a user accepting a pending friend request.
// It updates the relationship status to 'accepted'.
// Expects JSON: { "friend_id": "uuid-string" } where friend_id is the sender of the request.
func AcceptFriendHandler(w http.ResponseWriter, r *http.Request) {
	userUUID, ok := authenticateAndGetUser(w, r) // The user accepting the request.
	if !ok {
		return
	}

	var req struct {
		FriendID string `json:"friend_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}
	friendUUID, err := uuid.Parse(req.FriendID) // The user who sent the request.
	if err != nil {
		http.Error(w, "Invalid friend_id format", http.StatusBadRequest)
		return
	}

	// The database function expects (sender, receiver).
	err = database.AcceptFriend(r.Context(), friendUUID, userUUID)
	if err != nil {
		// Check if the error indicates no pending request was found.
		if errors.Is(err, pgx.ErrNoRows) || strings.Contains(err.Error(), "no pending friend request found") {
			http.Error(w, "No pending friend request found from this user", http.StatusNotFound)
		} else {
			log.Printf("Failed to accept friend request from %s for user %s: %v", friendUUID, userUUID, err)
			http.Error(w, "Failed to accept friend request", http.StatusInternalServerError)
		}
		return
	}

	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "Friend request accepted.")
}

// FriendListRow is a single entry in the GET /friends/list response, resolved from
// the authenticated caller's perspective: UserID is always the counterpart, never
// the caller. Online presence is omitted: connection tracking (hub.Hub.conns) is
// keyed per-lobby, not per-user, so there is no O(1) global "is this user online"
// lookup available to join in here (see cambia-481 report).
type FriendListRow struct {
	UserID   uuid.UUID `json:"userId"`
	Username string    `json:"username"`
	Status   string    `json:"status"`
}

// ListFriendsHandler returns all friend relationships (pending or accepted) for the
// authenticated user, resolved to the counterpart's userId and username.
func ListFriendsHandler(w http.ResponseWriter, r *http.Request) {
	userUUID, ok := authenticateAndGetUser(w, r)
	if !ok {
		return
	}

	ctx := r.Context()
	friends, err := database.ListFriendsEnriched(ctx, userUUID)
	if err != nil {
		log.Printf("Failed to list friends for user %s: %v", userUUID, err)
		http.Error(w, "Failed to retrieve friends list", http.StatusInternalServerError)
		return
	}

	rows := make([]FriendListRow, 0, len(friends))
	for _, f := range friends {
		rows = append(rows, FriendListRow{UserID: f.CounterpartID, Username: f.Username, Status: f.Status})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(rows)
}

// RemoveFriendHandler handles removing a friend relationship or declining/canceling a request.
// It deletes the corresponding row from the friends table.
// Expects JSON: { "friend_id": "uuid-string" }
func RemoveFriendHandler(w http.ResponseWriter, r *http.Request) {
	userUUID, ok := authenticateAndGetUser(w, r)
	if !ok {
		return
	}

	var req struct {
		FriendID string `json:"friend_id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}
	friendUUID, err := uuid.Parse(req.FriendID)
	if err != nil {
		http.Error(w, "Invalid friend_id format", http.StatusBadRequest)
		return
	}

	err = database.RemoveFriend(r.Context(), userUUID, friendUUID)
	if err != nil {
		log.Printf("Failed to remove friend relationship between %s and %s: %v", userUUID, friendUUID, err)
		// Don't necessarily return 500 if the relationship didn't exist, 200 might be okay.
		// However, a DB error during delete is likely a 500.
		http.Error(w, "Failed to remove friend", http.StatusInternalServerError)
		return
	}
	w.WriteHeader(http.StatusOK)
	fmt.Fprintln(w, "Friend removed or request canceled/declined.")
}
