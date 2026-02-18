// internal/handlers/friend_test.go
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
	_ "github.com/joho/godotenv/autoload" // Load .env for database connection.
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setupFriendTest initializes the database and auth for friend tests.
func setupFriendTest(t *testing.T) {
	// Initialize authentication (generates keys).
	auth.Init()
	// Connect to the test database (ensure .env points to a test DB).
	database.ConnectDB()
	// Optional: Clean up tables before test if needed.
	// clearFriendTables(t)
}

// clearFriendTables is a helper to clear relevant tables (useful for isolated tests).
func clearFriendTables(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := database.DB.Exec(ctx, "DELETE FROM friends; DELETE FROM users;")
	require.NoError(t, err, "Failed to clear friend/user tables")
}

// createTestUser is a helper to create a user directly in the database for testing.
func createTestUser(t *testing.T, email, pass, uname string) models.User {
	u := models.User{
		Email:       email,
		Password:    pass, // Will be hashed by CreateUser.
		Username:    uname,
		IsEphemeral: false,
	}
	ctx := context.Background()
	err := database.CreateUser(ctx, &u)
	// Handle potential unique constraint errors if running tests multiple times without cleanup.
	if err != nil && !strings.Contains(err.Error(), "23505") { // Ignore unique violation.
		require.NoError(t, err, "CreateUser failed unexpectedly")
	} else if err == nil {
		t.Logf("Created test user %s (%s)", uname, u.ID)
	} else {
		// If user already exists from previous run, fetch them.
		existingUser, fetchErr := database.GetUserByEmail(ctx, email)
		require.NoError(t, fetchErr, "Failed to fetch existing user")
		require.NotNil(t, existingUser, "Existing user should not be nil")
		return *existingUser
	}
	return u
}

// TestFriendFlow is an integration test covering the friend request -> accept -> list flow.
func TestFriendFlow(t *testing.T) {
	setupFriendTest(t)
	// Optional: Defer cleanup if needed.
	// defer clearFriendTables(t)

	// 1. Create two users.
	userAlice := createTestUser(t, "alice@example.com", "password123", "alice")
	userBob := createTestUser(t, "bob@example.com", "password456", "bob")

	// 2. Generate JWT tokens for authentication.
	aliceToken, err := auth.CreateJWT(userAlice.ID.String())
	require.NoError(t, err, "Failed to create Alice's token")
	bobToken, err := auth.CreateJWT(userBob.ID.String())
	require.NoError(t, err, "Failed to create Bob's token")

	// 3. Alice sends a friend request to Bob.
	addReqBody := `{"friend_id":"` + userBob.ID.String() + `"}`
	addReq := httptest.NewRequest("POST", "/friends/add", bytes.NewBufferString(addReqBody))
	addReq.Header.Set("Cookie", "auth_token="+aliceToken)
	addRecorder := httptest.NewRecorder()
	AddFriendHandler(addRecorder, addReq)
	require.Equal(t, http.StatusCreated, addRecorder.Code, "AddFriend request failed: %s", addRecorder.Body.String())

	// 4. Verify the pending request exists (optional direct DB check or ListFriends).
	friendsBobBeforeAccept, err := database.ListFriends(context.Background(), userBob.ID)
	require.NoError(t, err, "Failed to list Bob's friends before accept")
	require.Len(t, friendsBobBeforeAccept, 1, "Bob should have 1 pending request")
	require.Equal(t, "pending", friendsBobBeforeAccept[0].Status)
	require.Equal(t, userAlice.ID, friendsBobBeforeAccept[0].User1ID) // Alice (sender) is user1.
	require.Equal(t, userBob.ID, friendsBobBeforeAccept[0].User2ID)   // Bob (receiver) is user2.

	// 5. Bob accepts the friend request from Alice.
	acceptReqBody := `{"friend_id":"` + userAlice.ID.String() + `"}` // Bob accepts Alice's request.
	acceptReq := httptest.NewRequest("POST", "/friends/accept", bytes.NewBufferString(acceptReqBody))
	acceptReq.Header.Set("Cookie", "auth_token="+bobToken)
	acceptRecorder := httptest.NewRecorder()
	AcceptFriendHandler(acceptRecorder, acceptReq)
	require.Equal(t, http.StatusOK, acceptRecorder.Code, "AcceptFriend request failed: %s", acceptRecorder.Body.String())

	// 6. Verify the relationship is now accepted using ListFriendsHandler.
	listReqBob := httptest.NewRequest("GET", "/friends/list", nil)
	listReqBob.Header.Set("Cookie", "auth_token="+bobToken)
	listRecorderBob := httptest.NewRecorder()
	ListFriendsHandler(listRecorderBob, listReqBob)
	require.Equal(t, http.StatusOK, listRecorderBob.Code, "ListFriends for Bob failed: %s", listRecorderBob.Body.String())

	var friendsListBob []models.Friend
	err = json.Unmarshal(listRecorderBob.Body.Bytes(), &friendsListBob)
	require.NoError(t, err, "Failed to decode Bob's friend list response")
	require.Len(t, friendsListBob, 1, "Bob should have 1 accepted friend relationship")
	assert.Equal(t, "accepted", friendsListBob[0].Status)
	assert.Equal(t, userAlice.ID, friendsListBob[0].User1ID)
	assert.Equal(t, userBob.ID, friendsListBob[0].User2ID)

	// 7. Verify the relationship for Alice as well.
	listReqAlice := httptest.NewRequest("GET", "/friends/list", nil)
	listReqAlice.Header.Set("Cookie", "auth_token="+aliceToken)
	listRecorderAlice := httptest.NewRecorder()
	ListFriendsHandler(listRecorderAlice, listReqAlice)
	require.Equal(t, http.StatusOK, listRecorderAlice.Code, "ListFriends for Alice failed: %s", listRecorderAlice.Body.String())

	var friendsListAlice []models.Friend
	err = json.Unmarshal(listRecorderAlice.Body.Bytes(), &friendsListAlice)
	require.NoError(t, err, "Failed to decode Alice's friend list response")
	require.Len(t, friendsListAlice, 1, "Alice should have 1 accepted friend relationship")
	assert.Equal(t, "accepted", friendsListAlice[0].Status)

	// 8. Alice removes Bob as a friend.
	removeReqBody := `{"friend_id":"` + userBob.ID.String() + `"}`
	removeReq := httptest.NewRequest("POST", "/friends/remove", bytes.NewBufferString(removeReqBody))
	removeReq.Header.Set("Cookie", "auth_token="+aliceToken)
	removeRecorder := httptest.NewRecorder()
	RemoveFriendHandler(removeRecorder, removeReq)
	require.Equal(t, http.StatusOK, removeRecorder.Code, "RemoveFriend request failed: %s", removeRecorder.Body.String())

	// 9. Verify the relationship is gone for both.
	listReqBobAfterRemove := httptest.NewRequest("GET", "/friends/list", nil)
	listReqBobAfterRemove.Header.Set("Cookie", "auth_token="+bobToken)
	listRecorderBobAfterRemove := httptest.NewRecorder()
	ListFriendsHandler(listRecorderBobAfterRemove, listReqBobAfterRemove)
	require.Equal(t, http.StatusOK, listRecorderBobAfterRemove.Code)
	var friendsListBobAfterRemove []models.Friend
	err = json.Unmarshal(listRecorderBobAfterRemove.Body.Bytes(), &friendsListBobAfterRemove)
	require.NoError(t, err)
	assert.Empty(t, friendsListBobAfterRemove, "Bob should have no friends after removal")

	listReqAliceAfterRemove := httptest.NewRequest("GET", "/friends/list", nil)
	listReqAliceAfterRemove.Header.Set("Cookie", "auth_token="+aliceToken)
	listRecorderAliceAfterRemove := httptest.NewRecorder()
	ListFriendsHandler(listRecorderAliceAfterRemove, listReqAliceAfterRemove)
	require.Equal(t, http.StatusOK, listRecorderAliceAfterRemove.Code)
	var friendsListAliceAfterRemove []models.Friend
	err = json.Unmarshal(listRecorderAliceAfterRemove.Body.Bytes(), &friendsListAliceAfterRemove)
	require.NoError(t, err)
	assert.Empty(t, friendsListAliceAfterRemove, "Alice should have no friends after removal")
}
