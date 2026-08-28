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

	"github.com/google/uuid"
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
	// Connect to the test database exactly once for the whole package (cambia-908), skipping
	// cleanly up front if no DB is reachable (ensure .env points to a test DB).
	ensureTestDB(t)
}

// Isolation comes from per-run fixture rows and their registered cleanups, not from clearing
// tables: the `DELETE FROM friends; DELETE FROM users;` helper that used to sit here (unused,
// commented out at both call sites) would have wiped every account in the shared dev database,
// the same class of over-deletion cambia-942 F5 fixes in createTestUser below.

// createTestUser is a helper to create a user directly in the database for testing. Registers
// a t.Cleanup deleting the created row (and anything a test drove it to accumulate: hosted
// lobbies/games/game_results, ratings, friends) so repeated runs against the shared dev DB do
// not grow those tables without bound (cambia-890 F4).
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
		t.Cleanup(func() { cleanupTestUserRows(t, u.ID) })
	} else {
		// The email is already taken, so this call created nothing: reuse the existing row and
		// register no cleanup for it (cambia-942 F5). Registering one here made the suite delete
		// an account it did not create - along with its lobbies, games, game_results and ratings,
		// which cleanupTestUserRows cascades - whenever a real account happened to hold a fixture
		// address; the fixed alice@example.com / bob@example.com and lb-*@example.com pairs made
		// that reachable, and every caller now passes a per-run address instead. The row is left
		// behind rather than deleted, so this branch is a last-resort path, not a cleanup route.
		existingUser, fetchErr := database.GetUserByEmail(ctx, email)
		require.NoError(t, fetchErr, "Failed to fetch existing user")
		require.NotNil(t, existingUser, "Existing user should not be nil")
		t.Logf("reusing pre-existing user %s (%s); not registering cleanup for a row this test did not create", uname, existingUser.ID)
		return *existingUser
	}
	return u
}

// TestFriendFlow is an integration test covering the friend request -> accept -> list flow.
func TestFriendFlow(t *testing.T) {
	setupFriendTest(t)

	// 1. Create two users.
	// Per-run email addresses, for the reason in createTestUser's fallback branch: the fixed
	// alice@example.com / bob@example.com pair collided with whatever account already held those
	// addresses, and the test then ran against - and used to delete - rows it never created
	// (cambia-942 F5). The usernames stay fixed; the assertions below read those.
	userAlice := createTestUser(t, "alice-"+uuid.NewString()+"@example.com", "password123", "alice")
	userBob := createTestUser(t, "bob-"+uuid.NewString()+"@example.com", "password456", "bob")

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

	var friendsListBob []FriendListRow
	err = json.Unmarshal(listRecorderBob.Body.Bytes(), &friendsListBob)
	require.NoError(t, err, "Failed to decode Bob's friend list response")
	require.Len(t, friendsListBob, 1, "Bob should have 1 accepted friend relationship")
	assert.Equal(t, "accepted", friendsListBob[0].Status)
	// Resolved from Bob's perspective: userId is the counterpart (Alice).
	assert.Equal(t, userAlice.ID, friendsListBob[0].UserID)
	assert.Equal(t, "alice", friendsListBob[0].Username)

	// 7. Verify the relationship for Alice as well.
	listReqAlice := httptest.NewRequest("GET", "/friends/list", nil)
	listReqAlice.Header.Set("Cookie", "auth_token="+aliceToken)
	listRecorderAlice := httptest.NewRecorder()
	ListFriendsHandler(listRecorderAlice, listReqAlice)
	require.Equal(t, http.StatusOK, listRecorderAlice.Code, "ListFriends for Alice failed: %s", listRecorderAlice.Body.String())

	var friendsListAlice []FriendListRow
	err = json.Unmarshal(listRecorderAlice.Body.Bytes(), &friendsListAlice)
	require.NoError(t, err, "Failed to decode Alice's friend list response")
	require.Len(t, friendsListAlice, 1, "Alice should have 1 accepted friend relationship")
	assert.Equal(t, "accepted", friendsListAlice[0].Status)
	// Resolved from Alice's perspective: userId is the counterpart (Bob).
	assert.Equal(t, userBob.ID, friendsListAlice[0].UserID)
	assert.Equal(t, "bob", friendsListAlice[0].Username)

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
	var friendsListBobAfterRemove []FriendListRow
	err = json.Unmarshal(listRecorderBobAfterRemove.Body.Bytes(), &friendsListBobAfterRemove)
	require.NoError(t, err)
	assert.Empty(t, friendsListBobAfterRemove, "Bob should have no friends after removal")

	listReqAliceAfterRemove := httptest.NewRequest("GET", "/friends/list", nil)
	listReqAliceAfterRemove.Header.Set("Cookie", "auth_token="+aliceToken)
	listRecorderAliceAfterRemove := httptest.NewRecorder()
	ListFriendsHandler(listRecorderAliceAfterRemove, listReqAliceAfterRemove)
	require.Equal(t, http.StatusOK, listRecorderAliceAfterRemove.Code)
	var friendsListAliceAfterRemove []FriendListRow
	err = json.Unmarshal(listRecorderAliceAfterRemove.Body.Bytes(), &friendsListAliceAfterRemove)
	require.NoError(t, err)
	assert.Empty(t, friendsListAliceAfterRemove, "Alice should have no friends after removal")
}
