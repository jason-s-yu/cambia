// internal/handlers/user.go
package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// newGuestUsername derives a short, readable, per-user guest display name from a freshly
// generated user id, e.g. "Guest-A1B2C3D4". The id itself guarantees uniqueness (it is the
// user's primary key), but the displayed name is a truncated 32-bit slice of it (8 hex chars)
// and users.username carries no UNIQUE constraint, so two guests CAN show the same label: by
// the birthday approximation (n ~= 1.18 * sqrt(2^32)), a 50% chance of at least one collision
// arrives around 77,000 guest rows, and a 1% chance around 9,300. That is an acceptable
// display-only collision rate for a lobby-scale app; it is not a substitute for a real
// uniqueness guarantee (cambia-890 - every ephemeral user previously stored the fixed string
// "Guest", so a table of guests rendered identical labels in every seat, in the lobby list,
// chat, and results).
func newGuestUsername(id uuid.UUID) string {
	hex := strings.ReplaceAll(id.String(), "-", "")
	return "Guest-" + strings.ToUpper(hex[:8])
}

// newEphemeralUser creates a guest user row and mints its JWT. It writes no
// cookie and touches no response, so it is the single mint point shared by the
// cookie bootstrap (EnsureEphemeralUser, which sets the cookie itself) and the
// tab-scoped paths that must not touch the shared cookie jar at all: POST
// /user/guest with X-Cambia-Session: tab, and POST /dev/session with no name
// (cambia-1149).
func newEphemeralUser(ctx context.Context) (models.User, string, error) {
	// The id is generated here, rather than left for database.CreateUser to assign, so a
	// unique guest username can be derived from it and persisted in the same INSERT
	// (CreateUser only generates an id itself when the passed-in one is uuid.Nil).
	id, err := uuid.NewRandom()
	if err != nil {
		return models.User{}, "", fmt.Errorf("failed to generate ephemeral user id: %w", err)
	}
	ephemeralUser := models.User{
		ID:          id,
		Email:       "", // Ephemeral users don't have email/password initially.
		Password:    "",
		Username:    newGuestUsername(id),
		IsEphemeral: true,
	}
	if err := database.CreateUser(ctx, &ephemeralUser); err != nil {
		return models.User{}, "", fmt.Errorf("failed to create ephemeral user: %w", err)
	}
	token, err := auth.CreateJWT(ephemeralUser.ID.String())
	if err != nil {
		// Attempt to clean up the created user if JWT creation fails? Complex.
		return models.User{}, "", fmt.Errorf("failed to create JWT for ephemeral user: %w", err)
	}
	return ephemeralUser, token, nil
}

// ErrExplicitTokenInvalid reports that the caller sent a token of its own
// (Authorization: Bearer, or a cambia-token.<jwt> handshake entry) that did not
// verify. Callers answer 401 rather than minting a guest: the caller asked to
// be somebody specific, and quietly handing back a different identity would
// leave a tab holding a stale token looking signed in as a stranger
// (cambia-1149).
var ErrExplicitTokenInvalid = errors.New("the token on the request did not verify")

// EnsureEphemeralUser checks for a token the caller sent explicitly
// (Authorization: Bearer, or a cambia-token.<jwt> WebSocket handshake entry),
// then for an existing `auth_token` cookie.
// If one verifies, it authenticates the user and returns their UUID.
// If none is present, it creates a new ephemeral guest user, sets a new `auth_token` cookie,
// and returns the new guest user's UUID.
// An explicit token that does not verify is an error rather than a fresh guest:
// see auth.ResolveAuthToken for why a caller that sends one gets no fallback.
func EnsureEphemeralUser(w http.ResponseWriter, r *http.Request) (uuid.UUID, error) {
	// Helper function to create and set cookie for a new ephemeral user.
	createAndSetEphemeralUser := func() (uuid.UUID, error) {
		ephemeralUser, newToken, err := newEphemeralUser(context.Background())
		if err != nil {
			return uuid.Nil, err
		}
		auth.SetAuthTokenCookie(w, newToken, auth.TOKEN_EXPIRE_TIME_SEC)
		log.Printf("Created ephemeral user %s and set auth cookie.", ephemeralUser.ID)
		return ephemeralUser.ID, nil
	}

	// Resolve the request's credential: explicit token first, then any
	// auth_token cookie(s), accepting the first that verifies and self-healing
	// stale/invalid duplicates (see auth.ResolveAuthToken doc comment).
	userIDStr, sawAny, ok := auth.ResolveAuthToken(w, r)
	if !ok {
		if sawAny {
			// An explicit token that failed is a client error, not a reason to hand out a
			// second identity: the caller asked to be somebody specific.
			if _, explicit := auth.ExplicitToken(r); explicit {
				return uuid.Nil, ErrExplicitTokenInvalid
			}
			log.Printf("No auth_token cookie on the request verified. Creating new ephemeral user.")
		}
		// No token found (or none valid), create a new ephemeral user.
		return createAndSetEphemeralUser()
	}

	// Token is valid, parse the user ID.
	uuidVal, parseErr := uuid.Parse(userIDStr)
	if parseErr != nil {
		// User ID in token is not a valid UUID, this indicates a problem.
		// Treat as invalid token scenario.
		log.Printf("Invalid user ID format in token: %v. Creating new ephemeral user.", parseErr)
		return createAndSetEphemeralUser()
	}

	// Successfully authenticated existing user (could be ephemeral or persistent).
	return uuidVal, nil
}

// GuestHandler provisions an ephemeral guest session via REST (no WebSocket required).
// GET or POST /user/guest - if the caller already holds a valid credential (an explicit
// token or an auth_token cookie), returns the existing user; otherwise creates a
// new ephemeral user and sets the cookie. An explicit token that does not verify
// is a 401, not a new guest.
//
// With the request header X-Cambia-Session: tab the response is tab-scoped
// instead (cambia-1149): a fresh guest is always minted, its token is returned
// in the body as {"id":..., "token":...}, and no cookie is written. The caller
// asked for an identity that only this browser tab holds, so reusing the shared
// cookie identity would defeat the request, and writing a cookie would move
// every other tab on the origin to the new guest. This mode needs no dev flag,
// so tab guests work on any deployment.
func GuestHandler(w http.ResponseWriter, r *http.Request) {
	if auth.IsTabSession(r) {
		guest, token, err := newEphemeralUser(r.Context())
		if err != nil {
			log.Printf("GuestHandler: failed to create tab guest: %v", err)
			http.Error(w, "failed to create guest session", http.StatusInternalServerError)
			return
		}
		log.Printf("Created tab-scoped ephemeral user %s (no cookie set).", guest.ID)
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]string{
			"id":    guest.ID.String(),
			"token": token,
		}); err != nil {
			log.Printf("GuestHandler: failed to write tab guest response: %v", err)
		}
		return
	}

	userID, err := EnsureEphemeralUser(w, r)
	if err != nil {
		if errors.Is(err, ErrExplicitTokenInvalid) {
			http.Error(w, "invalid authentication token", http.StatusUnauthorized)
			return
		}
		log.Printf("GuestHandler: failed to ensure ephemeral user: %v", err)
		http.Error(w, "failed to create guest session", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"id": userID.String()})
}

// ClaimEphemeralHandler handles requests to convert an ephemeral user account
// into a persistent one by adding email and password.
// Note: This handler is defined but not currently routed in main.go.
// If needed, a route like POST /user/claim should be added.
type claimEphemeralRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
	Username string `json:"username"` // Optional: Allow updating username during claim.
}

func ClaimEphemeralHandler(w http.ResponseWriter, r *http.Request) {
	userIDStr, _, ok := auth.ResolveAuthToken(w, r)
	if !ok {
		http.Error(w, "Invalid or missing authentication token", http.StatusForbidden)
		return
	}
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		http.Error(w, "Invalid user ID format in token", http.StatusForbidden)
		return
	}

	// Fetch the user associated with the token.
	u, err := database.GetUserByID(r.Context(), userID)
	if err != nil {
		// If user not found, token might be stale or DB issue.
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}
	// Check if the user is actually ephemeral.
	if !u.IsEphemeral {
		http.Error(w, "Account has already been claimed", http.StatusBadRequest)
		return
	}

	// Decode the request payload.
	var req claimEphemeralRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}
	// Basic validation for required fields.
	if req.Email == "" || req.Password == "" {
		http.Error(w, "Email and password are required to claim an account", http.StatusBadRequest)
		return
	}

	// Update user details.
	u.Email = req.Email
	u.Password = req.Password // Will be re-hashed by UpdateUserCredentials.
	if req.Username != "" {
		u.Username = req.Username // Update username if provided.
	}
	u.IsEphemeral = false // Mark as persistent.

	// Persist changes to the database.
	err = database.UpdateUserCredentials(r.Context(), u)
	if err != nil {
		// Handle potential constraint violations (e.g., email conflict).
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" { // Unique violation.
			http.Error(w, "Email address is already in use", http.StatusConflict)
			return
		}
		log.Printf("Failed to finalize ephemeral user %s: %v", userID, err)
		http.Error(w, "Failed to claim account", http.StatusInternalServerError)
		return
	}

	// Optionally issue a new token if claims need updating, though not strictly necessary here.

	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "Account claimed successfully.") // Simple confirmation message.
}

// LogoutHandler clears the auth_token cookie, effectively logging the user out.
func LogoutHandler(w http.ResponseWriter, r *http.Request) {
	auth.ExpireAuthTokenCookie(w)
	w.WriteHeader(http.StatusOK)
}

// CreateUserHandler handles new user registration requests.
// It expects email, password, and username in the JSON payload.
func CreateUserHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Email    string `json:"email"`
		Password string `json:"password"`
		Username string `json:"username"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}
	// Basic validation.
	if req.Email == "" || req.Password == "" || req.Username == "" {
		http.Error(w, "Email, password, and username are required", http.StatusBadRequest)
		return
	}

	user := models.User{
		Email:       req.Email,
		Password:    req.Password, // Will be hashed by database.CreateUser.
		Username:    req.Username,
		IsEphemeral: false, // New users created via this endpoint are persistent.
		IsAdmin:     false, // Default to non-admin.
	}

	ctx := r.Context()
	err := database.CreateUser(ctx, &user) // Creates user and hashes password.
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" { // Unique constraint violation.
			// Check if the violation is on the email field.
			if strings.Contains(pgErr.ConstraintName, "email") {
				http.Error(w, "Email address already exists", http.StatusConflict)
			} else {
				// Handle other unique constraints if any.
				http.Error(w, "Username or other field already exists", http.StatusConflict)
			}
			return
		}
		// Log the specific error for debugging.
		log.Printf("Error creating user %s: %v", req.Email, err)
		http.Error(w, "Error creating user account", http.StatusInternalServerError)
		return
	}

	// Return the created user object (excluding password).
	user.Password = "" // Clear password before encoding response.
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

// loginRequest defines the expected JSON structure for login attempts.
type loginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

// loginResponse defines the JSON structure returned upon successful login.
type loginResponse struct {
	Token string `json:"token"`
}

// LoginHandler handles user login requests.
// It authenticates the user based on email and password, generates a JWT,
// sets it as an HttpOnly cookie, and returns the token in the response body.
//
// With the request header X-Cambia-Session: tab the Set-Cookie is skipped
// (cambia-1149): the caller pins the returned token to one browser tab, so the
// shared cookie identity of the other tabs stays as it was. The response body
// is identical either way.
func LoginHandler(w http.ResponseWriter, r *http.Request) {
	var req loginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request payload", http.StatusBadRequest)
		return
	}
	if req.Email == "" || req.Password == "" {
		http.Error(w, "Email and password are required", http.StatusBadRequest)
		return
	}

	// AuthenticateUser verifies credentials and returns a JWT if valid.
	token, err := database.AuthenticateUser(context.Background(), req.Email, req.Password)
	if err != nil {
		// Log the failure reason but return a generic forbidden status.
		log.Printf("Authentication failed for user %s: %v", req.Email, err)
		http.Error(w, "Invalid email or password", http.StatusForbidden)
		return
	}

	// Set the JWT as an HttpOnly cookie. Secure/SameSite come from COOKIE_SECURE.
	// A tab-scoped login skips the cookie entirely and carries the token in the
	// body alone.
	if !auth.IsTabSession(r) {
		auth.SetAuthTokenCookie(w, token, auth.TOKEN_EXPIRE_TIME_SEC)
	}

	// Return the token in the response body as well.
	resp := loginResponse{Token: token}
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		// Log internal error if response writing fails.
		log.Printf("Failed to write login response for user %s: %v", req.Email, err)
		http.Error(w, "Failed to process login response", http.StatusInternalServerError)
		return
	}
}

// sanitizedUser is the public projection of a user row: identity and flags, no
// password hash and no email. Shared by GET /user/me and POST /dev/session so
// the two cannot drift.
type sanitizedUser struct {
	ID          uuid.UUID `json:"id"`
	Username    string    `json:"username"`
	IsEphemeral bool      `json:"is_ephemeral"`
	IsAdmin     bool      `json:"is_admin"`
	// Add other non-sensitive fields like Elo ratings if needed by the client.
	// Elo1v1      int       `json:"elo_1v1"`
}

// sanitizeUser projects a user row onto the fields safe to return to a client.
func sanitizeUser(u *models.User) sanitizedUser {
	return sanitizedUser{
		ID:          u.ID,
		Username:    u.Username,
		IsEphemeral: u.IsEphemeral,
		IsAdmin:     u.IsAdmin,
	}
}

// MeHandler retrieves and returns basic information about the currently authenticated user.
// It relies on a token the caller sent explicitly, or the `auth_token` cookie,
// being present and valid.
func MeHandler(w http.ResponseWriter, r *http.Request) {
	userIDStr, _, ok := auth.ResolveAuthToken(w, r) // Verifies token validity.
	if !ok {
		http.Error(w, "Invalid or missing authentication token", http.StatusForbidden)
		return
	}
	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		http.Error(w, "Invalid user ID format in token", http.StatusForbidden) // Should not happen with valid JWT.
		return
	}

	// Fetch user details from the database.
	user, err := database.GetUserByID(r.Context(), userID)
	if err != nil {
		// If user not found, the token might be for a deleted user.
		log.Printf("User %s from valid token not found in DB: %v", userID, err)
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	// Prepare a sanitized response object excluding sensitive fields like password hash.
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(sanitizeUser(user)); err != nil {
		log.Printf("Failed to write /user/me response for user %s: %v", userID, err)
		http.Error(w, "Failed to process user information", http.StatusInternalServerError)
		return
	}
}
