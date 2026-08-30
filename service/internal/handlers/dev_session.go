// internal/handlers/dev_session.go
package handlers

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// DevAccountsEnvVar gates the dev identity endpoints. Unset (the production
// default) leaves /dev/session unregistered, so the path 404s exactly like any
// other unknown path and the service exposes no way to mint an identity for an
// account nobody logged into (cambia-1149).
const DevAccountsEnvVar = "CAMBIA_DEV_ACCOUNTS"

// DevAccountEmailDomain is the email suffix that marks a user row as a dev
// switcher account. It is the account's key: names are upserted by
// <name>@dev.cambia.local, and the listing is every user whose email ends here.
// The domain is reserved (.local is never publicly resolvable), so a real
// signup can never collide with one.
const DevAccountEmailDomain = "@dev.cambia.local"

// devAccountNamePattern is the accepted account name, matching the client-side
// validation in the switcher: lowercase, digits, dash, underscore, 1-32 chars.
// Keeping it narrow keeps the derived email address and username predictable.
var devAccountNamePattern = regexp.MustCompile(`^[a-z0-9_-]{1,32}$`)

// DevAccountsEnabled reports whether the dev identity endpoints are switched on.
func DevAccountsEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(DevAccountsEnvVar))) {
	case "1", "true":
		return true
	}
	return false
}

// RegisterDevRoutes wires the dev identity endpoints onto mux when
// DevAccountsEnabled, and reports whether it did. Registration, not a runtime
// check inside the handler, is the gate: with the flag unset the mux has no
// route at all for /dev/session, so a probe cannot tell a service with the
// feature switched off from one that never shipped it.
func RegisterDevRoutes(mux *http.ServeMux) bool {
	if !DevAccountsEnabled() {
		return false
	}
	mux.HandleFunc("/dev/session", DevSessionHandler)
	return true
}

// devSessionRequest is the POST /dev/session body. An absent or empty name
// means "mint a fresh guest".
type devSessionRequest struct {
	Name string `json:"name"`
}

// devSessionResponse is the POST /dev/session response: the token the caller
// pins to its tab, plus the identity behind it. No cookie is ever set.
type devSessionResponse struct {
	Token string        `json:"token"`
	User  sanitizedUser `json:"user"`
}

// devSessionAccount is one entry of the GET /dev/session listing.
type devSessionAccount struct {
	Name string    `json:"name"`
	ID   uuid.UUID `json:"id"`
}

// devSessionListResponse is the GET /dev/session response.
type devSessionListResponse struct {
	Enabled  bool                `json:"enabled"`
	Accounts []devSessionAccount `json:"accounts"`
}

// DevSessionHandler serves the dev identity endpoints, registered only by
// RegisterDevRoutes:
//
//	POST /dev/session {"name":"alice"} - upsert the dev account and return a
//	  token for it; {} or {"name":""} mints a fresh guest instead.
//	GET  /dev/session - list the existing dev accounts.
//
// Neither response sets a cookie: the token is for one browser tab to hold, and
// writing a cookie would move every other tab on the origin with it.
func DevSessionHandler(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		devSessionList(w, r)
	case http.MethodPost:
		devSessionMint(w, r)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// devSessionMint upserts the named dev account (or mints a guest when no name
// is given) and returns a token for it.
func devSessionMint(w http.ResponseWriter, r *http.Request) {
	var req devSessionRequest
	// An entirely empty body is the documented "mint a guest" form, so EOF is
	// not an error here; anything else that fails to decode is.
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid request payload", http.StatusBadRequest)
		return
	}

	name := strings.TrimSpace(req.Name)

	var (
		user  models.User
		token string
		err   error
	)
	if name == "" {
		var guest models.User
		guest, token, err = newEphemeralUser(r.Context())
		user = guest
	} else {
		if !devAccountNamePattern.MatchString(name) {
			http.Error(w, "invalid account name: expected 1-32 characters from [a-z0-9_-]", http.StatusBadRequest)
			return
		}
		var account *models.User
		account, err = upsertDevAccount(r.Context(), name)
		if err == nil {
			user = *account
			token, err = auth.CreateJWT(user.ID.String())
		}
	}
	if err != nil {
		log.Printf("dev session: failed to mint a session for %q: %v", name, err)
		http.Error(w, "failed to create dev session", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(devSessionResponse{Token: token, User: sanitizeUser(&user)}); err != nil {
		log.Printf("dev session: failed to write response for %q: %v", name, err)
	}
}

// devSessionList returns the dev accounts that exist, so the switcher can offer
// them without the operator having to remember which names were used.
func devSessionList(w http.ResponseWriter, r *http.Request) {
	users, err := database.ListUsersByEmailSuffix(r.Context(), DevAccountEmailDomain)
	if err != nil {
		log.Printf("dev session: failed to list dev accounts: %v", err)
		http.Error(w, "failed to list dev accounts", http.StatusInternalServerError)
		return
	}

	accounts := make([]devSessionAccount, 0, len(users))
	for _, u := range users {
		accounts = append(accounts, devSessionAccount{
			Name: strings.TrimSuffix(u.Email, DevAccountEmailDomain),
			ID:   u.ID,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(devSessionListResponse{Enabled: true, Accounts: accounts}); err != nil {
		log.Printf("dev session: failed to write listing: %v", err)
	}
}

// upsertDevAccount returns the persistent user behind a dev account name,
// creating it on first use. Idempotent per name: the account is keyed by its
// derived email address, so repeat calls resolve to the same row and the same
// user id, which is what lets a browser verifier ask for "alice" across runs
// and get the same player.
//
// The row is created with a random password nobody holds, so the login form
// cannot reach a dev account: pinning one is the only way in, and that path
// exists only while DevAccountsEnabled.
func upsertDevAccount(ctx context.Context, name string) (*models.User, error) {
	email := name + DevAccountEmailDomain

	existing, err := database.GetUserByEmail(ctx, email)
	if err == nil {
		return existing, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return nil, fmt.Errorf("failed to look up dev account %q: %w", name, err)
	}

	password, err := unusablePassword()
	if err != nil {
		return nil, err
	}
	user := models.User{
		Email:       email,
		Password:    password, // Hashed by database.CreateUser.
		Username:    name,
		IsEphemeral: false,
	}
	if err := database.CreateUser(ctx, &user); err != nil {
		// A concurrent first use of the same name loses the insert race on the
		// email unique constraint; the winner's row is the account.
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" {
			return database.GetUserByEmail(ctx, email)
		}
		return nil, fmt.Errorf("failed to create dev account %q: %w", name, err)
	}
	log.Printf("dev session: created dev account %q (%s)", name, user.ID)
	return &user, nil
}

// unusablePassword returns a password nobody knows: 32 bytes of crypto/rand,
// discarded after hashing. It exists only so the stored hash is well formed.
func unusablePassword() (string, error) {
	buf := make([]byte, 32)
	if _, err := rand.Read(buf); err != nil {
		return "", fmt.Errorf("failed to generate dev account password: %w", err)
	}
	return base64.RawURLEncoding.EncodeToString(buf), nil
}
