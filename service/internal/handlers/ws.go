// internal/handlers/ws.go
package handlers

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/coder/websocket"
	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/database"
	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/wsopts"
	"github.com/sirupsen/logrus"
)

// HubWSHandler is the unified WebSocket endpoint for /ws/{lobbyId}.
// It upgrades the HTTP connection, authenticates the user, looks up the hub,
// creates a Connection, and spawns read/write pumps.
func HubWSHandler(logger *logrus.Logger, gs *GameServer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// 1. Extract lobby ID from path: /ws/{lobbyId}
		pathParts := strings.Split(strings.TrimPrefix(r.URL.Path, "/ws/"), "/")
		if len(pathParts) < 1 || pathParts[0] == "" {
			http.Error(w, "missing lobbyId", http.StatusBadRequest)
			return
		}
		lobbyID, err := uuid.Parse(pathParts[0])
		if err != nil {
			http.Error(w, "invalid lobbyId", http.StatusBadRequest)
			return
		}

		// 2. Look up the lobby, then its hub. The lobby store is the authority on whether this
		// URL still means anything: a teardown deletes the lobby before it stops the hub, and a
		// hub deregisters itself the moment its Run loop exits (cambia-808), so a lobby id that
		// resolves to a registered hub here is one that can still be served. A stale URL gets a
		// 404 refusal instead of an accepted socket nobody answers.
		lob, lobbyExists := gs.LobbyStore.GetLobby(lobbyID)
		h, hubExists := gs.HubStore.GetHub(lobbyID)
		if !lobbyExists || !hubExists {
			http.Error(w, "lobby not found", http.StatusNotFound)
			return
		}

		// 3. Authenticate user (before upgrade so we can return proper HTTP errors)
		userID, err := EnsureEphemeralUser(w, r)
		if err != nil {
			logger.Warnf("ws: auth failed for lobby %s: %v", lobbyID, err)
			http.Error(w, "authentication failed", http.StatusUnauthorized)
			return
		}

		// 4. Check private lobby access: membership is the whole gate. The host is auto-invited at
		// creation (cambia-771), so they hold an entry from the moment the lobby exists; once they
		// deliberately leave they hold none, and the host role has already moved to a remaining
		// member (cambia-835). cambia-807 let the host past this gate regardless of the map, to
		// keep a lobby whose HostUserID was orphaned on a departed host reachable by its owner;
		// with migration in place that allowance can only re-admit somebody who left.
		lob.Mu.Lock()
		_, isInUsers := lob.Users[userID]
		lobType := lob.Type
		lob.Mu.Unlock()

		if lobType == "private" && !isInUsers {
			http.Error(w, "not invited to private lobby", http.StatusForbidden)
			return
		}

		// 5. Upgrade WebSocket with subprotocol "cambia". A tab-pinned client also offers a
		// cambia-token.<jwt> entry (resolved in step 3, above the upgrade); it is never
		// selected, so the negotiated protocol is "cambia" either way.
		c, err := websocket.Accept(w, r, wsopts.AcceptOptions(wsopts.Subprotocol))
		if err != nil {
			logger.Warnf("ws: accept error for lobby %s: %v", lobbyID, err)
			return
		}
		defer c.Close(websocket.StatusInternalError, "handler exit")

		// 6. Fetch username (fallback to short UUID prefix)
		username := hubFetchUsername(r.Context(), userID)

		// 7. Connecting joins the lobby: it upgrades an invite to a joined membership and is how
		// a public lobby is entered without a prior POST /lobby/{id}/join. A client that has
		// just left must therefore close its socket before releasing membership, or its own
		// reconnect would hand the membership straight back (see the web client's leave path).
		lob.Mu.Lock()
		lob.MarkJoinedUnsafe(userID)
		lob.Mu.Unlock()

		ctx, cancel := context.WithCancel(r.Context())
		connID, _ := uuid.NewRandom()
		conn := hub.NewConnection(connID, userID, username, c, cancel)

		// 8. Register with hub
		h.Join(conn)
		logger.Infof("ws: user %s connected to hub %s", userID, lobbyID)

		// 9. Spawn write pump
		go conn.WritePump(ctx)

		// 10. Run read pump (blocks until disconnect or context done)
		conn.ReadPump(ctx, h.Incoming())

		// 11. Cleanup. Connection-level only: the user stays a lobby member, so a dropped socket
		// or a closed tab can reconnect and still shows up in GET /lobby/active (cambia-783).
		// Only POST /lobby/{id}/leave releases membership.
		//
		// Addressed to this connection, not to the user. A second tab, or a reconnect that beat
		// this socket's own read failure, has already displaced the hub's entry for this user;
		// a user-addressed leave here would evict and close that live socket instead, and the
		// disconnect grace it armed would expire into a forfeit (cambia-1543).
		logger.Infof("ws: user %s disconnected from hub %s", userID, lobbyID)
		h.LeaveConn(conn)
		cancel()
	}
}

// hubFetchUsername retrieves the username for a user ID from the database, falling back to a
// short UUID prefix on error. Bounded by both ctx and an internal 3s cap (the shorter of the two
// wins), so a caller's own deadline (a dropped client, a request timeout) is honored rather than
// this call outliving it.
func hubFetchUsername(ctx context.Context, userID uuid.UUID) string {
	// No database configured (or unreachable): fall back to a short UUID prefix rather than
	// dereferencing a nil pool, which would panic the WS handler.
	if database.DB == nil {
		return "User_" + userID.String()[:4]
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	user, err := database.GetUserByID(ctx, userID)
	if err != nil {
		return "User_" + userID.String()[:4]
	}
	return user.Username
}
