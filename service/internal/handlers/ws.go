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

		// 2. Look up hub from HubStore
		h, exists := gs.HubStore.GetHub(lobbyID)
		if !exists {
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

		// 4. Check private lobby access
		lob := h.Lobby
		lob.Mu.Lock()
		_, isInUsers := lob.Users[userID]
		lobType := lob.Type
		isHost := lob.HostUserID == userID
		lob.Mu.Unlock()

		if lobType == "private" && !isInUsers {
			http.Error(w, "not invited to private lobby", http.StatusForbidden)
			return
		}

		// 5. Upgrade WebSocket with subprotocol "cambia"
		c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
			Subprotocols:   []string{"cambia"},
			OriginPatterns: []string{"*"},
		})
		if err != nil {
			logger.Warnf("ws: accept error for lobby %s: %v", lobbyID, err)
			return
		}
		defer c.Close(websocket.StatusInternalError, "handler exit")

		// 6. Fetch username (fallback to short UUID prefix)
		username := hubFetchUsername(userID)

		// 7. Add user to lobby's Users map if not present, then create Connection
		lob.Mu.Lock()
		if lob.Users == nil {
			lob.Users = make(map[uuid.UUID]bool)
		}
		lob.Users[userID] = true
		lob.Mu.Unlock()

		ctx, cancel := context.WithCancel(r.Context())
		connID, _ := uuid.NewRandom()
		conn := hub.NewConnection(connID, userID, username, isHost, c, cancel)

		// 8. Register with hub
		h.Join(conn)
		logger.Infof("ws: user %s connected to hub %s", userID, lobbyID)

		// 9. Spawn write pump
		go conn.WritePump(ctx)

		// 10. Run read pump (blocks until disconnect or context done)
		conn.ReadPump(ctx, h.Incoming())

		// 11. Cleanup
		logger.Infof("ws: user %s disconnected from hub %s", userID, lobbyID)
		h.Leave(userID)
		cancel()
	}
}

// hubFetchUsername retrieves the username for a user ID from the database,
// falling back to a short UUID prefix on error.
func hubFetchUsername(userID uuid.UUID) string {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	user, err := database.GetUserByID(ctx, userID)
	if err != nil {
		return "User_" + userID.String()[:4]
	}
	return user.Username
}
