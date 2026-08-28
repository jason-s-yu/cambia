// internal/handlers/matchmaking.go
package handlers

import (
	"log"
	"time"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/hub"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// matchNoticeSendTimeout bounds how long a match notice waits for a hub's Run loop. The channel
// is buffered, so a live hub takes it immediately; the bound exists so a hub that stopped
// serving between the match and this send cannot park the matchmaker's goroutine.
const matchNoticeSendTimeout = 2 * time.Second

// HandleMatchFormed is the Matchmaker's OnMatchFormed callback: it turns a formed match into
// one playable lobby.
//
// A match is assembled from one lobby per party, and only one of them can hold the game. The
// matchmaker names that one (HostLobbyID); this moves every other party's members into it as
// members, and tells every participating hub which lobby the match is played in so its clients
// go there. Before cambia-933 only the host hub was notified, and the second party was left
// sitting in its own lobby with no way to reach the match.
//
// Runs on the matchmaker's goroutine, so it touches only lock-guarded lobby state and the hubs'
// channel API, never hub fields.
func (gs *GameServer) HandleMatchFormed(result matchmaking.MatchResult) {
	hostLob, ok := gs.LobbyStore.GetLobby(result.HostLobbyID)
	if !ok {
		log.Printf("Match formed in queue %s but host lobby %s is gone", result.QueueID, result.HostLobbyID)
		return
	}

	hostLob.Mu.Lock()
	hostUserID := hostLob.HostUserID
	hostLob.Mu.Unlock()

	// One entry per party lobby, in the matchmaker's order, so the host lobby's own members
	// come first and seating stays deterministic.
	lobbyIDs := make([]uuid.UUID, 0, len(result.Players))
	seen := make(map[uuid.UUID]bool, len(result.Players))
	for _, p := range result.Players {
		if p.LobbyID == uuid.Nil || seen[p.LobbyID] {
			continue
		}
		seen[p.LobbyID] = true
		lobbyIDs = append(lobbyIDs, p.LobbyID)
	}

	players := make([]hub.MatchedPlayer, 0, len(lobbyIDs))
	for _, lobbyID := range lobbyIDs {
		lob, exists := gs.LobbyStore.GetLobby(lobbyID)
		if !exists {
			log.Printf("Match in queue %s: party lobby %s is gone; skipping it", result.QueueID, lobbyID)
			continue
		}

		// JoinedUsers takes the lobby's own lock; the InviteUser below is the one that needs
		// the caller to hold it.
		members := lob.JoinedUsers()

		for _, uid := range members {
			username := ""
			if partyHub, hasHub := gs.HubStore.GetHub(lobbyID); hasHub {
				username = partyHub.UsernameOf(uid)
			}
			players = append(players, hub.MatchedPlayer{
				UserID:   uid,
				Username: username,
				IsHost:   uid == hostUserID,
			})

			// Membership of the match lobby is what the WebSocket upgrade, the lobby roster and
			// the ready check all run off, so an incoming player is invited into it before they
			// are told to connect. Their own connection promotes the invite to a join.
			if lobbyID != result.HostLobbyID {
				hostLob.Mu.Lock()
				hostLob.InviteUser(uid)
				hostLob.Mu.Unlock()
			}
		}
	}

	notice := hub.MatchNotice{LobbyID: result.HostLobbyID, Players: players}
	for _, lobbyID := range lobbyIDs {
		partyHub, hasHub := gs.HubStore.GetHub(lobbyID)
		if !hasHub || !partyHub.Alive() {
			log.Printf("Match in queue %s: no live hub for party lobby %s; its players were not notified", result.QueueID, lobbyID)
			continue
		}
		select {
		case partyHub.Matched() <- notice:
		case <-time.After(matchNoticeSendTimeout):
			log.Printf("Match in queue %s: hub %s did not take the match notice within %s", result.QueueID, lobbyID, matchNoticeSendTimeout)
		}
	}
}
