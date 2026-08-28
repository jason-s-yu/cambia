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
// A party whose members all closed their tabs is not seated: consolidating it would move the
// live players into a ready check the absent side can never answer, and nothing times that check
// out (cambia-933 F1). When that leaves the match short, no notice goes out at all and the
// surviving parties go back in the queue with their original queue time.
//
// Runs on the matchmaker's goroutine, so it touches only lock-guarded lobby state and the hubs'
// channel API, never hub fields.
func (gs *GameServer) HandleMatchFormed(result matchmaking.MatchResult) {
	// One entry per party lobby, in the matchmaker's order, so the host lobby's own members
	// come first and seating stays deterministic.
	lobbyIDs := make([]uuid.UUID, 0, len(result.Parties))
	seen := make(map[uuid.UUID]bool, len(result.Parties))
	seatable := make([]matchmaking.QueuedLobby, 0, len(result.Parties))
	seats := 0
	for _, party := range result.Parties {
		if party.LobbyID == uuid.Nil || seen[party.LobbyID] {
			continue
		}
		seen[party.LobbyID] = true

		// The matchmaker holds dormant parties out of matching already; this catches the one
		// that lost its last connection between that check and this callback.
		if live, serving := gs.HubStore.LiveConnections(party.LobbyID); !serving || live == 0 {
			log.Printf("Match in queue %s: party lobby %s has nobody connected; it is not being seated", result.QueueID, party.LobbyID)
			continue
		}
		lobbyIDs = append(lobbyIDs, party.LobbyID)
		seatable = append(seatable, party)
		seats += party.PlayerCount
	}

	// The match is played in the host party's lobby, so a dormant host is as fatal as a short
	// table however the remaining seats add up.
	if seats < result.TargetCount || !containsLobby(lobbyIDs, result.HostLobbyID) {
		log.Printf("Match in queue %s abandoned: %d of %d seats still connected; requeueing the live parties",
			result.QueueID, seats, result.TargetCount)
		gs.requeueParties(result, seatable)
		return
	}

	hostLob, ok := gs.LobbyStore.GetLobby(result.HostLobbyID)
	if !ok {
		log.Printf("Match formed in queue %s but host lobby %s is gone", result.QueueID, result.HostLobbyID)
		gs.requeueParties(result, seatable)
		return
	}

	hostLob.Mu.Lock()
	hostUserID := hostLob.HostUserID
	hostLob.Mu.Unlock()

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

// containsLobby reports whether id is in the list.
func containsLobby(ids []uuid.UUID, id uuid.UUID) bool {
	for _, candidate := range ids {
		if candidate == id {
			return true
		}
	}
	return false
}

// requeueParties puts the still-connected parties of an abandoned match back in their queue,
// keeping their original queue time so a party does not lose its place in line over somebody
// else's closed tab. Their lobbies stay searching: nothing was sent to them, so from the client's
// side the search simply continues.
//
// Party size is re-read from the lobby rather than taken from the stale queue entry, since
// members may have left while the match was being assembled. Dormant parties are left alone here:
// their hubs' idle window is the single authority on when a lobby with no connections is really
// gone (cambia-884), and it is what holds the grace period a page refresh needs.
func (gs *GameServer) requeueParties(result matchmaking.MatchResult, parties []matchmaking.QueuedLobby) {
	for _, party := range parties {
		lob, exists := gs.LobbyStore.GetLobby(party.LobbyID)
		if !exists {
			continue
		}
		lob.Mu.Lock()
		party.PlayerCount = lob.JoinedCount()
		searching := lob.Searching
		lob.Mu.Unlock()

		if !searching || party.PlayerCount <= 0 {
			log.Printf("Match in queue %s: lobby %s is no longer a queueable party; leaving it out",
				result.QueueID, party.LobbyID)
			continue
		}
		entry := party
		if err := gs.Matchmaker.Enqueue(&entry); err != nil {
			log.Printf("Match in queue %s: could not requeue lobby %s: %v", result.QueueID, party.LobbyID, err)
		}
	}
}

// PartyIsLive is the Matchmaker's PartyLive predicate: a queued party counts as live while its
// hub still holds at least one WebSocket connection. Membership alone cannot answer it, since a
// lobby keeps its members across a dropped socket (cambia-807, cambia-884).
func (gs *GameServer) PartyIsLive(lobbyID uuid.UUID) bool {
	live, serving := gs.HubStore.LiveConnections(lobbyID)
	return serving && live > 0
}

// WireMatchmaker binds the matchmaker's callbacks to this server. Both are set together so a
// caller cannot take the match callback without the liveness gate it relies on (cambia-933 F1).
func (gs *GameServer) WireMatchmaker() {
	gs.Matchmaker.OnMatchFormed = gs.HandleMatchFormed
	gs.Matchmaker.PartyLive = gs.PartyIsLive
}
