// internal/handlers/matchmaking_dormant_test.go
package handlers

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// TestHandleMatchFormedSkipsDormantParty is the cambia-933 F1 regression at the callback: a party
// that lost its last connection between the matchmaker's liveness check and the callback must not
// be seated. Consolidating it would move the connected player into a ready check the absent side
// can never answer, with no timeout on it and no way out but leaving the lobby.
func TestHandleMatchFormedSkipsDormantParty(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	mux := http.NewServeMux()
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	tokenLive, _ := auth.CreateJWT(uuid.New().String())
	tokenDormant, _ := auth.CreateJWT(uuid.New().String())

	lobLive := createdLobby(t, postCreateLobby(t, gs, tokenLive, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))
	lobDormant := createdLobby(t, postCreateLobby(t, gs, tokenDormant, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))

	// Only one side ever opens a socket; the other is the closed tab.
	client := dialWSClient(t, ctx, ts.URL, lobLive.ID.String(), tokenLive)
	defer client.close()
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("the connected client never received lobby_state")
	}

	if w := postSearch(t, gs, tokenLive, lobLive.ID); w.Code != http.StatusOK {
		t.Fatalf("search for the connected lobby failed: %d %s", w.Code, w.Body.String())
	}
	if w := postSearch(t, gs, tokenDormant, lobDormant.ID); w.Code != http.StatusOK {
		t.Fatalf("search for the dormant lobby failed: %d %s", w.Code, w.Body.String())
	}

	// Stand in for commitMatch, which takes both entries out of the queue before it calls back.
	gs.Matchmaker.Dequeue(lobLive.ID)
	gs.Matchmaker.Dequeue(lobDormant.ID)

	queuedAt := time.Now().Add(-90 * time.Second)
	party := func(id uuid.UUID) matchmaking.QueuedLobby {
		return matchmaking.QueuedLobby{
			LobbyID:     id,
			PlayerCount: 1,
			QueueID:     "h2h_quickplay",
			TargetCount: 2,
			IsRanked:    true,
			QueuedAt:    queuedAt,
		}
	}
	gs.HandleMatchFormed(matchmaking.MatchResult{
		HostLobbyID: lobLive.ID,
		Parties:     []matchmaking.QueuedLobby{party(lobLive.ID), party(lobDormant.ID)},
		QueueID:     "h2h_quickplay",
		TargetCount: 2,
		IsRanked:    true,
	})

	if env := client.waitForType("match_found", 2*time.Second); env != nil {
		t.Fatalf("the connected player was sent into a match with an absent opponent: %s", string(env.Payload))
	}

	// The absent player must not have been pulled into the live lobby either.
	lob, ok := gs.LobbyStore.GetLobby(lobLive.ID)
	if !ok {
		t.Fatalf("the connected lobby is gone")
	}
	lob.Mu.Lock()
	members := len(lob.Users)
	searching := lob.Searching
	lob.Mu.Unlock()
	if members != 1 {
		t.Fatalf("expected the connected lobby to keep its single member, got %d", members)
	}
	if !searching {
		t.Fatalf("the connected lobby stopped searching; nothing told its client the search ended")
	}

	// The live party goes back in the queue, keeping its original queue time, and the dormant one
	// does not: it is released by its hub's idle window, not from here.
	stat := gs.Matchmaker.QueueStats()["h2h_quickplay"]
	if stat.PlayerCount != 1 {
		t.Fatalf("expected only the connected party requeued, got playerCount %d", stat.PlayerCount)
	}
	if stat.AvgWaitSec < 60 {
		t.Fatalf("expected the requeued party to keep its place in line, got avgWait %.1fs", stat.AvgWaitSec)
	}
}

// TestHandleMatchFormedSeatsBothLiveParties pins the other direction: with both parties connected
// the liveness gate is invisible and the match consolidates as before.
func TestHandleMatchFormedSeatsBothLiveParties(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)
	mux := http.NewServeMux()
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	tokenA, _ := auth.CreateJWT(uuid.New().String())
	tokenB, _ := auth.CreateJWT(uuid.New().String())

	lobA := createdLobby(t, postCreateLobby(t, gs, tokenA, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))
	lobB := createdLobby(t, postCreateLobby(t, gs, tokenB, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))

	clientA := dialWSClient(t, ctx, ts.URL, lobA.ID.String(), tokenA)
	defer clientA.close()
	clientB := dialWSClient(t, ctx, ts.URL, lobB.ID.String(), tokenB)
	defer clientB.close()
	if clientA.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("client A never received lobby_state")
	}
	if clientB.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("client B never received lobby_state")
	}

	party := func(id uuid.UUID) matchmaking.QueuedLobby {
		return matchmaking.QueuedLobby{
			LobbyID:     id,
			PlayerCount: 1,
			QueueID:     "h2h_quickplay",
			TargetCount: 2,
			IsRanked:    true,
			QueuedAt:    time.Now().Add(-10 * time.Second),
		}
	}
	gs.HandleMatchFormed(matchmaking.MatchResult{
		HostLobbyID: lobA.ID,
		Parties:     []matchmaking.QueuedLobby{party(lobA.ID), party(lobB.ID)},
		QueueID:     "h2h_quickplay",
		TargetCount: 2,
		IsRanked:    true,
	})

	if clientA.waitForType("match_found", 5*time.Second) == nil {
		t.Fatalf("client A never received match_found")
	}
	if clientB.waitForType("match_found", 5*time.Second) == nil {
		t.Fatalf("client B never received match_found")
	}
	if got := gs.Matchmaker.QueueStats()["h2h_quickplay"].PlayerCount; got != 0 {
		t.Fatalf("a consolidated match must not requeue anybody, got playerCount %d", got)
	}
}
