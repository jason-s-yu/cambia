// internal/handlers/idle_reap_test.go
//
// Idle reaping end to end (cambia-836). Only a deliberate leave releases membership (cambia-807),
// so a lobby everyone closed their tab on keeps its members, never reaches OnEmpty, and since
// cambia-808 keeps a parked hub goroutine for the life of the process. The reaper runs the same
// whole-lobby teardown the last leave runs.
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

// waitLobbyGone polls until the lobby has left the store, or the timeout expires.
func waitLobbyGone(t *testing.T, gs *GameServer, lobbyID uuid.UUID, timeout time.Duration) bool {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, exists := gs.LobbyStore.GetLobby(lobbyID); !exists {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

// TestAbandonedLobbyIsReapedWithItsHubAndQueueState drives the whole path: a member connects,
// closes the tab without leaving, and everything the lobby owns is released once the idle window
// elapses.
func TestAbandonedLobbyIsReapedWithItsHubAndQueueState(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	gs.LobbyIdleTTL = 150 * time.Millisecond

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())

	lobbyUUID := createPublicLobby(t, gs, hostToken)
	lob, exists := gs.LobbyStore.GetLobby(lobbyUUID)
	if !exists {
		t.Fatalf("lobby %s missing from store after create", lobbyUUID)
	}
	h, hasHub := gs.HubStore.GetHub(lobbyUUID)
	if !hasHub {
		t.Fatalf("hub %s missing from store after create", lobbyUUID)
	}

	// State the teardown has to reach beyond the lobby and hub entries.
	if err := gs.Matchmaker.Enqueue(&matchmaking.QueuedLobby{
		LobbyID:     lobbyUUID,
		PlayerCount: 1,
		QueueID:     "h2h_quickplay",
		TargetCount: 2,
		QueuedAt:    time.Now(),
	}); err != nil {
		t.Fatalf("failed to enqueue lobby for matchmaking: %v", err)
	}
	gs.CircuitStore.Set(lobbyUUID, nil, map[uuid.UUID]int{hostID: 0})

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	client := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}

	// Closing the tab: the socket goes, the membership stays.
	client.close()
	time.Sleep(50 * time.Millisecond)
	if !isMember(lob, hostID) {
		t.Fatalf("precondition: a dropped connection must keep membership, which is why the reaper exists")
	}

	if !waitLobbyGone(t, gs, lobbyUUID, 3*time.Second) {
		t.Fatalf("an abandoned lobby was never reaped")
	}
	if _, stillRegistered := gs.HubStore.GetHub(lobbyUUID); stillRegistered {
		t.Fatalf("the reap must deregister the lobby's hub")
	}
	waitHubAlive(t, h, false)

	if _, playerMap := gs.CircuitStore.Get(lobbyUUID); playerMap != nil {
		t.Fatalf("the reap must release the lobby's circuit state")
	}
	if stat := gs.Matchmaker.QueueStats()["h2h_quickplay"]; stat.PlayerCount != 0 {
		t.Fatalf("the reap must dequeue the lobby from matchmaking, still %d queued", stat.PlayerCount)
	}
	if resp := getActiveSession(t, gs, hostToken); resp.Active != nil {
		t.Fatalf("expected no resumable session for a reaped lobby, got %+v", *resp.Active)
	}

	// The stale URL is refused, the same as after a deliberate last leave (cambia-808).
	if _, _, err := tryDialWS(ctx, ts.URL, lobbyUUID.String(), hostToken); err == nil {
		t.Fatalf("expected the reaped lobby's URL to be refused")
	}
}

// TestOccupiedLobbyIsNotReaped is the false-positive guard at the handler level: a client that
// stays connected past several idle windows keeps its lobby.
func TestOccupiedLobbyIsNotReaped(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	gs.LobbyIdleTTL = 80 * time.Millisecond

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())

	lobbyUUID := createPublicLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	client := dialWSClient(t, ctx, ts.URL, lobbyUUID.String(), hostToken)
	defer client.close()
	if client.waitForType("lobby_state", 5*time.Second) == nil {
		t.Fatalf("connection never received lobby_state")
	}

	time.Sleep(500 * time.Millisecond)

	if _, exists := gs.LobbyStore.GetLobby(lobbyUUID); !exists {
		t.Fatalf("a lobby with a live connection was reaped")
	}
	if _, hasHub := gs.HubStore.GetHub(lobbyUUID); !hasHub {
		t.Fatalf("the hub of an occupied lobby was deregistered")
	}
}

// TestInGameLobbyIsNotReaped keeps the reaper away from a running table whose players have all
// dropped: the game still has to finish on its own terms (turn timers, or the forfeit rule wired
// in cambia-837), and reaping the lobby would take the game with it.
func TestInGameLobbyIsNotReaped(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	gs.LobbyIdleTTL = 80 * time.Millisecond

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	playerID := uuid.New()

	lob, _, _ := newRunningLobby(t, gs, hostToken, `{"type":"public","gameMode":"head_to_head"}`)
	joinLobbyAs(t, gs, lob.ID, hostToken)
	startTestGame(t, gs, lob, []uuid.UUID{hostID, playerID})

	time.Sleep(500 * time.Millisecond)

	if _, exists := gs.LobbyStore.GetLobby(lob.ID); !exists {
		t.Fatalf("an in-game lobby was reaped while its game was still running")
	}

	// Once the game ends the lobby is reconsidered without needing another departure.
	lob.Mu.Lock()
	lob.InGame = false
	lob.GameID = uuid.Nil
	lob.Mu.Unlock()

	if !waitLobbyGone(t, gs, lob.ID, 3*time.Second) {
		t.Fatalf("a lobby idle since its game ended was never reaped")
	}
}
