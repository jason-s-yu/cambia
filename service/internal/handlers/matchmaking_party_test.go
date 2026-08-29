// internal/handlers/matchmaking_party_test.go
package handlers

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/matchmaking"
)

// postCancelSearch runs CancelSearchHandler for DELETE /lobby/{id}/search as token's user.
func postCancelSearch(t *testing.T, gs *GameServer, token string, lobbyID uuid.UUID) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("DELETE", fmt.Sprintf("/lobby/%s/search", lobbyID), nil)
	req.Header.Set("Cookie", "auth_token="+token)
	w := httptest.NewRecorder()
	CancelSearchHandler(gs).ServeHTTP(w, req)
	return w
}

// TestPublicLobbyQueueIDDerivesRankedMode is the cambia-966 item 2 regression: a public or
// private lobby that carries a queue id on create (the "queue a standing party" shape
// SearchLobbyHandler already accepts, cambia-933) must have its Mode derived from that queue's
// Ranked flag exactly like a matchmaking-typed lobby does, since NewCambiaGameFromLobby reads
// Mode - not QueueID - to decide whether the resulting game is rated. Before this fix the id was
// accepted and validated but Mode stayed "casual" forever, so a party that queued into a ranked
// queue this way played an unrated game.
func TestPublicLobbyQueueIDDerivesRankedMode(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, token, `{"type":"public","queueID":"h2h_quickplay"}`))
	if lob.Mode != "ranked" {
		t.Fatalf("expected a public lobby carrying a ranked queue id to derive mode ranked, got %q", lob.Mode)
	}
	if lob.Type != "public" {
		t.Fatalf("queue id must not change the lobby's own type, got %q", lob.Type)
	}
}

// TestPublicLobbyNoQueueIDStaysCasual pins the negative: a public lobby with no queue id is
// unaffected by the item 2 fix and stays casual as before.
func TestPublicLobbyNoQueueIDStaysCasual(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, token, `{"type":"public","gameMode":"head_to_head"}`))
	if lob.Mode != "casual" {
		t.Fatalf("a public lobby with no queue id must stay casual, got %q", lob.Mode)
	}
}

// TestSearchRejectsOversizeH2HParty is the cambia-966 item 4 regression: matchmaking.ValidateParty
// existed with no caller, so a party of two queuing into an H2H (solo-queue-only) queue passed
// Matchmaker.Enqueue's own check (party size <= target size) and would always be paired with
// itself, skipping matchmaking and its quality gate entirely. SearchLobbyHandler must reject the
// party before it ever reaches Enqueue.
func TestSearchRejectsOversizeH2HParty(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	hostToken, _ := auth.CreateJWT(uuid.New().String())
	friendToken, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, hostToken, `{"type":"public","queueID":"h2h_quickplay"}`))

	// A public/private lobby's host is only invited at create time, not joined (that promotion
	// is the WebSocket upgrade's job, cambia-771); the search handler counts JoinedCount, so the
	// test joins both the host and the friend explicitly rather than opening real sockets.
	joinLobbyAs(t, gs, lob.ID, hostToken)
	joinLobbyAs(t, gs, lob.ID, friendToken)

	w := postSearch(t, gs, hostToken, lob.ID)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for a 2-player party in a solo-queue-only queue, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "solo queue") {
		t.Fatalf("400 body should name the solo-queue rule, got %q", w.Body.String())
	}

	lob.Mu.Lock()
	searching := lob.Searching
	lob.Mu.Unlock()
	if searching {
		t.Fatalf("a rejected party must not be left in the searching state")
	}
}

// TestSearchThenCancelLeavesConsistentState is a sequential smoke test for the cambia-966 item 3
// fix's code path: search and cancel now write lob.Searching and notify the hub inside the same
// lob.Mu critical section (previously two separate ones), so the two states can no longer
// disagree. The bug itself was a goroutine race in the gap between those two sections - two
// concurrent requests on the same lobby landing their hub notices in the opposite order from the
// one their lob.Searching writes settled on - which a sequential test cannot reproduce; the fix
// is a mutex-ordering invariant (whichever critical section runs last is also the one whose hub
// notice is sent last) verified by inspection and by this suite passing under -race.
func TestSearchThenCancelLeavesConsistentState(t *testing.T) {
	auth.Init()
	gs := NewGameServer()
	token, _ := auth.CreateJWT(uuid.New().String())

	lob := createdLobby(t, postCreateLobby(t, gs, token, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))

	if w := postSearch(t, gs, token, lob.ID); w.Code != http.StatusOK {
		t.Fatalf("search failed: %d %s", w.Code, w.Body.String())
	}
	if w := postCancelSearch(t, gs, token, lob.ID); w.Code != http.StatusOK {
		t.Fatalf("cancel failed: %d %s", w.Code, w.Body.String())
	}

	lob.Mu.Lock()
	searching := lob.Searching
	lob.Mu.Unlock()
	if searching {
		t.Fatalf("lob.Searching must be false after a cancel that ran after the search")
	}
}

// TestSearchPopulatesPartyRatingForQualityGate is the cambia-1041 regression: QueuedLobby.AvgRating
// and MaxRD had no writer anywhere, so every entry SearchLobbyHandler enqueued carried the zero
// value regardless of the party's real rating, and glicko2Quality's spread term was zero for
// every pairing. Both test users are freshly minted JWTs with no users row, so this also covers
// the no-rating-row default path (rating.DefaultMu/DefaultPhi): the match must still form (no
// blocked-on-DB failure) and must carry the pool defaults, not zero.
//
// PartyLive is left unwired (WireMatchmaker is not called) so the match forms with no WebSocket
// connection open, mirroring TestNilPartyLiveMatchesEverything in the matchmaking package.
func TestSearchPopulatesPartyRatingForQualityGate(t *testing.T) {
	auth.Init()
	gs := NewGameServer()

	resultCh := make(chan matchmaking.MatchResult, 1)
	gs.Matchmaker.OnMatchFormed = func(r matchmaking.MatchResult) { resultCh <- r }

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	go gs.Matchmaker.Run(ctx)

	tokenA, _ := auth.CreateJWT(uuid.New().String())
	tokenB, _ := auth.CreateJWT(uuid.New().String())

	lobA := createdLobby(t, postCreateLobby(t, gs, tokenA, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))
	lobB := createdLobby(t, postCreateLobby(t, gs, tokenB, `{"type":"matchmaking","queueID":"h2h_quickplay"}`))

	if w := postSearch(t, gs, tokenA, lobA.ID); w.Code != http.StatusOK {
		t.Fatalf("search for lobby A failed: %d %s", w.Code, w.Body.String())
	}
	if w := postSearch(t, gs, tokenB, lobB.ID); w.Code != http.StatusOK {
		t.Fatalf("search for lobby B failed: %d %s", w.Code, w.Body.String())
	}

	// The matchmaker ticks every 5 seconds; allow two ticks plus slack.
	select {
	case result := <-resultCh:
		if len(result.Parties) != 2 {
			t.Fatalf("expected 2 parties in the match, got %d", len(result.Parties))
		}
		for _, p := range result.Parties {
			if p.AvgRating != 1500 {
				t.Errorf("party %s: AvgRating = %v, want the pool default 1500 (no rating row)", p.LobbyID, p.AvgRating)
			}
			if p.MaxRD != 350 {
				t.Errorf("party %s: MaxRD = %v, want the pool default 350 (no rating row)", p.LobbyID, p.MaxRD)
			}
		}
	case <-time.After(15 * time.Second):
		t.Fatalf("no match formed within 15s")
	}
}
