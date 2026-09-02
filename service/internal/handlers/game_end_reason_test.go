// internal/handlers/game_end_reason_test.go
//
// The results frame's `reason` field, over the real WS handlers (cambia-1831). game_results is
// built here, in attachOnGameEnd, and is the only one of the two results frames a client that
// reconnects into a finished game ever sees, so a game the panic guard aborted is named as such
// on this frame or is not named at all. The abort itself is covered in internal/game, and what
// the hub does with the frame afterwards in internal/hub; this covers the frame's own shape.
package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"

	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/game"
)

// endReasonHarness drives a lobby to a started game over real WebSockets and hands back the two
// clients and the game, which is the state either half of this file begins from.
func endReasonHarness(t *testing.T) (*wsTestClient, *wsTestClient, uuid.UUID, uuid.UUID, uuid.UUID, *game.CambiaGame) {
	t.Helper()
	auth.Init()

	gs := NewGameServer()
	gs.CountdownDuration = 50 * time.Millisecond

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))
	ts := httptest.NewServer(mux)
	t.Cleanup(ts.Close)

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())

	lobUUID := createPublicLobby(t, gs, hostToken)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	t.Cleanup(cancel)

	host := dialWSClient(t, ctx, ts.URL, lobUUID.String(), hostToken)
	t.Cleanup(host.close)
	p2 := dialWSClient(t, ctx, ts.URL, lobUUID.String(), p2Token)
	t.Cleanup(p2.close)
	host.settle()
	p2.settle()

	host.sendReliable("ready")
	host.settle()
	p2.settle()
	p2.sendReliable("ready")

	if host.waitForType("game_started", 5*time.Second) == nil {
		t.Fatal("host never received game_started")
	}
	if p2.waitForType("game_started", 5*time.Second) == nil {
		t.Fatal("player 2 never received game_started")
	}
	host.settle()
	p2.settle()

	g := gs.GameStore.GetGameByLobbyID(lobUUID)
	if g == nil {
		t.Fatalf("no CambiaGame registered for lobby %s", lobUUID)
	}
	return host, p2, lobUUID, hostID, p2ID, g
}

// TestGameResultsNamesTheInternalErrorTheGuardEndedOn drives the callback with the reason the
// abort produces and reads the frame back off both sockets. Before this the table was handed an
// ordinary scoreboard, built from whatever hands the panic left mid-move and indistinguishable
// from a game somebody won.
func TestGameResultsNamesTheInternalErrorTheGuardEndedOn(t *testing.T) {
	host, p2, lobUUID, hostID, p2ID, g := endReasonHarness(t)

	scores := map[uuid.UUID]int{hostID: 31, p2ID: 38}
	g.OnGameEnd(lobUUID, hostID, scores, map[uuid.UUID]string{}, scores, uuid.Nil, nil, game.EndReasonInternalError)

	for name, client := range map[string]*wsTestClient{"host": host, "player 2": p2} {
		env := client.waitForType("game_results", 5*time.Second)
		if env == nil {
			t.Fatalf("%s never received game_results", name)
		}
		var payload struct {
			Reason string `json:"reason"`
		}
		if err := json.Unmarshal(env.Payload, &payload); err != nil {
			t.Fatalf("%s: decode game_results: %v", name, err)
		}
		if payload.Reason != string(game.EndReasonInternalError) {
			t.Errorf("%s: game_results reason = %q, want %q", name, payload.Reason, game.EndReasonInternalError)
		}
	}
}

// TestGameResultsNamesNoReasonForAnOrdinaryEnding is the negative: a game that reached one of its
// rulebook endings keeps the frame every existing client already reads, with no reason field on
// it at all.
func TestGameResultsNamesNoReasonForAnOrdinaryEnding(t *testing.T) {
	host, _, lobUUID, hostID, p2ID, g := endReasonHarness(t)

	scores := map[uuid.UUID]int{hostID: 31, p2ID: 38}
	g.OnGameEnd(lobUUID, hostID, scores, map[uuid.UUID]string{}, scores, uuid.Nil, nil, game.EndReasonNormal)

	env := host.waitForType("game_results", 5*time.Second)
	if env == nil {
		t.Fatal("host never received game_results")
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(env.Payload, &payload); err != nil {
		t.Fatalf("decode game_results: %v", err)
	}
	if _, named := payload["reason"]; named {
		t.Errorf("an ordinary result must carry no reason field, got %v", payload["reason"])
	}
}
