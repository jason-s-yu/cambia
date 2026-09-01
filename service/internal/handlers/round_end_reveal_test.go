// internal/handlers/round_end_reveal_test.go
//
// The round-end reveal over the real WS handlers (RULES.md 3C, cambia-1542). attachOnGameEnd's
// game_results is the frame that has to carry it: the game is dropped from the store on the line
// after it is emitted, so a client that reconnects into the results is answered with the hub's
// held copy of that frame and never sees game_end.
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

// TestGameResultsCarriesTheRoundEndReveal plays a real casual table to its end and reads the
// game_results both seats receive: it must name every scored seat's hand, card for card, matching
// the hands the same endGame scored.
func TestGameResultsCarriesTheRoundEndReveal(t *testing.T) {
	auth.Init()

	gs := NewGameServer()
	gs.CountdownDuration = 50 * time.Millisecond

	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	mux := http.NewServeMux()
	mux.HandleFunc("/lobby/create", CreateLobbyHandler(gs))
	mux.Handle("/ws/", HubWSHandler(logger, gs))

	ts := httptest.NewServer(mux)
	defer ts.Close()

	hostID := uuid.New()
	hostToken, _ := auth.CreateJWT(hostID.String())
	p2ID := uuid.New()
	p2Token, _ := auth.CreateJWT(p2ID.String())

	lobUUID := createPublicLobby(t, gs, hostToken)
	lobbyID := lobUUID.String()

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	host := dialWSClient(t, ctx, ts.URL, lobbyID, hostToken)
	defer host.close()
	p2 := dialWSClient(t, ctx, ts.URL, lobbyID, p2Token)
	defer p2.close()
	host.settle()
	p2.settle()

	host.sendReliable("ready")
	host.settle()
	p2.settle()
	p2.sendReliable("ready")

	if host.waitForType("game_started", 5*time.Second) == nil {
		t.Fatalf("host never received game_started")
	}
	if p2.waitForType("game_started", 5*time.Second) == nil {
		t.Fatalf("player 2 never received game_started")
	}
	host.settle()
	p2.settle()

	g := gs.GameStore.GetGameByLobbyID(lobUUID)
	if g == nil {
		t.Fatalf("no CambiaGame registered for lobby %s", lobbyID)
	}

	// End the round through the public entry every terminal path funnels into (the engine calls
	// the same endGame once it reports terminal, whether that is a called Cambia, the turn cap or
	// an exhausted stockpile).
	g.EndGame()

	for name, client := range map[string]*wsTestClient{"host": host, "player 2": p2} {
		env := client.waitForType("game_results", 5*time.Second)
		if env == nil {
			t.Fatalf("%s never received game_results", name)
		}
		var payload struct {
			FinalHands []game.FinalHand `json:"finalHands"`
			Scores     map[string]int   `json:"scores"`
		}
		if err := json.Unmarshal(env.Payload, &payload); err != nil {
			t.Fatalf("%s: decode game_results: %v", name, err)
		}
		if len(payload.FinalHands) != 2 {
			t.Fatalf("%s: game_results carried %d revealed hands, want 2 (RULES.md 3C)", name, len(payload.FinalHands))
		}
		for _, hand := range payload.FinalHands {
			if len(hand.Cards) == 0 {
				t.Errorf("%s: seat %s revealed no cards", name, hand.PlayerID)
			}
			total := 0
			for _, card := range hand.Cards {
				if card.Rank == "" {
					t.Errorf("%s: seat %s slot %d carries no rank", name, hand.PlayerID, card.Idx)
				}
				if card.ID == uuid.Nil {
					t.Errorf("%s: seat %s slot %d carries no card id", name, hand.PlayerID, card.Idx)
				}
				total += card.Value
			}
			if got := payload.Scores[hand.PlayerID.String()]; got != total {
				t.Errorf("%s: seat %s revealed values sum to %d, the same frame scores it %d",
					name, hand.PlayerID, total, got)
			}
		}
	}
}
