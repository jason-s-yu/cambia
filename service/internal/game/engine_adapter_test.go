// internal/game/engine_adapter_test.go
package game

import (
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
)

// addTestPlayers appends n bare players to g via AddPlayer, mirroring lobby join order.
func addTestPlayers(g *CambiaGame, n int) {
	for i := 0; i < n; i++ {
		g.AddPlayer(&models.Player{
			ID:        uuid.New(),
			Connected: true,
			User:      &models.User{ID: uuid.New(), Username: "Player" + string(rune('A'+i))},
		})
	}
}

// TestMapHouseRulesToEngine_NonCircuit_MatchesServiceRules pins mapHouseRulesToEngine's
// non-circuit output field-for-field. It fails loudly the moment a mapped or intentionally
// pinned field is dropped from the literal in engine_adapter.go (cambia-508: NumJokers,
// LockCallerHand, and NumPlayers were all silently omitted, so live non-circuit games dealt a
// 52-card deck with no Jokers, an unlocked Cambia-caller hand, and only 2 dealt hands regardless
// of lobby size).
func TestMapHouseRulesToEngine_NonCircuit_MatchesServiceRules(t *testing.T) {
	g := NewCambiaGame()
	g.HouseRules.AllowDrawFromDiscardPile = true
	g.HouseRules.AllowReplaceAbilities = true
	g.HouseRules.SnapRace = true
	g.HouseRules.PenaltyDrawCount = 3
	addTestPlayers(g, 3)

	got := g.mapHouseRulesToEngine()

	want := engine.HouseRules{
		MaxGameTurns:          46,
		CardsPerPlayer:        4,
		CambiaAllowedRound:    0,
		PenaltyDrawCount:      3,
		AllowDrawFromDiscard:  true,
		AllowReplaceAbilities: true,
		AllowOpponentSnapping: true,
		SnapRace:              true,
		NumJokers:             2,
		LockCallerHand:        true,
		NumPlayers:            3,
		InitialViewCount:      2,
		// NumDecks intentionally left at zero: NewGame treats NumDecks==0 as 1 (engine/game.go),
		// matching DefaultHouseRules, so leaving it unmapped here is not an omission.
	}
	assert.Equal(t, want, got)
}

// TestMapHouseRulesToEngine_NonCircuit_PenaltyDrawDefault covers the PenaltyDrawCount==0 fallback
// path (service HouseRules left at its zero value) alongside the same field set as above.
func TestMapHouseRulesToEngine_NonCircuit_PenaltyDrawDefault(t *testing.T) {
	g := NewCambiaGame()
	g.HouseRules.PenaltyDrawCount = 0
	addTestPlayers(g, 2)

	got := g.mapHouseRulesToEngine()

	want := engine.HouseRules{
		MaxGameTurns:          46,
		CardsPerPlayer:        4,
		CambiaAllowedRound:    0,
		PenaltyDrawCount:      2,
		AllowDrawFromDiscard:  false,
		AllowReplaceAbilities: false,
		AllowOpponentSnapping: true,
		SnapRace:              false,
		NumJokers:             2,
		LockCallerHand:        true,
		NumPlayers:            2,
		InitialViewCount:      2,
	}
	assert.Equal(t, want, got)
}

// TestMapHouseRulesToEngine_Circuit_MatchesServiceRules pins the circuit-mode mapping so the two
// literals stay consistent: circuit already derives NumPlayers from len(g.Players) and inherits
// NumJokers/LockCallerHand/NumDecks/InitialViewCount from engine.TournamentHouseRules. This locks
// that behavior in as a regression guard alongside the non-circuit test above.
func TestMapHouseRulesToEngine_Circuit_MatchesServiceRules(t *testing.T) {
	g := NewCambiaGame()
	g.Circuit.Enabled = true
	g.HouseRules.PenaltyDrawCount = 5
	addTestPlayers(g, 4)

	got := g.mapHouseRulesToEngine()

	want := engine.TournamentHouseRules()
	want.PenaltyDrawCount = 5
	want.NumPlayers = 4
	assert.Equal(t, want, got)
}

// TestBeginPreGame_DealsFullJokerDeckAndAllHands is a functional regression test for cambia-508:
// it drives mapHouseRulesToEngine through the real BeginPreGame -> Deal() path (rather than
// calling the mapper directly) and checks observable deal state, mirroring the live WS trace that
// exposed the bug (stockpile 43 instead of 45 post-deal for a 2-player game).
func TestBeginPreGame_DealsFullJokerDeckAndAllHands(t *testing.T) {
	g, players, _ := setupTestGame(t, 3, nil)

	// 54-card deck (RULES.md §1) minus 3*4 dealt minus 1 discard flip.
	assert.Equal(t, uint8(41), g.Engine.StockLen, "stockpile should reflect a 54-card deck, not 52")
	assert.Equal(t, uint8(1), g.Engine.DiscardLen)

	// All 3 players must have a dealt hand; before the NumPlayers fix, Deal() used the
	// numPlayers()-defaults-to-2 fallback and left the third player's hand empty.
	for i, p := range players {
		engineIdx := g.PlayerToEngine[p.ID]
		assert.Equal(t, uint8(4), g.Engine.Players[engineIdx].HandLen, "player %d should have a dealt hand", i)
	}

	assert.True(t, g.Engine.Rules.LockCallerHand, "non-circuit games should lock the Cambia caller's hand by default")
}
