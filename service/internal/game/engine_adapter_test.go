// internal/game/engine_adapter_test.go
package game

import (
	"fmt"
	"testing"

	"github.com/google/uuid"
	engine "github.com/jason-s-yu/cambia/engine"
	"github.com/jason-s-yu/cambia/service/internal/models"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
		NumDecks:              1,
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
		NumDecks:              1,
	}
	assert.Equal(t, want, got)
}

// TestMapHouseRulesToEngine_AllowOpponentSnapping_UpdateRules pins the allowOpponentSnapping
// house rule end to end: HouseRules.Update (the same call path as the "update_rules" WS message)
// through mapHouseRulesToEngine's non-circuit output (cambia-781: engine_adapter.go previously
// hardcoded AllowOpponentSnapping: true regardless of what update_rules set, making the client
// checkbox a no-op).
func TestMapHouseRulesToEngine_AllowOpponentSnapping_UpdateRules(t *testing.T) {
	t.Run("defaults true when never set", func(t *testing.T) {
		g := NewCambiaGame()
		addTestPlayers(g, 2)

		got := g.mapHouseRulesToEngine()

		assert.True(t, got.AllowOpponentSnapping)
	})

	t.Run("update_rules false lands false in engine config", func(t *testing.T) {
		g := NewCambiaGame()
		addTestPlayers(g, 2)

		err := g.HouseRules.Update(map[string]interface{}{"allowOpponentSnapping": false})
		assert.NoError(t, err)

		got := g.mapHouseRulesToEngine()

		assert.False(t, got.AllowOpponentSnapping)
	})

	t.Run("update_rules true survives round trip", func(t *testing.T) {
		g := NewCambiaGame()
		addTestPlayers(g, 2)

		err := g.HouseRules.Update(map[string]interface{}{"allowOpponentSnapping": false})
		assert.NoError(t, err)
		err = g.HouseRules.Update(map[string]interface{}{"allowOpponentSnapping": true})
		assert.NoError(t, err)

		got := g.mapHouseRulesToEngine()

		assert.True(t, got.AllowOpponentSnapping)
	})
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

// TestMapHouseRulesToEngine_DefaultsMatchLegacyHardcodes pins the fields cambia-782 unpinned.
// Before that change mapHouseRulesToEngine wrote MaxGameTurns/CardsPerPlayer/CambiaAllowedRound/
// NumJokers/LockCallerHand/InitialViewCount as literals; they now come from the lobby's house
// rules, so a game whose rules were never touched must still produce exactly those values.
func TestMapHouseRulesToEngine_DefaultsMatchLegacyHardcodes(t *testing.T) {
	g := NewCambiaGame()
	addTestPlayers(g, 2)

	got := g.mapHouseRulesToEngine()

	assert.Equal(t, uint16(46), got.MaxGameTurns)
	assert.Equal(t, uint8(4), got.CardsPerPlayer)
	assert.Equal(t, uint8(0), got.CambiaAllowedRound)
	assert.Equal(t, uint8(2), got.NumJokers)
	assert.True(t, got.LockCallerHand)
	assert.Equal(t, uint8(2), got.InitialViewCount)
	// NumDecks was previously left unmapped and NewGame read 0 as 1; it is now written
	// explicitly, so the engine config carries the same single deck without the sentinel.
	assert.Equal(t, uint8(1), got.NumDecks)
}

// TestMapHouseRulesToEngine_ExposedRules_UpdateRules walks every house rule cambia-782 exposed
// through the same call path as the "update_rules" WS message and checks the value lands in the
// engine config. Extends the cambia-781 pattern to the numeric knobs.
func TestMapHouseRulesToEngine_ExposedRules_UpdateRules(t *testing.T) {
	cases := []struct {
		key    string
		value  interface{}
		assert func(*testing.T, engine.HouseRules)
	}{
		{"maxGameTurns", float64(0), func(t *testing.T, hr engine.HouseRules) {
			assert.Equal(t, uint16(0), hr.MaxGameTurns, "0 means unlimited turns")
		}},
		{"maxGameTurns", float64(120), func(t *testing.T, hr engine.HouseRules) {
			assert.Equal(t, uint16(120), hr.MaxGameTurns)
		}},
		{"cardsPerPlayer", float64(6), func(t *testing.T, hr engine.HouseRules) {
			assert.Equal(t, uint8(6), hr.CardsPerPlayer)
		}},
		{"cambiaAllowedRound", float64(3), func(t *testing.T, hr engine.HouseRules) {
			assert.Equal(t, uint8(3), hr.CambiaAllowedRound)
		}},
		{"numJokers", float64(0), func(t *testing.T, hr engine.HouseRules) {
			assert.Equal(t, uint8(0), hr.NumJokers, "a jokerless deck must survive the mapping")
		}},
		{"numDecks", float64(4), func(t *testing.T, hr engine.HouseRules) {
			assert.Equal(t, uint8(4), hr.NumDecks)
		}},
		{"lockCallerHand", false, func(t *testing.T, hr engine.HouseRules) {
			assert.False(t, hr.LockCallerHand)
		}},
		{"initialViewCount", float64(0), func(t *testing.T, hr engine.HouseRules) {
			assert.Equal(t, uint8(0), hr.InitialViewCount, "a blind start must survive the mapping")
		}},
		{"initialViewCount", float64(1), func(t *testing.T, hr engine.HouseRules) {
			assert.Equal(t, uint8(1), hr.InitialViewCount)
		}},
	}

	for _, tc := range cases {
		t.Run(fmt.Sprintf("%s=%v", tc.key, tc.value), func(t *testing.T) {
			g := NewCambiaGame()
			addTestPlayers(g, 2)

			require.NoError(t, g.HouseRules.Update(map[string]interface{}{tc.key: tc.value}))

			tc.assert(t, g.mapHouseRulesToEngine())
		})
	}
}

// TestMapHouseRulesToEngine_ZeroValueCardsPerPlayer covers a CambiaGame whose HouseRules were
// assigned from a struct built without DefaultHouseRules: 0 cards per player is not a config any
// lobby can produce, so the mapper reads it as unset and deals the standard 4 rather than nothing.
func TestMapHouseRulesToEngine_ZeroValueCardsPerPlayer(t *testing.T) {
	g := NewCambiaGame()
	g.HouseRules = HouseRules{}
	addTestPlayers(g, 2)

	got := g.mapHouseRulesToEngine()

	assert.Equal(t, uint8(4), got.CardsPerPlayer)
	assert.Equal(t, uint8(2), got.PenaltyDrawCount)
}

// TestBeginPreGame_HonorsExposedDealRules drives the exposed deal knobs through the real
// BeginPreGame -> Deal() path and checks observable state, the way cambia-508's regression test
// does for the defaults: a mapping that compiles but never reaches Deal() would still pass the
// mapper-level tests above.
func TestBeginPreGame_HonorsExposedDealRules(t *testing.T) {
	t.Run("cards per player, jokers and decks size the deal", func(t *testing.T) {
		hr := testHouseRules(0, 2)
		hr.CardsPerPlayer = 6
		hr.NumJokers = 0
		hr.NumDecks = 2
		g, players, _ := setupTestGame(t, 3, hr)

		for i, p := range players {
			engineIdx := g.PlayerToEngine[p.ID]
			assert.Equal(t, uint8(6), g.Engine.Players[engineIdx].HandLen, "player %d hand size", i)
		}
		// 2 decks * 52 jokerless cards, minus 3*6 dealt, minus the discard flip.
		assert.Equal(t, uint8(2*52-18-1), g.Engine.StockLen)
	})

	t.Run("the largest legal deal fits the engine and the UUID tracker", func(t *testing.T) {
		hr := testHouseRules(0, 2)
		hr.NumDecks = 4
		hr.NumJokers = 2
		hr.CardsPerPlayer = 6
		g, players, _ := setupTestGame(t, engine.MaxPlayers, hr)

		// 4 * 54 is exactly engine.MaxDeckSize, the ceiling numDecks is validated against.
		assert.Equal(t, uint8(engine.MaxDeckSize-8*6-1), g.Engine.StockLen)
		assert.Len(t, g.CardTracker.Registry, engine.MaxDeckSize, "every dealt card needs a UUID")
		for i, p := range players {
			engineIdx := g.PlayerToEngine[p.ID]
			assert.Equal(t, uint8(6), g.Engine.Players[engineIdx].HandLen, "player %d hand size", i)
		}
	})

	// The reveal is read off the private_initial_cards payload, not off sync_state: since
	// cambia-1094 the self-view hides every own card in every phase, so a sync snapshot can no
	// longer tell a one-card peek from a two-card one.
	t.Run("initial view count drives the pregame reveal", func(t *testing.T) {
		hr := testHouseRules(0, 2)
		hr.InitialViewCount = 1
		g, players, _ := setupTestGame(t, 2, hr)

		for i, p := range players {
			engineIdx := g.PlayerToEngine[p.ID]
			assert.Equal(t, uint8(1), g.Engine.Players[engineIdx].InitialPeekCount, "player %d peek count", i)

			cards := g.pregameInitialCards(p.ID)
			require.Lenf(t, cards, 1, "player %d: the reveal should name exactly the peeked card", i)
			require.NotNil(t, cards[0].Idx)
			assert.Equal(t, int(g.Engine.Players[engineIdx].InitialPeek[0]), *cards[0].Idx, "the reveal should name the engine's peeked slot")
			assert.NotEmpty(t, cards[0].Rank, "the reveal carries the face")

			hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(p.ID), p.ID)
			require.NotEmpty(t, hand)
			for slot, c := range hand {
				assert.Falsef(t, c.Known, "slot %d: no own card is face-up in sync_state", slot)
			}
		}
	})

	t.Run("zero initial view count reveals nothing", func(t *testing.T) {
		hr := testHouseRules(0, 2)
		hr.InitialViewCount = 0
		g, players, _ := setupTestGame(t, 2, hr)

		for _, p := range players {
			assert.Empty(t, g.pregameInitialCards(p.ID), "no peek means no reveal")
			hand := selfRevealedHand(g.GetCurrentObfuscatedGameState(p.ID), p.ID)
			for slot, c := range hand {
				assert.False(t, c.Known, "slot %d must stay hidden with no pregame peek", slot)
			}
		}
	})
}
