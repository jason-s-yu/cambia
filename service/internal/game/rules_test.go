// internal/game/rules_test.go
package game

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestHouseRulesUpdate_AcceptsInRangeValues walks every key the "update_rules" WS message can
// carry and checks it lands on the struct, including the boundary values of each range.
func TestHouseRulesUpdate_AcceptsInRangeValues(t *testing.T) {
	rules := DefaultHouseRules()

	require.NoError(t, rules.Update(map[string]interface{}{
		"allowDrawFromDiscardPile": true,
		"allowReplaceAbilities":    true,
		"allowOpponentSnapping":    false,
		"snapRace":                 true,
		"lockCallerHand":           false,
		"forfeitOnDisconnect":      false,
		"disconnectGraceSec":       float64(3600),
		"penaltyDrawCount":         float64(6),
		"turnTimerSec":             float64(0),
		"maxGameTurns":             float64(65535),
		"cardsPerPlayer":           float64(6),
		"cambiaAllowedRound":       float64(255),
		"numJokers":                float64(0),
		"numDecks":                 float64(4),
		"initialViewCount":         float64(0),
	}))

	assert.Equal(t, HouseRules{
		AllowDrawFromDiscardPile: true,
		AllowReplaceAbilities:    true,
		AllowOpponentSnapping:    false,
		SnapRace:                 true,
		LockCallerHand:           false,
		ForfeitOnDisconnect:      false,
		DisconnectGraceSec:       3600,
		PenaltyDrawCount:         6,
		TurnTimerSec:             0,
		MaxGameTurns:             65535,
		CardsPerPlayer:           6,
		CambiaAllowedRound:       255,
		NumJokers:                0,
		NumDecks:                 4,
		InitialViewCount:         0,
	}, rules)
}

// TestHouseRulesUpdate_RejectsOutOfRange covers the server-side bounds. Each ceiling exists
// because the engine cannot represent (or would misread) a larger value, so an out-of-range
// update must be refused rather than silently truncated by the uint8/uint16 conversions in
// mapHouseRulesToEngine.
func TestHouseRulesUpdate_RejectsOutOfRange(t *testing.T) {
	cases := []struct {
		name  string
		key   string
		value interface{}
	}{
		{"negative penalty draw", "penaltyDrawCount", float64(-1)},
		{"penalty draw past a full hand", "penaltyDrawCount", float64(7)},
		{"negative turn timer", "turnTimerSec", float64(-5)},
		{"negative disconnect grace", "disconnectGraceSec", float64(-1)},
		{"disconnect grace past an hour", "disconnectGraceSec", float64(3601)},
		{"negative turn cap", "maxGameTurns", float64(-1)},
		{"turn cap past uint16", "maxGameTurns", float64(65536)},
		{"zero cards per player", "cardsPerPlayer", float64(0)},
		{"cards per player past MaxHandSize", "cardsPerPlayer", float64(7)},
		{"negative cambia round", "cambiaAllowedRound", float64(-1)},
		{"cambia round past uint8", "cambiaAllowedRound", float64(256)},
		{"negative jokers", "numJokers", float64(-1)},
		{"three jokers", "numJokers", float64(3)},
		{"zero decks", "numDecks", float64(0)},
		{"five decks", "numDecks", float64(5)},
		{"negative initial view", "initialViewCount", float64(-1)},
		{"initial view past MaxHandSize", "initialViewCount", float64(7)},
		{"initial view past the dealt hand", "initialViewCount", float64(5)},
		{"fractional value", "cardsPerPlayer", float64(4.5)},
		{"wrong type", "numJokers", "two"},
		{"wrong type for a toggle", "lockCallerHand", float64(1)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rules := DefaultHouseRules()
			before := rules

			err := rules.Update(map[string]interface{}{tc.key: tc.value})

			require.Error(t, err)
			assert.Equal(t, before, rules, "a rejected update must leave the rules untouched")
		})
	}
}

// TestHouseRulesUpdate_IgnoresAbsentKeys pins the partial-update contract the settings panel
// relies on: keys the client omits keep their current value rather than resetting to zero.
func TestHouseRulesUpdate_IgnoresAbsentKeys(t *testing.T) {
	rules := DefaultHouseRules()
	rules.NumJokers = 0

	require.NoError(t, rules.Update(map[string]interface{}{"cardsPerPlayer": float64(5)}))

	assert.Equal(t, 5, rules.CardsPerPlayer)
	assert.Equal(t, 0, rules.NumJokers, "an omitted key must not be reset")
	assert.Equal(t, 2, rules.InitialViewCount)
}

// TestDefaultHouseRules_DisconnectGraceSecIsNinety pins the default reconnect grace (raised
// 60 -> 90 by cambia-1609; MATCHMAKING.md 8, RULES.md T5). This is service-owned (no engine
// counterpart), so TestDefaultHouseRules_MatchesEngineDefaults below cannot cover it.
func TestDefaultHouseRules_DisconnectGraceSecIsNinety(t *testing.T) {
	assert.Equal(t, 90, DefaultHouseRules().DisconnectGraceSec)
}

// TestDefaultHouseRules_MatchesEngineDefaults ties the service defaults to the engine's own
// DefaultHouseRules for every shared field, so the two cannot drift apart unnoticed.
func TestDefaultHouseRules_MatchesEngineDefaults(t *testing.T) {
	svc := DefaultHouseRules()
	g := NewCambiaGame()
	g.HouseRules = svc
	addTestPlayers(g, 2)

	got := g.mapHouseRulesToEngine()

	assert.Equal(t, uint16(46), got.MaxGameTurns)
	assert.Equal(t, uint8(4), got.CardsPerPlayer)
	assert.Equal(t, uint8(2), got.PenaltyDrawCount)
	assert.Equal(t, uint8(2), got.NumJokers)
	assert.Equal(t, uint8(1), got.NumDecks)
	assert.Equal(t, uint8(2), got.InitialViewCount)
	assert.True(t, got.LockCallerHand)
	assert.True(t, got.AllowOpponentSnapping)
}
