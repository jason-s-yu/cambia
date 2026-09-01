// internal/game/rules.go
package game

import (
	"fmt"
	"math"

	engine "github.com/jason-s-yu/cambia/engine"
)

// Bounds for the numeric house rules. Every ceiling is derived from the engine code that
// consumes the field, not from taste: a value the engine cannot represent (or that would
// index past a fixed-size array) is rejected at the lobby edge rather than silently
// truncated by the uint8/uint16 conversions in mapHouseRulesToEngine.
const (
	// engine.HouseRules.MaxGameTurns is a uint16 (engine/rules.go); 0 means unlimited.
	maxGameTurnsMin = 0
	maxGameTurnsMax = math.MaxUint16

	// Deal() writes Hand[c] for every c < CardsPerPlayer and PlayerState.Hand is
	// [engine.MaxHandSize]Card, so anything above MaxHandSize indexes past the array.
	// 0 is rejected outright: a hand of no cards is not a playable game.
	cardsPerPlayerMin = 1
	cardsPerPlayerMax = engine.MaxHandSize

	// engine.HouseRules.CambiaAllowedRound is a uint8 compared against
	// TurnNumber/numPlayers() (engine/legal.go, engine/actions.go).
	cambiaAllowedRoundMin = 0
	cambiaAllowedRoundMax = math.MaxUint8

	// NewGame appends jokers with `for j := 0; j < NumJokers && j < 2` but sizes StockLen as
	// NumDecks*(52+NumJokers), so a count above 2 claims stockpile slots that hold no card.
	numJokersMin = 0
	numJokersMax = 2

	// GameState.Stockpile is [engine.MaxDeckSize]Card and MaxDeckSize is 216 = 4 * 54, the
	// exact capacity of four full joker decks.
	numDecksMin = 1
	numDecksMax = engine.MaxDeckSize / engine.StandardDeckSize

	// Deal() writes InitialPeek[i] for every i < InitialViewCount and PlayerState.InitialPeek is
	// [engine.MaxHandSize]uint8, so this is the absolute ceiling. The playable ceiling is lower
	// still: a peek cannot cover more cards than the hand holds, which the cross-field check in
	// Update enforces against CardsPerPlayer. The pregame reveal ships the whole peek as a list
	// on private_initial_cards, so any count in range has somewhere to go (cambia-817).
	initialViewCountMin = 0
	initialViewCountMax = engine.MaxHandSize

	// PenaltyDrawCount converts to a uint8 in mapHouseRulesToEngine and snap.go stops adding
	// penalty cards once a hand reaches MaxHandSize, so a larger value cannot take effect.
	penaltyDrawCountMin = 0
	penaltyDrawCountMax = engine.MaxHandSize

	// TurnTimerSec has no engine counterpart (the service owns the turn timer); 0 disables it
	// and the ceiling is a day, well past any playable turn length.
	turnTimerSecMin = 0
	turnTimerSecMax = 86400

	// DisconnectGraceSec has no engine counterpart either: it is how long a dropped socket holds
	// its seat before something acts on the drop. Under ForfeitOnDisconnect that is the forfeit;
	// in a circuit round (created with ForfeitOnDisconnect off) it is the seat's takeover by the
	// turn clock instead (cambia-1233). 0 acts on the drop itself. The ceiling is an hour rather
	// than a day because the hub never reaps a lobby whose game is still running
	// (hub.handleIdleReap), so the grace is also how long an abandoned table can pin its lobby.
	disconnectGraceSecMin = 0
	disconnectGraceSecMax = 3600
)

// HouseRules defines optional game rules that can modify standard play. Every field maps to
// an engine.HouseRules field except ForfeitOnDisconnect and TurnTimerSec, which the service
// owns (the engine has no connection or wall-clock concept).
type HouseRules struct {
	AllowDrawFromDiscardPile bool `json:"allowDrawFromDiscardPile"` // Allow players to draw from the discard pile instead of the stockpile.
	AllowReplaceAbilities    bool `json:"allowReplaceAbilities"`    // Allow cards discarded via replacement to trigger their special abilities.
	AllowOpponentSnapping    bool `json:"allowOpponentSnapping"`    // Allow snapping a matching card out of an opponent's hand.
	SnapRace                 bool `json:"snapRace"`                 // Only the first player to successfully snap gets the benefit; others are penalized.
	LockCallerHand           bool `json:"lockCallerHand"`           // Protect the Cambia caller's hand from snaps, swaps and replacements (RULES.md 3C and 7).
	ForfeitOnDisconnect      bool `json:"forfeitOnDisconnect"`      // If a player disconnects, their game is forfeited. If false, they can rejoin.
	DisconnectGraceSec       int  `json:"disconnectGraceSec"`       // Seconds a dropped player keeps their seat before the forfeit lands, or before a circuit round takes the seat over (0 acts immediately).
	PenaltyDrawCount         int  `json:"penaltyDrawCount"`         // Number of cards to draw as penalty for an invalid snap.
	TurnTimerSec             int  `json:"turnTimerSec"`             // Duration (in seconds) for each player's turn (0 disables timer).
	MaxGameTurns             int  `json:"maxGameTurns"`             // Turn cap after which the game ends (0 = unlimited).
	CardsPerPlayer           int  `json:"cardsPerPlayer"`           // Cards dealt to each player (RULES.md 2 deals 4).
	CambiaAllowedRound       int  `json:"cambiaAllowedRound"`       // Earliest round in which Cambia may be called (0 = from the opening turn).
	NumJokers                int  `json:"numJokers"`                // Jokers shuffled into each deck copy (RULES.md 7 useJokers, as a count).
	NumDecks                 int  `json:"numDecks"`                 // Standard decks shuffled together.
	InitialViewCount         int  `json:"initialViewCount"`         // Cards each player peeks at during the pregame reveal (RULES.md 2 peeks 2).

	// numDecksExplicit records whether NumDecks came from an explicit "numDecks" key in an
	// Update call, as opposed to sitting at whatever DefaultHouseRules left it. Unexported so it
	// never crosses the wire: no client reads or writes it, and the JSON payload keeps carrying
	// only the resolved-or-default NumDecks value it already did. mapHouseRulesToEngine reads it
	// to decide whether an untouched NumDecks should resolve from the seated player count at game
	// creation (DefaultNumDecksForPlayers, MATCHMAKING.md 1.1, cambia-1564) rather than deal a
	// single-deck game to a 5-8 seat casual table out of a 21-card stockpile. A host who sets
	// NumDecks explicitly, including to a value equal to the default, keeps exactly that value at
	// every seat count.
	numDecksExplicit bool
}

// DefaultHouseRules returns the house rules a fresh lobby or game starts with. These values
// reproduce what mapHouseRulesToEngine used to hardcode, so a lobby that never touches the
// settings panel plays exactly as it did before the rules were exposed.
func DefaultHouseRules() HouseRules {
	return HouseRules{
		AllowDrawFromDiscardPile: false,
		AllowReplaceAbilities:    false,
		AllowOpponentSnapping:    true,
		SnapRace:                 false,
		LockCallerHand:           true,
		ForfeitOnDisconnect:      true,
		DisconnectGraceSec:       90,
		PenaltyDrawCount:         2,
		TurnTimerSec:             15,
		MaxGameTurns:             46,
		CardsPerPlayer:           4,
		CambiaAllowedRound:       0,
		NumJokers:                2,
		NumDecks:                 1,
		InitialViewCount:         2,
		numDecksExplicit:         false,
	}
}

// DefaultNumDecksForPlayers resolves the deck count a casual lobby deals from when its host has
// not set NumDecks explicitly, per MATCHMAKING.md 1.1: one deck for 2-4 seated players, two for
// 5-8 (the full range BeginPreGame admits, 2 through engine.MaxPlayers). Ranked queues seat 2 or
// 4 and so always resolve to one deck either way, matching MATCHMAKING.md 1's table.
func DefaultNumDecksForPlayers(numPlayers int) int {
	if numPlayers >= 5 {
		return 2
	}
	return 1
}

// Update applies changes from a map to the HouseRules struct.
// It validates input types and ranges where applicable. Nothing is committed until every check
// has passed, so a rejected update leaves the receiver exactly as it was.
func (rules *HouseRules) Update(newRules map[string]interface{}) error {
	// Assignments land on a copy and are committed at the bottom. The per-key helpers already
	// refuse an out-of-range value before writing it, but the cross-field check below cannot know
	// a value is bad until every key in the map has been applied, so it needs somewhere to roll
	// back to (cambia-817).
	next := *rules

	var err error // Declare error variable

	// Helper function to handle type assertion and assignment for booleans. The assertion
	// result is checked before the field is written: assigning through the two-value form
	// first would clobber the rule with false on a type error, leaving a mistyped update
	// half-applied for any caller that does not roll the struct back itself.
	assignBool := func(field *bool, key string) error {
		if val, exists := newRules[key]; exists && val != nil {
			parsed, ok := val.(bool)
			if !ok {
				return fmt.Errorf("invalid type for %s, expected boolean", key)
			}
			*field = parsed
		}
		return nil
	}

	// Helper function to handle type assertion and assignment for integers, rejecting any
	// value outside [minVal, maxVal]. The field is written only once the value is known to be
	// in range, so a rejected key never lands a partial value.
	assignInt := func(field *int, key string, minVal, maxVal int) error {
		val, exists := newRules[key]
		if !exists || val == nil {
			return nil
		}
		var parsed int
		// JSON numbers arrive as float64; accept a plain int too for in-process callers.
		if floatVal, isFloat := val.(float64); isFloat {
			// Check if float has fractional part before converting.
			if floatVal != math.Trunc(floatVal) {
				return fmt.Errorf("invalid value for %s, must be a whole number", key)
			}
			// Range-check in float space first: a value past the int range wraps on conversion.
			if floatVal < float64(minVal) || floatVal > float64(maxVal) {
				return fmt.Errorf("invalid value for %s: must be between %d and %d", key, minVal, maxVal)
			}
			parsed = int(floatVal)
		} else {
			intVal, isInt := val.(int)
			if !isInt {
				return fmt.Errorf("invalid type for %s, expected number", key)
			}
			parsed = intVal
		}
		if parsed < minVal || parsed > maxVal {
			return fmt.Errorf("invalid value for %s: must be between %d and %d", key, minVal, maxVal)
		}
		*field = parsed
		return nil
	}

	// Apply updates using helpers.
	if err = assignBool(&next.AllowDrawFromDiscardPile, "allowDrawFromDiscardPile"); err != nil {
		return err
	}
	if err = assignBool(&next.AllowReplaceAbilities, "allowReplaceAbilities"); err != nil {
		return err
	}
	if err = assignBool(&next.AllowOpponentSnapping, "allowOpponentSnapping"); err != nil {
		return err
	}
	if err = assignBool(&next.SnapRace, "snapRace"); err != nil {
		return err
	}
	if err = assignBool(&next.LockCallerHand, "lockCallerHand"); err != nil {
		return err
	}
	if err = assignBool(&next.ForfeitOnDisconnect, "forfeitOnDisconnect"); err != nil {
		return err
	}
	if err = assignInt(&next.DisconnectGraceSec, "disconnectGraceSec", disconnectGraceSecMin, disconnectGraceSecMax); err != nil {
		return err
	}
	if err = assignInt(&next.PenaltyDrawCount, "penaltyDrawCount", penaltyDrawCountMin, penaltyDrawCountMax); err != nil {
		return err
	}
	if err = assignInt(&next.TurnTimerSec, "turnTimerSec", turnTimerSecMin, turnTimerSecMax); err != nil {
		return err
	}
	if err = assignInt(&next.MaxGameTurns, "maxGameTurns", maxGameTurnsMin, maxGameTurnsMax); err != nil {
		return err
	}
	if err = assignInt(&next.CardsPerPlayer, "cardsPerPlayer", cardsPerPlayerMin, cardsPerPlayerMax); err != nil {
		return err
	}
	if err = assignInt(&next.CambiaAllowedRound, "cambiaAllowedRound", cambiaAllowedRoundMin, cambiaAllowedRoundMax); err != nil {
		return err
	}
	if err = assignInt(&next.NumJokers, "numJokers", numJokersMin, numJokersMax); err != nil {
		return err
	}
	if err = assignInt(&next.NumDecks, "numDecks", numDecksMin, numDecksMax); err != nil {
		return err
	}
	// Mirrors assignInt's own presence gate: a key that is absent or explicitly nil never wrote
	// NumDecks above and must not mark it explicit here either.
	if val, exists := newRules["numDecks"]; exists && val != nil {
		next.numDecksExplicit = true
	}
	if err = assignInt(&next.InitialViewCount, "initialViewCount", initialViewCountMin, initialViewCountMax); err != nil {
		return err
	}

	// Cross-field ceiling: the pregame peek cannot cover more cards than the hand holds. Deal()
	// clamps InitialViewCount to CardsPerPlayer rather than failing, so an over-large value would
	// otherwise be accepted at the lobby edge and silently play as something else. Checked after
	// both assignments because either key may arrive in the same update (cambia-817).
	if next.InitialViewCount > next.CardsPerPlayer {
		return fmt.Errorf("invalid value for initialViewCount: must be between %d and %d (cardsPerPlayer)", initialViewCountMin, next.CardsPerPlayer)
	}

	*rules = next
	return nil
}

// ParseRules is deprecated as HouseRules.Update provides the same functionality directly.
// Keeping for potential compatibility but recommend direct use of Update.
// Deprecated: Use the Update method directly on a HouseRules instance.
func ParseRules(rules map[string]interface{}, current HouseRules) (HouseRules, error) {
	houseRules := current
	err := houseRules.Update(rules)
	return houseRules, err
}
