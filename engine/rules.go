package engine

import "fmt"

// HouseRules holds configurable game rule settings.
type HouseRules struct {
	MaxGameTurns          uint16 // 0 = unlimited
	CardsPerPlayer        uint8
	CambiaAllowedRound    uint8 // earliest round player can call Cambia
	PenaltyDrawCount      uint8
	AllowDrawFromDiscard  bool
	AllowReplaceAbilities bool // if true, replacing also triggers ability
	AllowOpponentSnapping bool
	SnapRace              bool  // if true, only one successful snap per discard
	NumJokers             uint8 // 0, 1, or 2 jokers in the deck
	LockCallerHand        bool  // if true, the Cambia caller cannot replace cards from hand
	NumPlayers            uint8 // number of active players (2–8); 0 treated as 2
	InitialViewCount      uint8 // how many cards each player peeks at game start (default 2)
	NumDecks              uint8 // number of standard decks shuffled together (1–4; 0 treated as 1)
}

// DefaultHouseRules returns the standard Cambia house rules.
func DefaultHouseRules() HouseRules {
	return HouseRules{
		MaxGameTurns:          46,
		CardsPerPlayer:        4,
		CambiaAllowedRound:    0,
		PenaltyDrawCount:      2,
		AllowDrawFromDiscard:  true,
		AllowReplaceAbilities: false,
		AllowOpponentSnapping: true,
		SnapRace:              false,
		NumJokers:             2,
		LockCallerHand:        true,
		NumPlayers:            2,
		InitialViewCount:      2,
		NumDecks:              1,
	}
}

// numPlayers returns the effective number of players, treating 0 as 2 and
// clamping to [2, MaxPlayers]. The clamp is defense-in-depth for cambia-542
// F3: Validate() is the intended reject-invalid-input gate at construction,
// but every [MaxPlayers]-sized array access in the engine (Players, Snappers,
// score/utility scratch arrays) flows through this accessor, so clamping here
// guarantees no out-of-range HouseRules can reach a panicking index no matter
// how it was constructed (e.g. a caller that builds HouseRules directly and
// skips Validate()).
func (r *HouseRules) numPlayers() uint8 {
	n := r.NumPlayers
	if n == 0 {
		return 2
	}
	if n < 2 {
		return 2
	}
	if n > MaxPlayers {
		return MaxPlayers
	}
	return n
}

// Validate reports whether NumPlayers is within the supported range
// (2-MaxPlayers; 0 is a valid sentinel meaning "default to 2"). Callers that
// accept a raw, externally-supplied player count (the cgo constructors, which
// take an untrusted uint8 straight from Python) must call this before
// building a GameState: nothing else in the engine bounds-checks NumPlayers,
// and every fixed-size [MaxPlayers]... array indexed by player (Players,
// Snappers, the per-player score/utility scratch arrays) panics on an
// out-of-range index rather than failing gracefully (cambia-542 F3).
//
// Deliberately inspects the raw NumPlayers field rather than going through
// the numPlayers() accessor: that accessor clamps out-of-range values (a
// defense-in-depth belt-and-suspenders for callers that skip Validate()), so
// reading through it here would make the >MaxPlayers branch below dead code.
func (r *HouseRules) Validate() error {
	n := r.NumPlayers
	if n == 0 {
		n = 2 // documented default sentinel, always valid
	}
	if n < 2 {
		return fmt.Errorf("HouseRules.NumPlayers: %d players is below the minimum of 2", n)
	}
	if n > MaxPlayers {
		return fmt.Errorf("HouseRules.NumPlayers: %d players exceeds MaxPlayers (%d)", n, MaxPlayers)
	}
	return nil
}
