package engine

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
	NumPlayers            uint8 // number of active players (2–6); 0 treated as 2
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
	}
}

// numPlayers returns the effective number of players, treating 0 as 2.
func (r *HouseRules) numPlayers() uint8 {
	if r.NumPlayers == 0 {
		return 2
	}
	return r.NumPlayers
}
