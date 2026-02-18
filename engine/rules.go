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
	SnapRace              bool // if true, only one successful snap per discard
	NumJokers             uint8 // 0, 1, or 2 jokers in the deck
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
	}
}
