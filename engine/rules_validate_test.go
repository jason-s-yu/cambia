package engine

import "testing"

// The rule bounds Validate() gained in cambia-1555 and cambia-1478, exercised
// directly (no cgo, no GameState) so the invalid cases never touch the
// array-index panic paths they used to leave reachable. TestHouseRulesValidate
// in rules_test.go covers NumPlayers, the cambia-542 F3 precedent these follow:
// reject, never clamp, so a game is either the one that was configured or no
// game at all.

// TestHouseRulesValidateCardsPerPlayer covers the hand-size bound. Deal writes
// Hand[c] into a fixed [MaxHandSize]Card array, so CardsPerPlayer=7 panicked
// inside libcambia.so. 0 is refused too: a hand of nothing is not a game a
// caller means to configure.
func TestHouseRulesValidateCardsPerPlayer(t *testing.T) {
	cases := []struct {
		name    string
		cpp     uint8
		wantErr bool
	}{
		{"zero_rejected", 0, true},
		{"one_minimum", 1, false},
		{"four_default", 4, false},
		{"six_maximum", MaxHandSize, false},
		{"seven_exceeds_maximum", MaxHandSize + 1, true},
		{"max_uint8", 255, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := DefaultHouseRules()
			r.CardsPerPlayer = c.cpp
			r.InitialViewCount = 0 // isolate the field under test
			err := r.Validate()
			if c.wantErr && err == nil {
				t.Errorf("CardsPerPlayer=%d: expected an error, got nil", c.cpp)
			}
			if !c.wantErr && err != nil {
				t.Errorf("CardsPerPlayer=%d: unexpected error: %v", c.cpp, err)
			}
		})
	}
}

// TestHouseRulesValidateInitialViewCount covers the peek bound: no player can
// be dealt a peek at a card the hand does not hold.
func TestHouseRulesValidateInitialViewCount(t *testing.T) {
	cases := []struct {
		name    string
		ivc     uint8
		wantErr bool
	}{
		{"none", 0, false},
		{"two_default", 2, false},
		{"equal_to_the_hand", 4, false},
		{"one_above_the_hand", 5, true},
		{"max_uint8", 255, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := DefaultHouseRules() // CardsPerPlayer = 4
			r.InitialViewCount = c.ivc
			err := r.Validate()
			if c.wantErr && err == nil {
				t.Errorf("InitialViewCount=%d: expected an error, got nil", c.ivc)
			}
			if !c.wantErr && err != nil {
				t.Errorf("InitialViewCount=%d: unexpected error: %v", c.ivc, err)
			}
		})
	}
}

// TestHouseRulesValidateNumJokers covers the joker bound: the deck build
// writes at most MaxJokersPerDeck per copy, so a higher count asked for cards
// the deck never held and left an unwritten slot in the stockpile.
func TestHouseRulesValidateNumJokers(t *testing.T) {
	cases := []struct {
		name    string
		jokers  uint8
		wantErr bool
	}{
		{"none", 0, false},
		{"one", 1, false},
		{"two_maximum", MaxJokersPerDeck, false},
		{"three_exceeds_maximum", MaxJokersPerDeck + 1, true},
		{"max_uint8", 255, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := DefaultHouseRules()
			r.NumJokers = c.jokers
			err := r.Validate()
			if c.wantErr && err == nil {
				t.Errorf("NumJokers=%d: expected an error, got nil", c.jokers)
			}
			if !c.wantErr && err != nil {
				t.Errorf("NumJokers=%d: unexpected error: %v", c.jokers, err)
			}
		})
	}
}

// TestHouseRulesValidateNumDecks covers the deck-count bound: NewGame writes
// NumDecks * StandardDeckSize cards into a fixed [MaxDeckSize]Card array, so 5
// panicked there. 0 keeps its documented "default to 1" meaning.
func TestHouseRulesValidateNumDecks(t *testing.T) {
	cases := []struct {
		name    string
		decks   uint8
		wantErr bool
	}{
		{"zero_defaults_to_one", 0, false},
		{"one", 1, false},
		{"four_maximum", MaxDecks, false},
		{"five_exceeds_maximum", MaxDecks + 1, true},
		{"max_uint8", 255, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := DefaultHouseRules()
			r.NumDecks = c.decks
			err := r.Validate()
			if c.wantErr && err == nil {
				t.Errorf("NumDecks=%d: expected an error, got nil", c.decks)
			}
			if !c.wantErr && err != nil {
				t.Errorf("NumDecks=%d: unexpected error: %v", c.decks, err)
			}
		})
	}
}

// TestHouseRulesValidateDeckRanks covers the reduced-deck mask (cambia-1478):
// the 0 sentinel and any subset of the 13 suited ranks are accepted, a mask
// naming no suited rank or setting a bit outside them is not.
func TestHouseRulesValidateDeckRanks(t *testing.T) {
	cases := []struct {
		name    string
		mask    uint16
		wantErr bool
	}{
		{"zero_is_the_all_ranks_sentinel", 0, false},
		{"every_suited_rank", AllDeckRanks, false},
		{"ace_and_six", (1 << RankAce) | (1 << RankSix), false},
		{"joker_bit_alone_selects_nothing", 1 << RankJoker, true},
		{"high_bit_alone_selects_nothing", 1 << 15, true},
		{"joker_bit_alongside_a_rank", (1 << RankJoker) | (1 << RankAce), true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := DefaultHouseRules()
			r.CardsPerPlayer = 2
			r.InitialViewCount = 1
			r.DeckRanks = c.mask
			err := r.Validate()
			if c.wantErr && err == nil {
				t.Errorf("DeckRanks=%#04x: expected an error, got nil", c.mask)
			}
			if !c.wantErr && err != nil {
				t.Errorf("DeckRanks=%#04x: unexpected error: %v", c.mask, err)
			}
		})
	}
}

// TestHouseRulesValidateDeckTooSmallForTheDeal rejects rules whose deck cannot
// cover CardsPerPlayer per seat plus the opening discard flip (cambia-1478).
// Without the check Deal underflows StockLen from 0 to 255 and deals whatever
// the rest of the Stockpile array happens to hold.
func TestHouseRulesValidateDeckTooSmallForTheDeal(t *testing.T) {
	// One rank, no jokers: 4 cards. Two seats at 2 cards each need 5.
	r := DefaultHouseRules()
	r.DeckRanks = 1 << RankAce
	r.NumJokers = 0
	r.CardsPerPlayer = 2
	r.InitialViewCount = 1
	if err := r.Validate(); err == nil {
		t.Errorf("a %d-card deck dealing 2x2+1: expected an error, got nil", r.DeckSize())
	}

	// The same deal against two ranks (8 cards) fits.
	r.DeckRanks = (1 << RankAce) | (1 << RankSix)
	if err := r.Validate(); err != nil {
		t.Errorf("an %d-card deck dealing 2x2+1: unexpected error: %v", r.DeckSize(), err)
	}

	// Jokers count toward the deal, so one rank plus two jokers fits as well.
	r.DeckRanks = 1 << RankAce
	r.NumJokers = 2
	if err := r.Validate(); err != nil {
		t.Errorf("a %d-card deck dealing 2x2+1: unexpected error: %v", r.DeckSize(), err)
	}

	// The largest standard deal (8 seats x MaxHandSize = 49 cards) still fits
	// a single 54-card deck, so the check never fires on real rules.
	full := DefaultHouseRules()
	full.NumPlayers = MaxPlayers
	full.CardsPerPlayer = MaxHandSize
	if err := full.Validate(); err != nil {
		t.Errorf("8 seats x %d cards from a full deck: unexpected error: %v", MaxHandSize, err)
	}
}
