package engine

import (
	"fmt"
	"math/bits"
)

const (
	// MaxJokersPerDeck is the largest number of jokers one deck copy can
	// contribute. NewGame writes at most this many per copy no matter what
	// NumJokers says; Validate rejects anything above it at the FFI edge so
	// the two can never disagree (cambia-1555).
	MaxJokersPerDeck = 2

	// MaxDecks is the largest number of standard decks that fit the
	// fixed-size Stockpile/DiscardPile arrays.
	MaxDecks = MaxDeckSize / StandardDeckSize

	// NumSuitedRanks is the count of suited (non-joker) ranks: Ace through
	// King, RankAce..RankKing in the engine's rank order.
	NumSuitedRanks = RankKing + 1

	// AllDeckRanks is the DeckRanks mask selecting every suited rank, and
	// the effective mask a HouseRules with DeckRanks == 0 deals from.
	AllDeckRanks uint16 = (1 << NumSuitedRanks) - 1
)

// HouseRules holds configurable game rule settings.
type HouseRules struct {
	MaxGameTurns uint16 // 0 = unlimited
	// DeckRanks selects which suited ranks the deck is built from, as a
	// bitmask in the engine's rank order (bit 0 = RankAce ... bit 12 =
	// RankKing). 0 is the documented sentinel for "every suited rank", so a
	// zero-valued HouseRules keeps the standard 52-card composition and
	// every existing caller is unaffected. Jokers are governed by NumJokers
	// alone and have no bit here.
	//
	// cambia-1478: this is the reduced-deck rule the Python config calls
	// deck_ranks, crossed over the FFI. Before it existed the bridge dropped
	// deck_ranks silently, so a tiny-game config dealt 2 ranks on the Python
	// engine and all 13 on this one.
	DeckRanks             uint16
	CardsPerPlayer        uint8
	CambiaAllowedRound    uint8 // earliest round player can call Cambia
	PenaltyDrawCount      uint8
	AllowDrawFromDiscard  bool
	AllowReplaceAbilities bool // if true, replacing also triggers ability
	AllowOpponentSnapping bool
	SnapRace              bool  // snap model: false = race-OFF (sequential discarder-first, the frozen default); true = race-ON, the true N-way race (simultaneous imperfect-info commit, one uniform-random winner among willing snappers, losers penalized) per cambia-564
	NumJokers             uint8 // 0, 1, or 2 jokers in the deck
	LockCallerHand        bool  // if true, the Cambia caller cannot replace cards from hand
	NumPlayers            uint8 // number of active players (2-8); 0 treated as 2
	InitialViewCount      uint8 // how many cards each player peeks at game start (default 2)
	NumDecks              uint8 // number of standard decks shuffled together (1-4; 0 treated as 1)
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

// DeckRankMask returns the effective set of suited ranks the deck is built
// from: AllDeckRanks when DeckRanks is 0 (the documented sentinel), otherwise
// DeckRanks with any bit outside the 13 suited ranks dropped. Validate rejects
// such a bit at the FFI edge; the mask here is the belt-and-suspenders that
// keeps NewGame's rank loop in range for a caller that skipped Validate,
// mirroring what numPlayers() does for NumPlayers.
func (r *HouseRules) DeckRankMask() uint16 {
	if r.DeckRanks == 0 {
		return AllDeckRanks
	}
	return r.DeckRanks & AllDeckRanks
}

// numDecks returns the effective deck-copy count, treating 0 as 1 and clamping
// to MaxDecks. Same defense-in-depth role as numPlayers(): Validate is the
// rejection gate, but NewGame writes numDecks * 54 cards into a fixed
// [MaxDeckSize]Card array, so an unclamped NumDecks=5 panicked inside
// libcambia.so (cambia-1555).
func (r *HouseRules) numDecks() uint8 {
	n := r.NumDecks
	if n == 0 {
		return 1
	}
	if n > MaxDecks {
		return MaxDecks
	}
	return n
}

// numJokers returns the effective per-deck joker count, clamped to
// MaxJokersPerDeck. NewGame's joker loop has always stopped at 2; this
// accessor is what StockLen is now derived through, so the two agree
// (cambia-1555: StockLen was sized from the requested count, leaving an
// unwritten Card(0) - the Ace of Hearts - in the stockpile at NumJokers=3).
func (r *HouseRules) numJokers() uint8 {
	if r.NumJokers > MaxJokersPerDeck {
		return MaxJokersPerDeck
	}
	return r.NumJokers
}

// cardsPerPlayer returns the effective hand size, clamped to MaxHandSize.
// Defense in depth for the same reason as numPlayers(): Deal writes
// Hand[c] for c in [0, CardsPerPlayer) into a fixed [MaxHandSize]Card array.
// 0 is left alone - it means "deal nothing", which Deal handles - and
// Validate rejects it at the FFI edge.
func (r *HouseRules) cardsPerPlayer() uint8 {
	if r.CardsPerPlayer > MaxHandSize {
		return MaxHandSize
	}
	return r.CardsPerPlayer
}

// DeckSize returns how many cards NewGame will actually write into the
// stockpile under these rules: one copy per included suited rank per suit,
// plus the jokers, times the deck count.
func (r *HouseRules) DeckSize() int {
	perDeck := 4*bits.OnesCount16(r.DeckRankMask()) + int(r.numJokers())
	return int(r.numDecks()) * perDeck
}

// Validate reports whether the externally-supplied rule fields are within the
// supported ranges. Callers that accept raw, untrusted rules (the cgo
// constructors, which take uint8s straight from Python) must call this before
// building a GameState: nothing else in the engine bounds-checks them, and
// each one indexes a fixed-size array or sizes the deck.
//
//   - NumPlayers 2-MaxPlayers (0 = "default to 2"). Every [MaxPlayers]...
//     array indexed by player (Players, Snappers, the per-player score and
//     utility scratch arrays) panics on an out-of-range index rather than
//     failing gracefully (cambia-542 F3).
//   - CardsPerPlayer 1-MaxHandSize. Deal writes Hand[c] into a fixed
//     [MaxHandSize]Card array; 7 panicked there (cambia-1555).
//   - InitialViewCount at most CardsPerPlayer. Peeking more cards than a hand
//     holds is not a game the deal can produce.
//   - NumJokers 0-MaxJokersPerDeck. NewGame writes at most 2 per deck copy,
//     so 3 previously left an unwritten Card(0) in the stockpile.
//   - NumDecks 1-MaxDecks (0 = "default to 1"). NewGame writes
//     NumDecks * 54 cards into a fixed [MaxDeckSize]Card array; 5 panicked.
//   - DeckRanks limited to the 13 suited-rank bits, and the resulting deck
//     large enough to complete the deal (cambia-1478).
//
// Rejection, never clamping: a clamped rule silently plays a different game
// than the one configured, which is exactly the failure cambia-1478 records.
// The accessors above still clamp, as belt-and-suspenders for a caller that
// builds HouseRules directly and skips Validate.
//
// Deliberately inspects the raw fields rather than going through those
// accessors: reading through them here would make every out-of-range branch
// below dead code.
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
	if r.CardsPerPlayer < 1 {
		return fmt.Errorf("HouseRules.CardsPerPlayer: %d is below the minimum of 1", r.CardsPerPlayer)
	}
	if r.CardsPerPlayer > MaxHandSize {
		return fmt.Errorf("HouseRules.CardsPerPlayer: %d exceeds MaxHandSize (%d)", r.CardsPerPlayer, MaxHandSize)
	}
	if r.InitialViewCount > r.CardsPerPlayer {
		return fmt.Errorf(
			"HouseRules.InitialViewCount: %d exceeds CardsPerPlayer (%d)",
			r.InitialViewCount, r.CardsPerPlayer,
		)
	}
	if r.NumJokers > MaxJokersPerDeck {
		return fmt.Errorf(
			"HouseRules.NumJokers: %d exceeds the maximum of %d per deck",
			r.NumJokers, MaxJokersPerDeck,
		)
	}
	if r.NumDecks > MaxDecks {
		return fmt.Errorf("HouseRules.NumDecks: %d exceeds the maximum of %d", r.NumDecks, MaxDecks)
	}
	// DeckRanks == 0 is the all-ranks sentinel, so the only way to write "no
	// ranks at all" is to set bits that are all outside the suited range
	// (e.g. the joker bit); that is what this first check catches. A mask
	// mixing valid and out-of-range bits falls to the second, since either
	// one is a typo and silently masking it off would deal a game other than
	// the configured one.
	if r.DeckRankMask() == 0 {
		return fmt.Errorf("HouseRules.DeckRanks: mask %#04x selects no suited rank", r.DeckRanks)
	}
	if r.DeckRanks&^AllDeckRanks != 0 {
		return fmt.Errorf(
			"HouseRules.DeckRanks: mask %#04x sets bits outside the %d suited ranks "+
				"(jokers are governed by NumJokers)",
			r.DeckRanks, NumSuitedRanks,
		)
	}
	// The deal takes CardsPerPlayer per seat and flips one more card to open
	// the discard pile; a deck short of that underflows StockLen to 255.
	need := int(n)*int(r.CardsPerPlayer) + 1
	if size := r.DeckSize(); size < need {
		return fmt.Errorf(
			"HouseRules: the deck holds %d cards but the deal needs %d "+
				"(%d players x %d cards, plus 1 for the opening discard)",
			size, need, n, r.CardsPerPlayer,
		)
	}
	return nil
}
