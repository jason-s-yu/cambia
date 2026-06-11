package agent

import engine "github.com/jason-s-yu/cambia/engine"

// CardBucket represents abstract categories that cards are mapped into.
// Values match the Python CardBucket enum in cfr/src/constants.py (0-8 for known
// buckets, 9 for unknown — compacted from Python's 99 to fit uint8 iota).
type CardBucket uint8

const (
	BucketZero      CardBucket = iota // 0: Joker (value 0)
	BucketNegKing                     // 1: Red King (value -1)
	BucketAce                         // 2: Ace (value 1)
	BucketLowNum                      // 3: 2-4 (values 2-4)
	BucketMidNum                      // 4: 5-6 (values 5-6)
	BucketPeekSelf                    // 5: 7-8 (peek own ability)
	BucketPeekOther                   // 6: 9-10 (peek other ability)
	BucketSwapBlind                   // 7: J-Q (blind swap ability)
	BucketHighKing                    // 8: Black King (value 13)
	BucketUnknown                     // 9: Unknown card
)

// DecayCategory represents broader categories used when belief decays.
// Values match Python DecayCategory ordinal order (0-3).
type DecayCategory uint8

const (
	DecayLikelyLow  DecayCategory = iota // 0: Joker/NegKing/Ace/2-4
	DecayLikelyMid                        // 1: 5-8
	DecayLikelyHigh                       // 2: 9-T/J-Q/BlackKing
	DecayUnknown                          // 3: Fully decayed
)

// StockpileEstimate represents abstract remaining stockpile size.
// Values match Python StockpileEstimate enum (0-3).
type StockpileEstimate uint8

const (
	StockHigh   StockpileEstimate = iota // 0: >30 cards
	StockMedium                          // 1: 15-30 cards
	StockLow                             // 2: 1-14 cards
	StockEmpty                           // 3: 0 cards
)

// GamePhase represents the abstract phase of the game.
// Values match Python GamePhase enum (0-5).
type GamePhase uint8

const (
	PhaseStart        GamePhase = iota // 0: Turn 0
	PhaseEarly                         // 1: Turns 1-8
	PhaseMid                           // 2: Turns 9-20
	PhaseLate                          // 3: Turns 21+
	PhaseCambiaCalled                  // 4: Cambia has been called
	PhaseTerminal                      // 5: Game over
)

// CambiaState encodes cambia-call status from the agent's perspective.
type CambiaState uint8

const (
	CambiaNone     CambiaState = iota // 0: No one called
	CambiaSelf                        // 1: This agent called
	CambiaOpponent                    // 2: Opponent called
)

// CardToBucket maps a Card to its CardBucket based on rank and suit.
// Jokers → BucketZero, Red Kings → BucketNegKing, Black Kings → BucketHighKing,
// Ace → BucketAce, 2-4 → BucketLowNum, 5-6 → BucketMidNum,
// 7-8 → BucketPeekSelf, 9-T → BucketPeekOther, J-Q → BucketSwapBlind.
func CardToBucket(c engine.Card) CardBucket {
	r := c.Rank()
	switch {
	case r == engine.RankJoker:
		return BucketZero
	case r == engine.RankKing:
		s := c.Suit()
		if s == engine.SuitHearts || s == engine.SuitDiamonds {
			return BucketNegKing
		}
		return BucketHighKing
	case r == engine.RankAce:
		return BucketAce
	case r >= engine.RankTwo && r <= engine.RankFour:
		return BucketLowNum
	case r == engine.RankFive || r == engine.RankSix:
		return BucketMidNum
	case r == engine.RankSeven || r == engine.RankEight:
		return BucketPeekSelf
	case r == engine.RankNine || r == engine.RankTen:
		return BucketPeekOther
	case r == engine.RankJack || r == engine.RankQueen:
		return BucketSwapBlind
	}
	return BucketUnknown
}

// EpistemicTag tracks who knows about a card slot's identity.
// Used by the EP-PBS (Epistemic Positional Belief State) encoding.
type EpistemicTag uint8

const (
	TagUnk     EpistemicTag = iota // Nobody relevant has observed
	TagPrivOwn                     // I privately know this card
	TagPrivOpp                     // Opponent privately knows it
	TagPub                         // Common knowledge (both players know)
)

const (
	EPPBSInputDim = 224 // Total dimensions for EncodeEPPBS output (was 200; +24 for obs ages, discard hist, turn progress)
	MaxSlots      = 12  // 6 own + 6 opp for 2P (engine.MaxHandSize * 2)
	OppSlotsStart = 6   // Opp slots start at index 6 in SlotTags/SlotBuckets
	MaxActiveMask = 3   // Max entries tracked in OwnActiveMask / OppActiveMask
	NumBuckets    = 10  // Number of card buckets (BucketZero through BucketUnknown)
)

// Encoding v2 constants (DESCA Phase 0 foundation).
const (
	// ActionHistorySize is the per-side ring buffer depth for the v2 encoding
	// action history window (last 3 observed own + 3 observed opp actions).
	ActionHistorySize = 3

	// V2CardCountDim is the number of card-counting posterior dimensions
	// (one per CardBucket 0..8; BucketUnknown is excluded).
	V2CardCountDim = 9

	// V2ActionHistoryEntryDim is the per-entry dimension of the action history
	// window: 3-one-hot category + 1 scalar target_slot_norm.
	V2ActionHistoryEntryDim = 4

	// V2ActionHistoryDim is the total action history window size
	// (6 entries = 3 own + 3 opp, each V2ActionHistoryEntryDim wide).
	V2ActionHistoryDim = 2 * ActionHistorySize * V2ActionHistoryEntryDim

	// V2CardCountOffset is the starting index of the posterior in the v2 vector.
	V2CardCountOffset = EPPBSInputDim

	// V2ActionHistoryOffset is the starting index of the action history window.
	V2ActionHistoryOffset = V2CardCountOffset + V2CardCountDim

	// EPPBSV2InputDim is the total v2 encoding dimension (expected 257).
	EPPBSV2InputDim = EPPBSInputDim + V2CardCountDim + V2ActionHistoryDim
)

// DeckBucketCounts lists the maximum number of cards per CardBucket
// across the full 54-card deck. Index aligns with CardBucket (0..8):
//
//	0: BucketZero (Jokers)       2
//	1: BucketNegKing (Red Kings) 2
//	2: BucketAce                 4
//	3: BucketLowNum (2-4)        12
//	4: BucketMidNum (5-6)        8
//	5: BucketPeekSelf (7-8)      8
//	6: BucketPeekOther (9-10)    8
//	7: BucketSwapBlind (J-Q)     8
//	8: BucketHighKing (Black K)  2
//
// Mirrors cfr/src/pbs.py DECK_COUNTS.
var DeckBucketCounts = [V2CardCountDim]uint8{2, 2, 4, 12, 8, 8, 8, 8, 2}

// Action category constants for the v2 encoding action history window.
// Values MUST stay in sync with cfr/src/constants.py V2_ACTION_CATEGORY_*.
//
//	0 = DRAW phase: DrawStockpile, DrawDiscard, CallCambia
//	1 = DISCARD phase: Discard*, Replace
//	2 = ABILITY_OR_SNAP: peek/look/swap (including king swap), snap_*, pass_snap
const (
	ActionCategoryDraw         uint8 = 0
	ActionCategoryDiscard      uint8 = 1
	ActionCategoryAbilityOrSnap uint8 = 2
)

// CategorizeAction maps a 2P action index to the v2 encoding tuple:
// (category, targetSlot, hasSlot). See ActionCategory* constants for semantics.
//
// The targetSlot is the primary hand-slot index (0..engine.MaxHandSize-1)
// associated with the action (own slot for Replace/PeekOwn/BlindSwap/KingLook/
// SnapOwn/SnapOpponentMove; opponent slot for PeekOther/SnapOpponent). For
// actions without a slot target, hasSlot is false and targetSlot is 0.
//
// Unknown or out-of-range action indices yield category AbilityOrSnap with
// hasSlot=false; the category one-hot will still be written but the slot
// will stay at zero.
func CategorizeAction(actionIdx uint16) (category uint8, targetSlot uint8, hasSlot bool) {
	switch actionIdx {
	case engine.ActionDrawStockpile,
		engine.ActionDrawDiscard,
		engine.ActionCallCambia:
		return ActionCategoryDraw, 0, false
	case engine.ActionDiscardNoAbility,
		engine.ActionDiscardWithAbility:
		return ActionCategoryDiscard, 0, false
	case engine.ActionKingSwapNo, engine.ActionKingSwapYes, engine.ActionPassSnap:
		return ActionCategoryAbilityOrSnap, 0, false
	}

	if slot, ok := engine.ActionIsReplace(actionIdx); ok {
		return ActionCategoryDiscard, slot, true
	}
	if slot, ok := engine.ActionIsPeekOwn(actionIdx); ok {
		return ActionCategoryAbilityOrSnap, slot, true
	}
	if slot, ok := engine.ActionIsPeekOther(actionIdx); ok {
		return ActionCategoryAbilityOrSnap, slot, true
	}
	if ownIdx, _, ok := engine.ActionIsBlindSwap(actionIdx); ok {
		return ActionCategoryAbilityOrSnap, ownIdx, true
	}
	if ownIdx, _, ok := engine.ActionIsKingLook(actionIdx); ok {
		return ActionCategoryAbilityOrSnap, ownIdx, true
	}
	if slot, ok := engine.ActionIsSnapOwn(actionIdx); ok {
		return ActionCategoryAbilityOrSnap, slot, true
	}
	if slot, ok := engine.ActionIsSnapOpponent(actionIdx); ok {
		return ActionCategoryAbilityOrSnap, slot, true
	}
	if ownIdx, _, ok := engine.ActionIsSnapOpponentMove(actionIdx); ok {
		return ActionCategoryAbilityOrSnap, ownIdx, true
	}
	return ActionCategoryAbilityOrSnap, 0, false
}

// N-Player constants (used when NumPlayers > 2).
// Derived from MaxPlayers=8 and MaxHandSize=6:
//   MaxTotalSlots       = MaxPlayers × MaxHandSize = 8 × 6 = 48
//   MaxKnowledgePlayers = MaxPlayers = 8
//   NPlayerPowersetDim  = MaxTotalSlots × MaxKnowledgePlayers = 48 × 8 = 384
//   NPlayerIdentityDim  = MaxTotalSlots × 9 (buckets) = 48 × 9 = 432
//   NPlayerInputDim     = NPlayerPowersetDim + NPlayerIdentityDim + 40 (public) = 856
//   NPlayerNumActions   = 620 (see engine.NPlayerNumActions; scales with MaxOpponents=7)
const (
	MaxTotalSlots       = 48                           // 8 players × 6 cards
	MaxKnowledgePlayers = engine.MaxPlayers            // 8
	NPlayerPowersetDim  = MaxTotalSlots * MaxKnowledgePlayers // 48 × 8 = 384
	NPlayerIdentityDim  = MaxTotalSlots * 9            // 48 × 9 = 432
	NPlayerInputDim     = NPlayerPowersetDim + NPlayerIdentityDim + 40 // 856
	NPlayerNumActions   = 620                          // matches engine.NPlayerNumActions
)

// MemoryArchetype defines how the agent handles memory decay and eviction.
type MemoryArchetype uint8

const (
	MemoryPerfect   MemoryArchetype = iota // No decay or eviction; retains all observations
	MemoryDecaying                          // Bayesian diffusion: PrivOwn slots decay with prob p = 1-exp(-λ)
	MemoryHumanLike                         // Stochastic saliency eviction; OwnActiveMask capped at MemoryCapacity
)

// BucketMidpoint returns the approximate midpoint card value for a CardBucket.
// Used for saliency-based eviction in OwnActiveMask.
func BucketMidpoint(b CardBucket) float32 {
	switch b {
	case BucketZero:
		return 0.0
	case BucketNegKing:
		return -1.0
	case BucketAce:
		return 1.0
	case BucketLowNum:
		return 3.0 // 2-4, midpoint 3
	case BucketMidNum:
		return 5.5 // 5-6, midpoint 5.5
	case BucketPeekSelf:
		return 7.5 // 7-8, midpoint 7.5
	case BucketPeekOther:
		return 9.5 // 9-T, midpoint 9.5
	case BucketSwapBlind:
		return 11.5 // J-Q, midpoint 11.5
	case BucketHighKing:
		return 13.0
	default: // BucketUnknown
		return 4.5 // Zero saliency → most evictable
	}
}

// BucketSaliency returns |BucketMidpoint(b) - 4.5|. Higher = more salient.
// Slots with LOWER saliency are evicted first from OwnActiveMask.
func BucketSaliency(b CardBucket) float32 {
	d := BucketMidpoint(b) - 4.5
	if d < 0 {
		d = -d
	}
	return d
}

// BucketToDecay maps a CardBucket to its DecayCategory.
func BucketToDecay(b CardBucket) DecayCategory {
	switch b {
	case BucketZero, BucketNegKing, BucketAce, BucketLowNum:
		return DecayLikelyLow
	case BucketMidNum, BucketPeekSelf:
		return DecayLikelyMid
	case BucketPeekOther, BucketSwapBlind, BucketHighKing:
		return DecayLikelyHigh
	default:
		return DecayUnknown
	}
}

// StockEstimateFromSize converts a stockpile card count to a StockpileEstimate.
// Thresholds match Python's _estimate_stockpile for a 54-card deck:
//
//	low_threshold = max(1, 54 // 5) = 10
//	med_threshold = max(11, 54 * 2 // 4) = 27
//
// >=27 → StockHigh, >=10 → StockMedium, >0 → StockLow, 0 → StockEmpty.
func StockEstimateFromSize(stockLen uint8) StockpileEstimate {
	switch {
	case stockLen >= 27:
		return StockHigh
	case stockLen >= 10:
		return StockMedium
	case stockLen > 0:
		return StockLow
	default:
		return StockEmpty
	}
}

// GamePhaseFromState derives the GamePhase from stockpile estimate and game flags.
// Matches Python's _estimate_game_phase which uses stockpile size (not turn number).
// Priority: gameOver > cambiaCalled > stockpile-based thresholds.
func GamePhaseFromState(stockLen uint8, cambiaCalled bool, gameOver bool) GamePhase {
	switch {
	case gameOver:
		return PhaseTerminal
	case cambiaCalled:
		return PhaseCambiaCalled
	default:
		est := StockEstimateFromSize(stockLen)
		switch est {
		case StockHigh:
			return PhaseEarly
		case StockMedium:
			return PhaseMid
		default: // StockLow or StockEmpty
			return PhaseLate
		}
	}
}
