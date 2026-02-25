package engine

// Suit constants — packed into upper 4 bits of Card.
const (
	SuitHearts    uint8 = 0
	SuitDiamonds  uint8 = 1
	SuitClubs     uint8 = 2
	SuitSpades    uint8 = 3
	SuitRedJoker  uint8 = 4
	SuitBlackJoker uint8 = 5
)

// Rank constants — packed into lower 4 bits of Card.
const (
	RankAce   uint8 = 0
	RankTwo   uint8 = 1
	RankThree uint8 = 2
	RankFour  uint8 = 3
	RankFive  uint8 = 4
	RankSix   uint8 = 5
	RankSeven uint8 = 6
	RankEight uint8 = 7
	RankNine  uint8 = 8
	RankTen   uint8 = 9
	RankJack  uint8 = 10
	RankQueen uint8 = 11
	RankKing  uint8 = 12
	RankJoker uint8 = 13
)

// Card is a packed uint8: upper 4 bits = suit, lower 4 bits = rank.
type Card uint8

// EmptyCard represents the absence of a card.
const EmptyCard Card = 0xFF

// NewCard constructs a Card from suit and rank.
func NewCard(suit, rank uint8) Card {
	return Card((suit << 4) | (rank & 0x0F))
}

// Suit returns the suit bits (upper 4).
func (c Card) Suit() uint8 { return uint8(c) >> 4 }

// Rank returns the rank bits (lower 4).
func (c Card) Rank() uint8 { return uint8(c) & 0x0F }

// Value returns the point value of the card.
//   - Joker (rank 13) → 0
//   - Ace (rank 0) → 1
//   - Two–Nine (ranks 1–8) → rank+1
//   - Ten (rank 9) → 10
//   - Jack (rank 10) → 11
//   - Queen (rank 11) → 12
//   - King (rank 12): Red (Hearts/Diamonds) → -1, Black (Clubs/Spades) → 13
func (c Card) Value() int8 {
	r := c.Rank()
	switch {
	case r == RankJoker:
		return 0
	case r == RankAce:
		return 1
	case r <= RankNine: // ranks 1–8: Two–Nine
		return int8(r + 1)
	case r == RankTen:
		return 10
	case r == RankJack:
		return 11
	case r == RankQueen:
		return 12
	case r == RankKing:
		s := c.Suit()
		if s == SuitHearts || s == SuitDiamonds {
			return -1
		}
		return 13
	}
	// EmptyCard or malformed — return 0
	return 0
}

// HasAbility returns true for ranks Seven through King (6–12).
func (c Card) HasAbility() bool {
	r := c.Rank()
	return r >= RankSeven && r <= RankKing
}

// AbilityType represents the kind of special ability a card grants.
type AbilityType uint8

const (
	AbilityNone      AbilityType = iota // 0
	AbilityPeekOwn                      // 1 — Seven, Eight
	AbilityPeekOther                    // 2 — Nine, Ten
	AbilityBlindSwap                    // 3 — Jack, Queen
	AbilityKingLook                     // 4 — King
)

// Ability returns the ability associated with discarding this card.
func (c Card) Ability() AbilityType {
	switch c.Rank() {
	case RankSeven, RankEight:
		return AbilityPeekOwn
	case RankNine, RankTen:
		return AbilityPeekOther
	case RankJack, RankQueen:
		return AbilityBlindSwap
	case RankKing:
		return AbilityKingLook
	default:
		return AbilityNone
	}
}

// DecisionContext describes what kind of decision a player must make.
type DecisionContext uint8

const (
	CtxStartTurn    DecisionContext = iota // 0
	CtxPostDraw                            // 1
	CtxSnapDecision                        // 2
	CtxAbilitySelect                       // 3
	CtxSnapMove                            // 4
	CtxTerminal                            // 5
)

// PendingType describes a pending ability or action waiting for resolution.
type PendingType uint8

const (
	PendingNone        PendingType = iota // 0
	PendingDiscard                        // 1
	PendingPeekOwn                        // 2
	PendingPeekOther                      // 3
	PendingBlindSwap                      // 4
	PendingKingLook                       // 5
	PendingKingDecision                   // 6
	PendingSnapMove                       // 7
)

// PendingAction holds a pending ability action waiting to be resolved.
type PendingAction struct {
	Type     PendingType
	PlayerID uint8
	Data     [4]uint8 // ability-specific data
}

// SnapState tracks the state of an active snap window.
type SnapState struct {
	Active            bool
	DiscardedRank     uint8
	Snappers          [MaxPlayers]uint8 // up to MaxPlayers snappers
	NumSnappers       uint8
	CurrentSnapperIdx uint8
}

// ---------------------------------------------------------------------------
// Action index constants
// ---------------------------------------------------------------------------

const (
	ActionDrawStockpile    uint16 = 0
	ActionDrawDiscard      uint16 = 1
	ActionCallCambia       uint16 = 2
	ActionDiscardNoAbility uint16 = 3
	ActionDiscardWithAbility uint16 = 4

	ActionBaseReplace          uint16 = 5   // Replace(0)..Replace(5)
	ActionBasePeekOwn          uint16 = 11  // PeekOwn(0)..PeekOwn(5)
	ActionBasePeekOther        uint16 = 17  // PeekOther(0)..PeekOther(5)
	ActionBaseBlindSwap        uint16 = 23  // BlindSwap(own*6+opp), 36 entries
	ActionBaseKingLook         uint16 = 59  // KingLook(own*6+opp), 36 entries
	ActionKingSwapNo           uint16 = 95
	ActionKingSwapYes          uint16 = 96
	ActionPassSnap             uint16 = 97
	ActionBaseSnapOwn          uint16 = 98  // SnapOwn(0)..SnapOwn(5)
	ActionBaseSnapOpponent     uint16 = 104 // SnapOpponent(0)..SnapOpponent(5)
	ActionBaseSnapOpponentMove uint16 = 110 // SnapOpponentMove(own*6+slot), 36 entries

	NumActions uint16 = 146
)

// ---------------------------------------------------------------------------
// Encode functions
// ---------------------------------------------------------------------------

// EncodeReplace returns the action index for replacing hand card at targetIdx.
func EncodeReplace(targetIdx uint8) uint16 { return ActionBaseReplace + uint16(targetIdx) }

// EncodePeekOwn returns the action index for peeking own card at targetIdx.
func EncodePeekOwn(targetIdx uint8) uint16 { return ActionBasePeekOwn + uint16(targetIdx) }

// EncodePeekOther returns the action index for peeking opponent card at targetIdx.
func EncodePeekOther(targetIdx uint8) uint16 { return ActionBasePeekOther + uint16(targetIdx) }

// EncodeBlindSwap returns the action index for blind-swapping own card ownIdx with opponent oppIdx.
func EncodeBlindSwap(ownIdx, oppIdx uint8) uint16 {
	return ActionBaseBlindSwap + uint16(ownIdx)*6 + uint16(oppIdx)
}

// EncodeKingLook returns the action index for king-look on own ownIdx / opponent oppIdx.
func EncodeKingLook(ownIdx, oppIdx uint8) uint16 {
	return ActionBaseKingLook + uint16(ownIdx)*6 + uint16(oppIdx)
}

// EncodeSnapOwn returns the action index for snapping own card at targetIdx.
func EncodeSnapOwn(targetIdx uint8) uint16 { return ActionBaseSnapOwn + uint16(targetIdx) }

// EncodeSnapOpponent returns the action index for snapping opponent's card at targetIdx.
func EncodeSnapOpponent(targetIdx uint8) uint16 {
	return ActionBaseSnapOpponent + uint16(targetIdx)
}

// EncodeSnapOpponentMove returns the action for moving drawn card to own ownIdx to resolve a snap penalty.
func EncodeSnapOpponentMove(ownIdx, slotIdx uint8) uint16 {
	return ActionBaseSnapOpponentMove + uint16(ownIdx)*6 + uint16(slotIdx)
}

// ---------------------------------------------------------------------------
// Decode / predicate functions
// ---------------------------------------------------------------------------

// ActionIsReplace returns the target hand index if idx encodes a Replace action.
func ActionIsReplace(idx uint16) (targetIdx uint8, ok bool) {
	if idx >= ActionBaseReplace && idx < ActionBasePeekOwn {
		return uint8(idx - ActionBaseReplace), true
	}
	return 0, false
}

// ActionIsPeekOwn returns the target index if idx encodes a PeekOwn action.
func ActionIsPeekOwn(idx uint16) (targetIdx uint8, ok bool) {
	if idx >= ActionBasePeekOwn && idx < ActionBasePeekOther {
		return uint8(idx - ActionBasePeekOwn), true
	}
	return 0, false
}

// ActionIsPeekOther returns the target index if idx encodes a PeekOther action.
func ActionIsPeekOther(idx uint16) (targetIdx uint8, ok bool) {
	if idx >= ActionBasePeekOther && idx < ActionBaseBlindSwap {
		return uint8(idx - ActionBasePeekOther), true
	}
	return 0, false
}

// ActionIsBlindSwap returns own/opp indices if idx encodes a BlindSwap action.
func ActionIsBlindSwap(idx uint16) (ownIdx, oppIdx uint8, ok bool) {
	if idx >= ActionBaseBlindSwap && idx < ActionBaseKingLook {
		offset := idx - ActionBaseBlindSwap
		return uint8(offset / 6), uint8(offset % 6), true
	}
	return 0, 0, false
}

// ActionIsKingLook returns own/opp indices if idx encodes a KingLook action.
func ActionIsKingLook(idx uint16) (ownIdx, oppIdx uint8, ok bool) {
	if idx >= ActionBaseKingLook && idx < ActionKingSwapNo {
		offset := idx - ActionBaseKingLook
		return uint8(offset / 6), uint8(offset % 6), true
	}
	return 0, 0, false
}

// ActionIsKingSwap returns whether this is a KingSwap action and whether a swap is performed.
func ActionIsKingSwap(idx uint16) (performSwap bool, ok bool) {
	if idx == ActionKingSwapNo {
		return false, true
	}
	if idx == ActionKingSwapYes {
		return true, true
	}
	return false, false
}

// ActionIsSnapOwn returns the target index if idx encodes a SnapOwn action.
func ActionIsSnapOwn(idx uint16) (targetIdx uint8, ok bool) {
	if idx >= ActionBaseSnapOwn && idx < ActionBaseSnapOpponent {
		return uint8(idx - ActionBaseSnapOwn), true
	}
	return 0, false
}

// ActionIsSnapOpponent returns the target index if idx encodes a SnapOpponent action.
func ActionIsSnapOpponent(idx uint16) (targetIdx uint8, ok bool) {
	if idx >= ActionBaseSnapOpponent && idx < ActionBaseSnapOpponentMove {
		return uint8(idx - ActionBaseSnapOpponent), true
	}
	return 0, false
}

// ActionIsSnapOpponentMove returns own/slot indices if idx encodes a SnapOpponentMove action.
func ActionIsSnapOpponentMove(idx uint16) (ownIdx, slotIdx uint8, ok bool) {
	if idx >= ActionBaseSnapOpponentMove && idx < NumActions {
		offset := idx - ActionBaseSnapOpponentMove
		return uint8(offset / 6), uint8(offset % 6), true
	}
	return 0, 0, false
}

// ---------------------------------------------------------------------------
// LastActionInfo — public observation of the last game action.
// ---------------------------------------------------------------------------

// LastActionInfo encodes a fully observable summary of the most recent action.
type LastActionInfo struct {
	ActionIdx    uint16
	ActingPlayer uint8
	RevealedCard Card
	RevealedIdx  uint8
	RevealedOwner uint8
	SwapOwnIdx   uint8
	SwapOppIdx   uint8
	SnapSuccess  bool
	SnapPenalty  uint8
	DrawnFrom    uint8 // DrawnFromStockpile or DrawnFromDiscard
}

const (
	DrawnFromStockpile uint8 = 0
	DrawnFromDiscard   uint8 = 1
)

// ---------------------------------------------------------------------------
// N-Player action index constants (452 actions for up to 6 players)
// ---------------------------------------------------------------------------
// Used when NumPlayers > 2. The legacy 146-action encoding above is unchanged.
//
// Layout:
//   0   DrawStockpile
//   1   DrawDiscard
//   2   CallCambia
//   3   DiscardNoAbility
//   4   DiscardWithAbility
//   5–10    Replace(slot), 6 entries
//   11–16   PeekOwn(slot), 6 entries
//   17–46   PeekOther(slot*5 + oppIdx), 6 slots × 5 opponents = 30 entries
//   47–226  BlindSwap(own*30 + oppSlot*5 + oppIdx), 6 own × 6 opp-slots × 5 opp-idx = 180 entries
//   227–406 KingLook(own*30 + oppSlot*5 + oppIdx), 180 entries
//   407     KingSwapNo
//   408     KingSwapYes
//   409     PassSnap
//   410–415 SnapOwn(slot), 6 entries
//   416–445 SnapOpponent(slot*5 + oppIdx), 6 slots × 5 opp-idx = 30 entries
//   446–451 SnapOpponentMove(ownCardIdx), 6 entries
//   Total: 452

const (
	NPlayerActionDrawStockpile      uint16 = 0
	NPlayerActionDrawDiscard        uint16 = 1
	NPlayerActionCallCambia         uint16 = 2
	NPlayerActionDiscardNoAbility   uint16 = 3
	NPlayerActionDiscardWithAbility uint16 = 4

	NPlayerActionBaseReplace          uint16 = 5   // Replace(slot), 6 entries
	NPlayerActionBasePeekOwn          uint16 = 11  // PeekOwn(slot), 6 entries
	NPlayerActionBasePeekOther        uint16 = 17  // PeekOther(slot, oppIdx), 30 entries
	NPlayerActionBaseBlindSwap        uint16 = 47  // BlindSwap(own, oppSlot, oppIdx), 180 entries
	NPlayerActionBaseKingLook         uint16 = 227 // KingLook(own, oppSlot, oppIdx), 180 entries
	NPlayerActionKingSwapNo           uint16 = 407
	NPlayerActionKingSwapYes          uint16 = 408
	NPlayerActionPassSnap             uint16 = 409
	NPlayerActionBaseSnapOwn          uint16 = 410 // SnapOwn(slot), 6 entries
	NPlayerActionBaseSnapOpponent     uint16 = 416 // SnapOpponent(slot, oppIdx), 30 entries
	NPlayerActionBaseSnapOpponentMove uint16 = 446 // SnapOpponentMove(ownCardIdx), 6 entries

	NPlayerNumActions uint16 = 452
	MaxOpponents      uint8  = 5 // MaxPlayers - 1
)

// ---------------------------------------------------------------------------
// N-Player encode functions
// ---------------------------------------------------------------------------

func NPlayerEncodeReplace(slot uint8) uint16 { return NPlayerActionBaseReplace + uint16(slot) }
func NPlayerEncodePeekOwn(slot uint8) uint16 { return NPlayerActionBasePeekOwn + uint16(slot) }

// NPlayerEncodePeekOther encodes PeekOther(slot, oppIdx) where oppIdx is 0-based
// index into Opponents(acting).
func NPlayerEncodePeekOther(slot, oppIdx uint8) uint16 {
	return NPlayerActionBasePeekOther + uint16(slot)*5 + uint16(oppIdx)
}

// NPlayerEncodeBlindSwap encodes BlindSwap(ownSlot, oppSlot, oppIdx).
func NPlayerEncodeBlindSwap(ownSlot, oppSlot, oppIdx uint8) uint16 {
	return NPlayerActionBaseBlindSwap + uint16(ownSlot)*30 + uint16(oppSlot)*5 + uint16(oppIdx)
}

// NPlayerEncodeKingLook encodes KingLook(ownSlot, oppSlot, oppIdx).
func NPlayerEncodeKingLook(ownSlot, oppSlot, oppIdx uint8) uint16 {
	return NPlayerActionBaseKingLook + uint16(ownSlot)*30 + uint16(oppSlot)*5 + uint16(oppIdx)
}

func NPlayerEncodeSnapOwn(slot uint8) uint16 { return NPlayerActionBaseSnapOwn + uint16(slot) }

// NPlayerEncodeSnapOpponent encodes SnapOpponent(slot, oppIdx).
func NPlayerEncodeSnapOpponent(slot, oppIdx uint8) uint16 {
	return NPlayerActionBaseSnapOpponent + uint16(slot)*5 + uint16(oppIdx)
}

// NPlayerEncodeSnapOpponentMove encodes SnapOpponentMove(ownCardIdx).
func NPlayerEncodeSnapOpponentMove(ownIdx uint8) uint16 {
	return NPlayerActionBaseSnapOpponentMove + uint16(ownIdx)
}

// ---------------------------------------------------------------------------
// N-Player decode functions
// ---------------------------------------------------------------------------

func NPlayerDecodeReplace(idx uint16) (slot uint8, ok bool) {
	if idx >= NPlayerActionBaseReplace && idx < NPlayerActionBasePeekOwn {
		return uint8(idx - NPlayerActionBaseReplace), true
	}
	return 0, false
}

func NPlayerDecodePeekOwn(idx uint16) (slot uint8, ok bool) {
	if idx >= NPlayerActionBasePeekOwn && idx < NPlayerActionBasePeekOther {
		return uint8(idx - NPlayerActionBasePeekOwn), true
	}
	return 0, false
}

func NPlayerDecodePeekOther(idx uint16) (slot, oppIdx uint8, ok bool) {
	if idx >= NPlayerActionBasePeekOther && idx < NPlayerActionBaseBlindSwap {
		offset := idx - NPlayerActionBasePeekOther
		return uint8(offset / 5), uint8(offset % 5), true
	}
	return 0, 0, false
}

func NPlayerDecodeBlindSwap(idx uint16) (ownSlot, oppSlot, oppIdx uint8, ok bool) {
	if idx >= NPlayerActionBaseBlindSwap && idx < NPlayerActionBaseKingLook {
		offset := idx - NPlayerActionBaseBlindSwap
		return uint8(offset / 30), uint8((offset % 30) / 5), uint8(offset % 5), true
	}
	return 0, 0, 0, false
}

func NPlayerDecodeKingLook(idx uint16) (ownSlot, oppSlot, oppIdx uint8, ok bool) {
	if idx >= NPlayerActionBaseKingLook && idx < NPlayerActionKingSwapNo {
		offset := idx - NPlayerActionBaseKingLook
		return uint8(offset / 30), uint8((offset % 30) / 5), uint8(offset % 5), true
	}
	return 0, 0, 0, false
}

func NPlayerDecodeSnapOwn(idx uint16) (slot uint8, ok bool) {
	if idx >= NPlayerActionBaseSnapOwn && idx < NPlayerActionBaseSnapOpponent {
		return uint8(idx - NPlayerActionBaseSnapOwn), true
	}
	return 0, false
}

func NPlayerDecodeSnapOpponent(idx uint16) (slot, oppIdx uint8, ok bool) {
	if idx >= NPlayerActionBaseSnapOpponent && idx < NPlayerActionBaseSnapOpponentMove {
		offset := idx - NPlayerActionBaseSnapOpponent
		return uint8(offset / 5), uint8(offset % 5), true
	}
	return 0, 0, false
}

func NPlayerDecodeSnapOpponentMove(idx uint16) (ownIdx uint8, ok bool) {
	if idx >= NPlayerActionBaseSnapOpponentMove && idx < NPlayerNumActions {
		return uint8(idx - NPlayerActionBaseSnapOpponentMove), true
	}
	return 0, false
}
