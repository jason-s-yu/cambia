// Package agent implements belief tracking and tensor encoding
// for CFR training agents.
package agent

import engine "github.com/jason-s-yu/cambia/engine"

// KnownCardInfo holds belief information for one of the agent's own hand slots.
type KnownCardInfo struct {
	Bucket       CardBucket
	LastSeenTurn uint16
	Card         engine.Card // actual card if known, EmptyCard if unknown
}

// BeliefValue encodes an opponent hand slot belief in a single byte.
// Values 0-9 = CardBucket (BucketZero through BucketUnknown).
// Values 10-13 = DecayCategory offset by 10 (DecayLikelyLow=10 .. DecayUnknown=13).
type BeliefValue uint8

const (
	BeliefOffsetDecay uint8 = 10 // DecayCategory values start here
)

// BucketBelief constructs a BeliefValue from a CardBucket.
func BucketBelief(b CardBucket) BeliefValue { return BeliefValue(b) }

// DecayBelief constructs a BeliefValue from a DecayCategory.
func DecayBelief(d DecayCategory) BeliefValue { return BeliefValue(uint8(d) + BeliefOffsetDecay) }

// IsBucket returns true if this belief encodes a CardBucket (not a decay category).
func (b BeliefValue) IsBucket() bool { return uint8(b) < BeliefOffsetDecay }

// IsDecay returns true if this belief encodes a DecayCategory.
func (b BeliefValue) IsDecay() bool { return uint8(b) >= BeliefOffsetDecay }

// Bucket returns the CardBucket for this belief (only valid when IsBucket() is true).
func (b BeliefValue) Bucket() CardBucket { return CardBucket(b) }

// Decay returns the DecayCategory for this belief (only valid when IsDecay() is true).
func (b BeliefValue) Decay() DecayCategory { return DecayCategory(uint8(b) - BeliefOffsetDecay) }

// AgentState holds the complete belief state for one player in a two-player game.
// It is a flat value type — no pointers, no maps, no slices — so it can be
// copied with =, saved on the stack, and used in zero-allocation CFR traversal.
type AgentState struct {
	PlayerID       uint8
	OpponentID     uint8
	MemoryLevel    uint8 // 0, 1, or 2
	TimeDecayTurns uint8

	// Own hand beliefs — indexed 0..OwnHandLen-1
	OwnHand    [engine.MaxHandSize]KnownCardInfo
	OwnHandLen uint8

	// Opponent beliefs — indexed 0..OppHandLen-1
	OppBelief      [engine.MaxHandSize]BeliefValue
	OppLastSeen    [engine.MaxHandSize]uint16
	OppHasLastSeen [engine.MaxHandSize]bool
	OppHandLen     uint8

	// Public knowledge
	DiscardTopBucket CardBucket
	StockEstimate    StockpileEstimate
	Phase            GamePhase
	CambiaState      CambiaState

	// Internal
	CurrentTurn uint16
}

// NewAgentState creates a zero-initialized AgentState with the given player IDs
// and memory configuration.
func NewAgentState(playerID, opponentID, memoryLevel, timeDecayTurns uint8) AgentState {
	return AgentState{
		PlayerID:       playerID,
		OpponentID:     opponentID,
		MemoryLevel:    memoryLevel,
		TimeDecayTurns: timeDecayTurns,
	}
}

// Initialize sets up the agent's belief state at game start (after Deal).
// The agent peeks indices 0 and 1 in its own hand (per the initial peek rule)
// and marks all other slots as unknown.
func (a *AgentState) Initialize(g *engine.GameState) {
	a.OwnHandLen = g.Players[a.PlayerID].HandLen
	a.OppHandLen = g.Players[a.OpponentID].HandLen

	// Own hand: peek indices from InitialPeek get actual bucket; rest are unknown.
	peekSet := [engine.MaxHandSize]bool{}
	peekIdx0 := g.Players[a.PlayerID].InitialPeek[0]
	peekIdx1 := g.Players[a.PlayerID].InitialPeek[1]
	if peekIdx0 < a.OwnHandLen {
		peekSet[peekIdx0] = true
	}
	if peekIdx1 < a.OwnHandLen {
		peekSet[peekIdx1] = true
	}

	for i := uint8(0); i < a.OwnHandLen; i++ {
		if peekSet[i] {
			card := g.Players[a.PlayerID].Hand[i]
			a.OwnHand[i] = KnownCardInfo{
				Bucket:       CardToBucket(card),
				LastSeenTurn: 0,
				Card:         card,
			}
		} else {
			a.OwnHand[i] = KnownCardInfo{
				Bucket:       BucketUnknown,
				LastSeenTurn: 0,
				Card:         engine.EmptyCard,
			}
		}
	}

	// Opponent hand: all slots unknown.
	for i := uint8(0); i < a.OppHandLen; i++ {
		a.OppBelief[i] = BucketBelief(BucketUnknown)
		a.OppLastSeen[i] = 0
		a.OppHasLastSeen[i] = false
	}

	// Public knowledge
	discardTop := g.DiscardTop()
	if discardTop == engine.EmptyCard {
		a.DiscardTopBucket = BucketUnknown
	} else {
		a.DiscardTopBucket = CardToBucket(discardTop)
	}
	a.StockEstimate = StockEstimateFromSize(g.StockLen)
	a.Phase = GamePhaseFromState(g.StockLen, g.IsCambiaCalled(), g.IsTerminal())
	a.CambiaState = CambiaNone
	a.CurrentTurn = 0
}

// Update refreshes the agent's beliefs based on what happened since the last action.
// It must be called ONCE after each action is applied to the game state, using
// g.LastAction to determine what happened.
func (a *AgentState) Update(g *engine.GameState) {
	// Step 1: Update public knowledge.
	discardTop := g.DiscardTop()
	if discardTop == engine.EmptyCard {
		a.DiscardTopBucket = BucketUnknown
	} else {
		a.DiscardTopBucket = CardToBucket(discardTop)
	}
	a.StockEstimate = StockEstimateFromSize(g.StockLen)
	a.Phase = GamePhaseFromState(g.StockLen, g.IsCambiaCalled(), g.IsTerminal())
	if g.IsCambiaCalled() && a.CambiaState == CambiaNone {
		if g.CambiaCaller == int8(a.PlayerID) {
			a.CambiaState = CambiaSelf
		} else {
			a.CambiaState = CambiaOpponent
		}
	}
	a.CurrentTurn = g.TurnNumber

	// Step 2: Process the last action.
	act := g.LastAction.ActionIdx
	actingPlayer := g.LastAction.ActingPlayer
	isSelf := actingPlayer == a.PlayerID

	switch {
	case act == engine.ActionDrawStockpile || act == engine.ActionDrawDiscard:
		// No belief changes needed for draw alone.

	case act == engine.ActionDiscardNoAbility || act == engine.ActionDiscardWithAbility:
		// Discarded card is now visible on discard pile; no hand index changes here.

	case act == engine.ActionCallCambia:
		// Cambia is handled via the IsCambiaCalled() flag above.

	default:
		if targetIdx, ok := engine.ActionIsReplace(act); ok {
			a.processReplace(g, isSelf, targetIdx)

		} else if targetIdx, ok := engine.ActionIsPeekOwn(act); ok {
			a.processPeekOwn(g, isSelf, targetIdx)

		} else if targetIdx, ok := engine.ActionIsPeekOther(act); ok {
			a.processPeekOther(g, isSelf, targetIdx)

		} else if ownIdx, oppIdx, ok := engine.ActionIsBlindSwap(act); ok {
			a.processBlindSwap(g, isSelf, ownIdx, oppIdx)

		} else if ownIdx, oppIdx, ok := engine.ActionIsKingLook(act); ok {
			a.processKingLook(g, isSelf, ownIdx, oppIdx)

		} else if _, ok := engine.ActionIsKingSwap(act); ok {
			if act == engine.ActionKingSwapYes {
				a.processKingSwapYes(g, isSelf)
			}
			// ActionKingSwapNo: no card movements.

		} else if act == engine.ActionPassSnap {
			// No belief changes.

		} else if targetIdx, ok := engine.ActionIsSnapOwn(act); ok {
			a.processSnapOwn(g, isSelf, targetIdx)

		} else if targetIdx, ok := engine.ActionIsSnapOpponent(act); ok {
			a.processSnapOpponent(g, isSelf, targetIdx)

		} else if ownIdx, slotIdx, ok := engine.ActionIsSnapOpponentMove(act); ok {
			a.processSnapOpponentMove(g, isSelf, ownIdx, slotIdx)
		}
	}

	// Step 3: Reconcile hand lengths against actual game state.
	// This handles edge cases where snap penalty draws fewer cards than
	// PenaltyDrawCount (e.g., stockpile empty or MaxHandSize cap).
	a.reconcileHandLengths(g)

	// Step 4: Apply time decay (memory level 2 only).
	a.applyTimeDecay()
}

// processReplace handles ActionBaseReplace+targetIdx.
func (a *AgentState) processReplace(g *engine.GameState, isSelf bool, targetIdx uint8) {
	if isSelf {
		// We replaced our card at targetIdx with the drawn card; we now know that card.
		card := g.Players[a.PlayerID].Hand[targetIdx]
		a.OwnHand[targetIdx] = KnownCardInfo{
			Bucket:       CardToBucket(card),
			LastSeenTurn: a.CurrentTurn,
			Card:         card,
		}
	} else {
		// Opponent replaced one of their cards — event decay on that slot.
		a.triggerEventDecay(targetIdx)
	}
}

// processPeekOwn handles EncodePeekOwn(targetIdx).
func (a *AgentState) processPeekOwn(g *engine.GameState, isSelf bool, targetIdx uint8) {
	if isSelf {
		// We peeked our own card.
		card := g.LastAction.RevealedCard
		a.OwnHand[targetIdx] = KnownCardInfo{
			Bucket:       CardToBucket(card),
			LastSeenTurn: a.CurrentTurn,
			Card:         card,
		}
	}
	// If opponent peeked their own card, we learn nothing.
}

// processPeekOther handles EncodePeekOther(targetIdx).
func (a *AgentState) processPeekOther(g *engine.GameState, isSelf bool, targetIdx uint8) {
	if isSelf {
		// We peeked the opponent's card at targetIdx.
		card := g.LastAction.RevealedCard
		a.OppBelief[targetIdx] = BucketBelief(CardToBucket(card))
		a.OppLastSeen[targetIdx] = a.CurrentTurn
		a.OppHasLastSeen[targetIdx] = true
	}
	// If opponent peeked our card, we learn nothing.
}

// processBlindSwap handles EncodeBlindSwap(ownIdx, oppIdx).
// In the encoding, ownIdx/oppIdx are relative to the acting player.
func (a *AgentState) processBlindSwap(g *engine.GameState, isSelf bool, ownIdx, oppIdx uint8) {
	if isSelf {
		// We swapped our card at ownIdx with opponent's card at oppIdx.
		// Our ownIdx now has the opponent's card (unknown to us).
		// Opponent's oppIdx now has our old card (event decay since it moved).
		a.OwnHand[ownIdx] = KnownCardInfo{
			Bucket:       BucketUnknown,
			LastSeenTurn: a.CurrentTurn,
			Card:         engine.EmptyCard,
		}
		a.triggerEventDecay(oppIdx)
	} else {
		// Opponent swapped: their ownIdx (their hand) <-> their oppIdx (our hand).
		// Our card at oppIdx is now unknown (we received opponent's old card).
		// Opponent's ownIdx slot gets event decay.
		a.OwnHand[oppIdx] = KnownCardInfo{
			Bucket:       BucketUnknown,
			LastSeenTurn: a.CurrentTurn,
			Card:         engine.EmptyCard,
		}
		a.triggerEventDecay(ownIdx)
	}
}

// processKingLook handles EncodeKingLook(ownIdx, oppIdx).
func (a *AgentState) processKingLook(g *engine.GameState, isSelf bool, ownIdx, oppIdx uint8) {
	if isSelf {
		// We looked at our card at ownIdx and opponent's card at oppIdx.
		ownCard := g.LastAction.RevealedCard
		a.OwnHand[ownIdx] = KnownCardInfo{
			Bucket:       CardToBucket(ownCard),
			LastSeenTurn: a.CurrentTurn,
			Card:         ownCard,
		}
		// We also see the opponent's card during king look.
		oppCard := g.Players[a.OpponentID].Hand[oppIdx]
		a.OppBelief[oppIdx] = BucketBelief(CardToBucket(oppCard))
		a.OppLastSeen[oppIdx] = a.CurrentTurn
		a.OppHasLastSeen[oppIdx] = true
	}
	// If opponent is king-looking, we learn nothing.
}

// processKingSwapYes handles ActionKingSwapYes.
func (a *AgentState) processKingSwapYes(g *engine.GameState, isSelf bool) {
	ownIdx := g.LastAction.SwapOwnIdx
	oppIdx := g.LastAction.SwapOppIdx

	if isSelf {
		// We decided to swap: our card at ownIdx went to opponent, opponent's oppIdx came to us.
		// We no longer know what's at our ownIdx (we got the opponent's old card — but wait,
		// we DID look at it during king look). However, the belief update tracks the post-swap
		// state: our ownIdx now contains the opponent's card, which we SAW during king look.
		// But the king look already updated beliefs, and now we performed the swap.
		// After swap: our ownIdx has what was opponent's oppIdx (which we know).
		// However for belief consistency, we mark it unknown after the swap since it's now
		// the card that was just moved.
		a.OwnHand[ownIdx] = KnownCardInfo{
			Bucket:       BucketUnknown,
			LastSeenTurn: a.CurrentTurn,
			Card:         engine.EmptyCard,
		}
		// Opponent's oppIdx now has our old card — trigger event decay.
		a.triggerEventDecay(oppIdx)
	} else {
		// Opponent swapped: opponent's ownIdx <-> our oppIdx.
		// Our card at oppIdx is now unknown.
		// Opponent's ownIdx slot gets event decay.
		ourIdx := oppIdx  // our hand index that got taken
		theirIdx := ownIdx // their hand index that gave us a card
		a.OwnHand[ourIdx] = KnownCardInfo{
			Bucket:       BucketUnknown,
			LastSeenTurn: a.CurrentTurn,
			Card:         engine.EmptyCard,
		}
		a.triggerEventDecay(theirIdx)
	}
}

// processSnapOwn handles EncodeSnapOwn(targetIdx) from snapper's perspective.
// Snapper is g.LastAction.ActingPlayer.
func (a *AgentState) processSnapOwn(g *engine.GameState, isSelf bool, targetIdx uint8) {
	snapSuccess := g.LastAction.SnapSuccess
	snapPenalty := g.LastAction.SnapPenalty

	if snapSuccess {
		if isSelf {
			// We successfully snapped our own card at targetIdx.
			a.removeOwnCard(targetIdx)
		} else {
			// Opponent successfully snapped their own card at targetIdx.
			a.removeOppCard(targetIdx)
		}
	} else {
		// Snap failed — penalty cards are drawn.
		penaltyCount := snapPenalty
		if isSelf {
			for i := uint8(0); i < penaltyCount; i++ {
				a.addOwnUnknown(a.CurrentTurn)
			}
		} else {
			for i := uint8(0); i < penaltyCount; i++ {
				a.addOppUnknown()
			}
		}
	}
}

// processSnapOpponent handles EncodeSnapOpponent(oppIdx).
// Snapper is g.LastAction.ActingPlayer.
func (a *AgentState) processSnapOpponent(g *engine.GameState, isSelf bool, oppIdx uint8) {
	snapSuccess := g.LastAction.SnapSuccess
	snapPenalty := g.LastAction.SnapPenalty

	if snapSuccess {
		if isSelf {
			// We snapped the opponent's card at oppIdx.
			a.removeOppCard(oppIdx)
		} else {
			// Opponent snapped our card at oppIdx.
			a.removeOwnCard(oppIdx)
		}
	} else {
		// Snap failed — penalty cards for the snapper.
		penaltyCount := snapPenalty
		if isSelf {
			for i := uint8(0); i < penaltyCount; i++ {
				a.addOwnUnknown(a.CurrentTurn)
			}
		} else {
			for i := uint8(0); i < penaltyCount; i++ {
				a.addOppUnknown()
			}
		}
	}
}

// processSnapOpponentMove handles EncodeSnapOpponentMove(ownIdx, slotIdx).
// Mover is g.LastAction.ActingPlayer.
func (a *AgentState) processSnapOpponentMove(g *engine.GameState, isSelf bool, ownIdx, slotIdx uint8) {
	if isSelf {
		// We moved our card at ownIdx to opponent's hand at slotIdx.
		a.removeOwnCard(ownIdx)
		// Insert unknown at slotIdx in opponent's hand (or append).
		a.insertOppUnknown(slotIdx)
	} else {
		// Opponent moved their card at ownIdx to our hand at slotIdx.
		a.removeOppCard(ownIdx)
		// Insert unknown at slotIdx in our hand.
		a.insertOwnUnknown(slotIdx, a.CurrentTurn)
	}
}

// ---------------------------------------------------------------------------
// Hand manipulation helpers
// ---------------------------------------------------------------------------

// removeOwnCard removes the card at idx from OwnHand, shifting remaining cards left.
func (a *AgentState) removeOwnCard(idx uint8) {
	if idx >= a.OwnHandLen {
		return
	}
	for i := idx; i < a.OwnHandLen-1; i++ {
		a.OwnHand[i] = a.OwnHand[i+1]
	}
	a.OwnHandLen--
	a.OwnHand[a.OwnHandLen] = KnownCardInfo{} // clear vacated slot
}

// removeOppCard removes the card at idx from OppBelief, shifting remaining slots left.
func (a *AgentState) removeOppCard(idx uint8) {
	if idx >= a.OppHandLen {
		return
	}
	for i := idx; i < a.OppHandLen-1; i++ {
		a.OppBelief[i] = a.OppBelief[i+1]
		a.OppLastSeen[i] = a.OppLastSeen[i+1]
		a.OppHasLastSeen[i] = a.OppHasLastSeen[i+1]
	}
	a.OppHandLen--
	// Clear vacated slot.
	a.OppBelief[a.OppHandLen] = 0
	a.OppLastSeen[a.OppHandLen] = 0
	a.OppHasLastSeen[a.OppHandLen] = false
}

// addOwnUnknown appends an unknown card to the end of OwnHand.
func (a *AgentState) addOwnUnknown(turn uint16) {
	if a.OwnHandLen >= engine.MaxHandSize {
		return
	}
	a.OwnHand[a.OwnHandLen] = KnownCardInfo{
		Bucket:       BucketUnknown,
		LastSeenTurn: turn,
		Card:         engine.EmptyCard,
	}
	a.OwnHandLen++
}

// addOppUnknown appends an unknown belief to the end of OppBelief.
func (a *AgentState) addOppUnknown() {
	if a.OppHandLen >= engine.MaxHandSize {
		return
	}
	a.OppBelief[a.OppHandLen] = BucketBelief(BucketUnknown)
	a.OppLastSeen[a.OppHandLen] = 0
	a.OppHasLastSeen[a.OppHandLen] = false
	a.OppHandLen++
}

// insertOwnUnknown inserts an unknown card into OwnHand at slotIdx, shifting right.
func (a *AgentState) insertOwnUnknown(slotIdx uint8, turn uint16) {
	if a.OwnHandLen >= engine.MaxHandSize {
		return
	}
	// Clamp slotIdx.
	if slotIdx > a.OwnHandLen {
		slotIdx = a.OwnHandLen
	}
	// Shift right to make room.
	for i := a.OwnHandLen; i > slotIdx; i-- {
		a.OwnHand[i] = a.OwnHand[i-1]
	}
	a.OwnHand[slotIdx] = KnownCardInfo{
		Bucket:       BucketUnknown,
		LastSeenTurn: turn,
		Card:         engine.EmptyCard,
	}
	a.OwnHandLen++
}

// insertOppUnknown inserts an unknown belief into OppBelief at slotIdx, shifting right.
func (a *AgentState) insertOppUnknown(slotIdx uint8) {
	if a.OppHandLen >= engine.MaxHandSize {
		return
	}
	// Clamp slotIdx.
	if slotIdx > a.OppHandLen {
		slotIdx = a.OppHandLen
	}
	// Shift right to make room.
	for i := a.OppHandLen; i > slotIdx; i-- {
		a.OppBelief[i] = a.OppBelief[i-1]
		a.OppLastSeen[i] = a.OppLastSeen[i-1]
		a.OppHasLastSeen[i] = a.OppHasLastSeen[i-1]
	}
	a.OppBelief[slotIdx] = BucketBelief(BucketUnknown)
	a.OppLastSeen[slotIdx] = 0
	a.OppHasLastSeen[slotIdx] = false
	a.OppHandLen++
}

// reconcileHandLengths ensures agent hand lengths match the actual game state.
// This handles edge cases where snap penalty draws fewer cards than configured
// (stockpile exhausted, MaxHandSize cap, etc.).
func (a *AgentState) reconcileHandLengths(g *engine.GameState) {
	actualOwn := g.Players[a.PlayerID].HandLen
	actualOpp := g.Players[a.OpponentID].HandLen

	// Reconcile own hand.
	for a.OwnHandLen < actualOwn {
		a.addOwnUnknown(a.CurrentTurn)
	}
	for a.OwnHandLen > actualOwn && a.OwnHandLen > 0 {
		a.OwnHandLen--
		a.OwnHand[a.OwnHandLen] = KnownCardInfo{}
	}

	// Reconcile opponent hand.
	for a.OppHandLen < actualOpp {
		a.addOppUnknown()
	}
	for a.OppHandLen > actualOpp && a.OppHandLen > 0 {
		a.OppHandLen--
		a.OppBelief[a.OppHandLen] = 0
		a.OppLastSeen[a.OppHandLen] = 0
		a.OppHasLastSeen[a.OppHandLen] = false
	}
}

// triggerEventDecay decays the opponent belief at oppIdx to a DecayCategory
// (memory levels 1 and 2 only). Does nothing for memory level 0.
func (a *AgentState) triggerEventDecay(oppIdx uint8) {
	if a.MemoryLevel == 0 {
		return
	}
	if oppIdx >= a.OppHandLen {
		return
	}
	b := a.OppBelief[oppIdx]
	if b.IsBucket() && b.Bucket() != BucketUnknown {
		decayed := BucketToDecay(b.Bucket())
		a.OppBelief[oppIdx] = DecayBelief(decayed)
		a.OppHasLastSeen[oppIdx] = false
	}
}

// applyTimeDecay applies time-based decay to all opponent beliefs that have aged
// past TimeDecayTurns. Only active for memory level 2.
func (a *AgentState) applyTimeDecay() {
	if a.MemoryLevel != 2 {
		return
	}
	for i := uint8(0); i < a.OppHandLen; i++ {
		if !a.OppHasLastSeen[i] {
			continue
		}
		if a.OppBelief[i].IsBucket() && a.OppBelief[i].Bucket() != BucketUnknown {
			if a.CurrentTurn-a.OppLastSeen[i] >= uint16(a.TimeDecayTurns) {
				decayed := BucketToDecay(a.OppBelief[i].Bucket())
				a.OppBelief[i] = DecayBelief(decayed)
				a.OppHasLastSeen[i] = false
			}
		}
	}
}

// Clone returns a copy of this AgentState. Since AgentState is a flat struct
// with no pointers, this is simply a value copy.
func (a *AgentState) Clone() AgentState { return *a }

// InfosetKey encodes the belief state into a fixed-size 16-byte array.
// Layout:
//
//	[0..5]   own hand buckets (OwnHand[0..5].Bucket), padded with BucketUnknown
//	[6..11]  opponent beliefs (OppBelief[0..5] raw byte), padded with 0xFF
//	[12]     OppHandLen
//	[13]     DiscardTopBucket
//	[14]     StockEstimate | (Phase << 4)
//	[15]     CambiaState
func (a *AgentState) InfosetKey() [16]uint8 {
	var key [16]uint8
	for i := uint8(0); i < engine.MaxHandSize; i++ {
		if i < a.OwnHandLen {
			key[i] = uint8(a.OwnHand[i].Bucket)
		} else {
			key[i] = uint8(BucketUnknown)
		}
	}
	for i := uint8(0); i < engine.MaxHandSize; i++ {
		if i < a.OppHandLen {
			key[6+i] = uint8(a.OppBelief[i])
		} else {
			key[6+i] = 0xFF // empty slot marker
		}
	}
	key[12] = a.OppHandLen
	key[13] = uint8(a.DiscardTopBucket)
	key[14] = uint8(a.StockEstimate) | (uint8(a.Phase) << 4)
	key[15] = uint8(a.CambiaState)
	return key
}
