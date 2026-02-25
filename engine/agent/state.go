// Package agent implements belief tracking and tensor encoding
// for CFR training agents.
package agent

import (
	"math"
	"math/rand/v2"

	engine "github.com/jason-s-yu/cambia/engine"
)

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

	// EP-PBS fields (2P epistemic positional belief state).
	// SlotTags tracks who knows each physical card slot (own slots 0-5, opp slots 6-11).
	// SlotBuckets holds the known bucket for slots with TagPrivOwn or TagPub; zero otherwise.
	// OwnActiveMask tracks slot indices we privately know (TagPrivOwn); saliency-evicted.
	// OppActiveMask tracks slot indices opponent privately knows (TagPrivOpp); FIFO-evicted.
	SlotTags         [MaxSlots]EpistemicTag
	SlotBuckets      [MaxSlots]CardBucket
	OwnActiveMask    [MaxActiveMask]uint8
	OwnActiveMaskLen uint8
	OppActiveMask    [MaxActiveMask]uint8
	OppActiveMaskLen uint8

	// N-Player fields (used when NumPlayers > 2).
	// NumPlayers is the total number of players in the game.
	// OpponentIDs holds the player IDs of opponents (up to 5).
	// KnowledgeMask[slot][playerID] is true when playerID has observed the card at slot.
	// Slot indices are: playerIndex * MaxHandSize + cardSlot.
	// NPlayerSlotBuckets holds the bucket for a slot when this agent knows it.
	// NPlayerSlotKnown indicates whether THIS agent knows the card at each slot.
	NumPlayers         uint8
	OpponentIDs        [5]uint8
	NumOpponents       uint8
	KnowledgeMask      [MaxTotalSlots][MaxKnowledgePlayers]bool
	NPlayerSlotBuckets [MaxTotalSlots]CardBucket
	NPlayerSlotKnown   [MaxTotalSlots]bool

	// Memory archetype controls per-turn decay/eviction behavior.
	// MemoryPerfect (default): no decay, full retention.
	// MemoryDecaying: probabilistic Bayesian diffusion per turn.
	// MemoryHumanLike: saliency eviction capped at MemoryCapacity.
	MemoryArchetype   MemoryArchetype
	MemoryDecayLambda float32 // Decay rate λ for MemoryDecaying (default 0.1)
	MemoryCapacity    uint8   // Max active mask size for MemoryHumanLike (default 3)
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

	// Initialize EP-PBS slot tags.
	a.OwnActiveMaskLen = 0
	a.OppActiveMaskLen = 0
	for i := uint8(0); i < engine.MaxHandSize; i++ {
		if i < a.OwnHandLen && peekSet[i] {
			a.SlotTags[i] = TagPrivOwn
			a.SlotBuckets[i] = a.OwnHand[i].Bucket
			a.appendOwnActive(i)
		} else {
			a.SlotTags[i] = TagUnk
			a.SlotBuckets[i] = 0
		}
		// All opp slots start unknown.
		a.SlotTags[OppSlotsStart+i] = TagUnk
		a.SlotBuckets[OppSlotsStart+i] = 0
	}
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
		// EP-PBS: we know the new card at own slot targetIdx.
		a.eppbsForceOwnSlotKnown(uint8(targetIdx), CardToBucket(card))
	} else {
		// Opponent replaced one of their cards — event decay on that slot.
		a.triggerEventDecay(targetIdx)
		// EP-PBS: opp knows their new card at opp slot; we don't.
		a.eppbsForceOppSlotPrivOpp(OppSlotsStart + uint8(targetIdx))
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
		// EP-PBS: we now privately know our slot targetIdx.
		a.setOwnSlotKnown(uint8(targetIdx), CardToBucket(card))
	} else {
		// Opponent peeked their own card at targetIdx → EP-PBS: opp privately knows opp slot.
		a.setOppSlotPrivOpp(OppSlotsStart + uint8(targetIdx))
	}
}

// processPeekOther handles EncodePeekOther(targetIdx).
func (a *AgentState) processPeekOther(g *engine.GameState, isSelf bool, targetIdx uint8) {
	if isSelf {
		// We peeked the opponent's card at targetIdx.
		card := g.LastAction.RevealedCard
		a.OppBelief[targetIdx] = BucketBelief(CardToBucket(card))
		a.OppLastSeen[targetIdx] = a.CurrentTurn
		a.OppHasLastSeen[targetIdx] = true
		// EP-PBS: we now privately know opp slot targetIdx.
		a.setOwnSlotKnown(OppSlotsStart+uint8(targetIdx), CardToBucket(card))
	} else {
		// Opponent peeked our card at targetIdx → EP-PBS: opp privately knows our slot.
		a.setOppSlotPrivOpp(uint8(targetIdx))
	}
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
		// EP-PBS: both physical positions now unknown to us.
		a.eppbsSetSlotUnk(ownIdx)
		a.eppbsSetSlotUnk(OppSlotsStart + oppIdx)
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
		// EP-PBS: both physical positions unknown.
		a.eppbsSetSlotUnk(oppIdx)
		a.eppbsSetSlotUnk(OppSlotsStart + ownIdx)
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
		// EP-PBS: we privately know both slots we looked at.
		a.setOwnSlotKnown(ownIdx, CardToBucket(ownCard))
		a.setOwnSlotKnown(OppSlotsStart+oppIdx, CardToBucket(oppCard))
	} else {
		// Opponent is king-looking: they see their ownIdx and our oppIdx.
		// EP-PBS: opp privately knows both slots.
		a.setOppSlotPrivOpp(OppSlotsStart + ownIdx)
		a.setOppSlotPrivOpp(oppIdx)
	}
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
		// EP-PBS: swap the epistemic state of both slots (we know both from king look).
		a.eppbsSwapSlots(ownIdx, OppSlotsStart+oppIdx)
	} else {
		// Opponent swapped: opponent's ownIdx <-> our oppIdx.
		// Our card at oppIdx is now unknown.
		// Opponent's ownIdx slot gets event decay.
		ourIdx := oppIdx   // our hand index that got taken
		theirIdx := ownIdx // their hand index that gave us a card
		a.OwnHand[ourIdx] = KnownCardInfo{
			Bucket:       BucketUnknown,
			LastSeenTurn: a.CurrentTurn,
			Card:         engine.EmptyCard,
		}
		a.triggerEventDecay(theirIdx)
		// EP-PBS: swap epistemic state (opp knows both from their king look).
		a.eppbsSwapSlots(ourIdx, OppSlotsStart+theirIdx)
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
	// EP-PBS: remove slot idx from any active masks, shift own slots left.
	a.removeOwnActive(idx)
	a.removeOppActive(idx)
	for i := idx; i < a.OwnHandLen-1; i++ {
		a.SlotTags[i] = a.SlotTags[i+1]
		a.SlotBuckets[i] = a.SlotBuckets[i+1]
	}
	a.SlotTags[a.OwnHandLen-1] = TagUnk
	a.SlotBuckets[a.OwnHandLen-1] = 0
	// Decrement active mask entries > idx (own slots only, < OppSlotsStart).
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		if a.OwnActiveMask[i] > idx && a.OwnActiveMask[i] < OppSlotsStart {
			a.OwnActiveMask[i]--
		}
	}
	for i := uint8(0); i < a.OppActiveMaskLen; i++ {
		if a.OppActiveMask[i] > idx && a.OppActiveMask[i] < OppSlotsStart {
			a.OppActiveMask[i]--
		}
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
	// EP-PBS: remove global slot OppSlotsStart+idx from masks, shift opp slots left.
	globalSlot := OppSlotsStart + idx
	a.removeOwnActive(globalSlot)
	a.removeOppActive(globalSlot)
	for i := globalSlot; i < OppSlotsStart+a.OppHandLen-1; i++ {
		a.SlotTags[i] = a.SlotTags[i+1]
		a.SlotBuckets[i] = a.SlotBuckets[i+1]
	}
	a.SlotTags[OppSlotsStart+a.OppHandLen-1] = TagUnk
	a.SlotBuckets[OppSlotsStart+a.OppHandLen-1] = 0
	// Decrement active mask entries > globalSlot (opp slots only).
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		if a.OwnActiveMask[i] > globalSlot {
			a.OwnActiveMask[i]--
		}
	}
	for i := uint8(0); i < a.OppActiveMaskLen; i++ {
		if a.OppActiveMask[i] > globalSlot {
			a.OppActiveMask[i]--
		}
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
	// EP-PBS: new slot is TagUnk.
	a.SlotTags[a.OwnHandLen] = TagUnk
	a.SlotBuckets[a.OwnHandLen] = 0
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
	// EP-PBS: new slot is TagUnk.
	a.SlotTags[OppSlotsStart+a.OppHandLen] = TagUnk
	a.SlotBuckets[OppSlotsStart+a.OppHandLen] = 0
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
	// EP-PBS: shift own slots right, insert TagUnk at slotIdx.
	for i := a.OwnHandLen; i > slotIdx; i-- {
		a.SlotTags[i] = a.SlotTags[i-1]
		a.SlotBuckets[i] = a.SlotBuckets[i-1]
	}
	a.SlotTags[slotIdx] = TagUnk
	a.SlotBuckets[slotIdx] = 0
	// Increment active mask entries >= slotIdx (own slots only).
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		if a.OwnActiveMask[i] >= slotIdx && a.OwnActiveMask[i] < OppSlotsStart {
			a.OwnActiveMask[i]++
		}
	}
	for i := uint8(0); i < a.OppActiveMaskLen; i++ {
		if a.OppActiveMask[i] >= slotIdx && a.OppActiveMask[i] < OppSlotsStart {
			a.OppActiveMask[i]++
		}
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
	// EP-PBS: shift opp slots right, insert TagUnk at global slot position.
	globalSlot := OppSlotsStart + slotIdx
	for i := OppSlotsStart + a.OppHandLen; i > globalSlot; i-- {
		a.SlotTags[i] = a.SlotTags[i-1]
		a.SlotBuckets[i] = a.SlotBuckets[i-1]
	}
	a.SlotTags[globalSlot] = TagUnk
	a.SlotBuckets[globalSlot] = 0
	// Increment active mask entries >= globalSlot.
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		if a.OwnActiveMask[i] >= globalSlot {
			a.OwnActiveMask[i]++
		}
	}
	for i := uint8(0); i < a.OppActiveMaskLen; i++ {
		if a.OppActiveMask[i] >= globalSlot {
			a.OppActiveMask[i]++
		}
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
		// EP-PBS: clear the popped slot.
		a.removeOwnActive(a.OwnHandLen)
		a.removeOppActive(a.OwnHandLen)
		a.SlotTags[a.OwnHandLen] = TagUnk
		a.SlotBuckets[a.OwnHandLen] = 0
		a.OwnHand[a.OwnHandLen] = KnownCardInfo{}
	}

	// Reconcile opponent hand.
	for a.OppHandLen < actualOpp {
		a.addOppUnknown()
	}
	for a.OppHandLen > actualOpp && a.OppHandLen > 0 {
		a.OppHandLen--
		// EP-PBS: clear the popped opp slot.
		slotIdx := OppSlotsStart + a.OppHandLen
		a.removeOwnActive(slotIdx)
		a.removeOppActive(slotIdx)
		a.SlotTags[slotIdx] = TagUnk
		a.SlotBuckets[slotIdx] = 0
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

// ApplyMemoryDecay runs per-turn memory decay or eviction according to MemoryArchetype.
// It should be called once per turn (e.g., at the start of the agent's decision step).
// rng must not be nil for MemoryDecaying; it is unused for MemoryPerfect/MemoryHumanLike.
func (a *AgentState) ApplyMemoryDecay(rng *rand.Rand) {
	switch a.MemoryArchetype {
	case MemoryPerfect:
		// No decay — retain all observations indefinitely.
		return

	case MemoryDecaying:
		// Each PrivOwn slot decays to TagUnk with probability p = 1 - exp(-λ).
		lambda := a.MemoryDecayLambda
		if lambda <= 0 {
			return
		}
		p := float32(1.0) - float32(math.Exp(float64(-lambda)))
		i := uint8(0)
		for i < a.OwnActiveMaskLen {
			if rng.Float32() < p {
				slot := a.OwnActiveMask[i]
				// Decay: clear tag, bucket, remove from active mask.
				a.SlotTags[slot] = TagUnk
				a.SlotBuckets[slot] = 0
				// Remove by swapping with last entry.
				a.OwnActiveMaskLen--
				a.OwnActiveMask[i] = a.OwnActiveMask[a.OwnActiveMaskLen]
				a.OwnActiveMask[a.OwnActiveMaskLen] = 0
				// Do NOT increment i — recheck this position.
			} else {
				i++
			}
		}

	case MemoryHumanLike:
		// Enforce capacity limit by evicting lowest-saliency slots.
		cap := a.MemoryCapacity
		if cap == 0 {
			cap = MaxActiveMask
		}
		for a.OwnActiveMaskLen > cap {
			// Find the slot with minimum saliency.
			minSal := float32(math.MaxFloat32)
			minIdx := uint8(0)
			for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
				slot := a.OwnActiveMask[i]
				s := BucketSaliency(a.SlotBuckets[slot])
				if s < minSal {
					minSal = s
					minIdx = i
				}
			}
			// Evict the least-salient slot.
			slot := a.OwnActiveMask[minIdx]
			a.SlotTags[slot] = TagUnk
			a.SlotBuckets[slot] = 0
			a.OwnActiveMaskLen--
			a.OwnActiveMask[minIdx] = a.OwnActiveMask[a.OwnActiveMaskLen]
			a.OwnActiveMask[a.OwnActiveMaskLen] = 0
		}
	}
}

// ---------------------------------------------------------------------------
// N-Player agent state
// ---------------------------------------------------------------------------

// NewNPlayerAgentState creates an AgentState for N-player games.
// Sets up OpponentIDs from playerID and numPlayers.
// For backward compatibility, 2P games also set OpponentID.
func NewNPlayerAgentState(playerID, numPlayers, memoryLevel, timeDecayTurns uint8) AgentState {
	a := AgentState{
		PlayerID:       playerID,
		NumPlayers:     numPlayers,
		MemoryLevel:    memoryLevel,
		TimeDecayTurns: timeDecayTurns,
	}
	idx := uint8(0)
	for i := uint8(0); i < numPlayers; i++ {
		if i != playerID {
			a.OpponentIDs[idx] = i
			idx++
		}
	}
	a.NumOpponents = idx
	// For backward compat with 2P code.
	if numPlayers == 2 {
		a.OpponentID = a.OpponentIDs[0]
	}
	return a
}

// InitializeNPlayer sets up N-player belief state from a freshly dealt game.
func (a *AgentState) InitializeNPlayer(g *engine.GameState) {
	n := g.Rules.NumPlayers
	if n == 0 {
		n = 2
	}
	a.NumPlayers = n

	a.OwnHandLen = g.Players[a.PlayerID].HandLen

	// Own initial peeks: set knowledge and bucket for peeked slots.
	peekIdx0 := g.Players[a.PlayerID].InitialPeek[0]
	peekIdx1 := g.Players[a.PlayerID].InitialPeek[1]
	for _, peekIdx := range []uint8{peekIdx0, peekIdx1} {
		if peekIdx < a.OwnHandLen {
			slot := int(a.PlayerID)*int(engine.MaxHandSize) + int(peekIdx)
			card := g.Players[a.PlayerID].Hand[peekIdx]
			bucket := CardToBucket(card)
			a.NPlayerSlotBuckets[slot] = bucket
			a.NPlayerSlotKnown[slot] = true
			a.KnowledgeMask[slot][a.PlayerID] = true
		}
	}

	// Public info.
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

// UpdateNPlayer updates N-player belief state from the last applied action.
// Must be called once after each action is applied to the game state.
func (a *AgentState) UpdateNPlayer(g *engine.GameState) {
	// Update public knowledge.
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

	// Update own hand length.
	a.OwnHandLen = g.Players[a.PlayerID].HandLen

	act := g.LastAction.ActionIdx
	actingPlayer := g.LastAction.ActingPlayer

	switch {
	case act == engine.NPlayerActionDrawStockpile || act == engine.NPlayerActionDrawDiscard:
		// No belief change on draw alone.

	case act == engine.NPlayerActionDiscardNoAbility || act == engine.NPlayerActionDiscardWithAbility:
		// Discarded card is now public; discard top bucket updated above.
		// Mark all players as knowing the top discard slot.

	case act == engine.NPlayerActionCallCambia:
		// Handled via IsCambiaCalled() above.

	case act == engine.NPlayerActionKingSwapNo:
		// No card movements.

	case act == engine.NPlayerActionKingSwapYes:
		a.nplayerProcessKingSwapYes(g, actingPlayer)

	case act == engine.NPlayerActionPassSnap:
		// No belief change.

	default:
		if slot, ok := engine.NPlayerDecodeReplace(act); ok {
			a.nplayerProcessReplace(g, actingPlayer, slot)
		} else if slot, ok := engine.NPlayerDecodePeekOwn(act); ok {
			a.nplayerProcessPeekOwn(g, actingPlayer, slot)
		} else if slot, oppIdx, ok := engine.NPlayerDecodePeekOther(act); ok {
			a.nplayerProcessPeekOther(g, actingPlayer, slot, oppIdx)
		} else if ownSlot, oppSlot, oppIdx, ok := engine.NPlayerDecodeBlindSwap(act); ok {
			a.nplayerProcessBlindSwap(g, actingPlayer, ownSlot, oppSlot, oppIdx)
		} else if ownSlot, oppSlot, oppIdx, ok := engine.NPlayerDecodeKingLook(act); ok {
			a.nplayerProcessKingLook(g, actingPlayer, ownSlot, oppSlot, oppIdx)
		} else if slot, ok := engine.NPlayerDecodeSnapOwn(act); ok {
			a.nplayerProcessSnapOwn(g, actingPlayer, slot)
		} else if slot, oppIdx, ok := engine.NPlayerDecodeSnapOpponent(act); ok {
			a.nplayerProcessSnapOpponent(g, actingPlayer, slot, oppIdx)
		} else if ownIdx, ok := engine.NPlayerDecodeSnapOpponentMove(act); ok {
			a.nplayerProcessSnapOpponentMove(g, actingPlayer, ownIdx)
		}
	}
}

// nplayerSlot returns the global slot index for a player+card slot.
func nplayerSlot(playerID, cardSlot uint8) int {
	return int(playerID)*int(engine.MaxHandSize) + int(cardSlot)
}

// nplayerOpponentAt returns the actual player ID of opponent at index oppIdx
// in the ordered Opponents list for actingPlayer.
func (a *AgentState) nplayerOpponentAt(g *engine.GameState, actingPlayer, oppIdx uint8) uint8 {
	opps := g.Opponents(actingPlayer)
	if int(oppIdx) < len(opps) {
		return opps[oppIdx]
	}
	return 255 // invalid
}

// nplayerSetKnown records that this agent knows the card at a slot.
func (a *AgentState) nplayerSetKnown(slot int, bucket CardBucket) {
	if slot < 0 || slot >= MaxTotalSlots {
		return
	}
	a.NPlayerSlotKnown[slot] = true
	a.NPlayerSlotBuckets[slot] = bucket
	a.KnowledgeMask[slot][a.PlayerID] = true
}

// nplayerSetPlayerKnows marks that a specific player knows the card at slot.
func (a *AgentState) nplayerSetPlayerKnows(slot int, pid uint8) {
	if slot < 0 || slot >= MaxTotalSlots || int(pid) >= MaxKnowledgePlayers {
		return
	}
	a.KnowledgeMask[slot][pid] = true
}

// nplayerClearKnowledge removes all knowledge of a slot (after a swap).
func (a *AgentState) nplayerClearKnowledge(slot int) {
	if slot < 0 || slot >= MaxTotalSlots {
		return
	}
	a.NPlayerSlotKnown[slot] = false
	a.NPlayerSlotBuckets[slot] = 0
	for p := 0; p < MaxKnowledgePlayers; p++ {
		a.KnowledgeMask[slot][p] = false
	}
}

// nplayerSwapKnowledge swaps all knowledge state between two slots.
func (a *AgentState) nplayerSwapKnowledge(slotA, slotB int) {
	if slotA < 0 || slotA >= MaxTotalSlots || slotB < 0 || slotB >= MaxTotalSlots {
		return
	}
	a.NPlayerSlotKnown[slotA], a.NPlayerSlotKnown[slotB] = a.NPlayerSlotKnown[slotB], a.NPlayerSlotKnown[slotA]
	a.NPlayerSlotBuckets[slotA], a.NPlayerSlotBuckets[slotB] = a.NPlayerSlotBuckets[slotB], a.NPlayerSlotBuckets[slotA]
	a.KnowledgeMask[slotA], a.KnowledgeMask[slotB] = a.KnowledgeMask[slotB], a.KnowledgeMask[slotA]
}

func (a *AgentState) nplayerProcessReplace(g *engine.GameState, actingPlayer, slot uint8) {
	globalSlot := nplayerSlot(actingPlayer, slot)
	if actingPlayer == a.PlayerID {
		// We replaced our card; we know the new card.
		card := g.Players[a.PlayerID].Hand[slot]
		a.nplayerSetKnown(globalSlot, CardToBucket(card))
	} else {
		// Opponent replaced: new card is unknown to us. Reset knowledge.
		// The acting player sees their new card.
		a.nplayerClearKnowledge(globalSlot)
		a.nplayerSetPlayerKnows(globalSlot, actingPlayer)
	}
}

func (a *AgentState) nplayerProcessPeekOwn(g *engine.GameState, actingPlayer, slot uint8) {
	globalSlot := nplayerSlot(actingPlayer, slot)
	a.nplayerSetPlayerKnows(globalSlot, actingPlayer)
	if actingPlayer == a.PlayerID {
		card := g.LastAction.RevealedCard
		a.nplayerSetKnown(globalSlot, CardToBucket(card))
	}
}

func (a *AgentState) nplayerProcessPeekOther(g *engine.GameState, actingPlayer, slot, oppIdx uint8) {
	targetPlayer := a.nplayerOpponentAt(g, actingPlayer, oppIdx)
	if targetPlayer == 255 {
		return
	}
	globalSlot := nplayerSlot(targetPlayer, slot)
	a.nplayerSetPlayerKnows(globalSlot, actingPlayer)
	if actingPlayer == a.PlayerID {
		card := g.LastAction.RevealedCard
		a.nplayerSetKnown(globalSlot, CardToBucket(card))
	}
}

func (a *AgentState) nplayerProcessBlindSwap(g *engine.GameState, actingPlayer, ownSlot, oppSlot, oppIdx uint8) {
	targetPlayer := a.nplayerOpponentAt(g, actingPlayer, oppIdx)
	if targetPlayer == 255 {
		return
	}
	slotA := nplayerSlot(actingPlayer, ownSlot)
	slotB := nplayerSlot(targetPlayer, oppSlot)
	// Swap all knowledge bits — both slots physically moved.
	a.nplayerSwapKnowledge(slotA, slotB)
	// After a blind swap, neither actor knows what they received (they didn't look).
	// Clear both slots' knowledge (the swap was blind).
	a.nplayerClearKnowledge(slotA)
	a.nplayerClearKnowledge(slotB)
}

func (a *AgentState) nplayerProcessKingLook(g *engine.GameState, actingPlayer, ownSlot, oppSlot, oppIdx uint8) {
	targetPlayer := a.nplayerOpponentAt(g, actingPlayer, oppIdx)
	if targetPlayer == 255 {
		return
	}
	slotA := nplayerSlot(actingPlayer, ownSlot)
	slotB := nplayerSlot(targetPlayer, oppSlot)
	// Acting player sees both cards.
	a.nplayerSetPlayerKnows(slotA, actingPlayer)
	a.nplayerSetPlayerKnows(slotB, actingPlayer)
	if actingPlayer == a.PlayerID {
		card := g.LastAction.RevealedCard
		a.nplayerSetKnown(slotA, CardToBucket(card))
		// Also look at target player's card via actual hand.
		oppCard := g.Players[targetPlayer].Hand[oppSlot]
		a.nplayerSetKnown(slotB, CardToBucket(oppCard))
	}
}

func (a *AgentState) nplayerProcessKingSwapYes(g *engine.GameState, actingPlayer uint8) {
	ownIdx := g.LastAction.SwapOwnIdx
	oppIdx := g.LastAction.SwapOppIdx
	// Find which player held the opposed slot. We need target player.
	// SwapOppIdx is relative to Opponents(actingPlayer)[0] for 2P; in N-player,
	// we infer the target from LastAction.RevealedOwner if available.
	targetPlayer := g.LastAction.RevealedOwner
	slotA := nplayerSlot(actingPlayer, ownIdx)
	slotB := nplayerSlot(targetPlayer, oppIdx)
	a.nplayerSwapKnowledge(slotA, slotB)
	// After swap, acting player still knows both slots (they looked during king look).
	a.nplayerSetPlayerKnows(slotA, actingPlayer)
	a.nplayerSetPlayerKnows(slotB, actingPlayer)
	// But unless this agent IS actingPlayer, we don't know the cards.
	// Actual knowledge is unchanged for us from what nplayerSwapKnowledge did.
}

func (a *AgentState) nplayerProcessSnapOwn(g *engine.GameState, actingPlayer, slot uint8) {
	// Snap outcome: if successful, the card is publicly revealed and removed.
	// We just clear knowledge of that slot; hand shrinks (tracked via OwnHandLen update above).
	globalSlot := nplayerSlot(actingPlayer, slot)
	if g.LastAction.SnapSuccess {
		// Card is now gone — mark all knowledge cleared.
		a.nplayerClearKnowledge(globalSlot)
	}
	// If snap failed: penalty cards added (unknown) — OwnHandLen already updated.
}

func (a *AgentState) nplayerProcessSnapOpponent(g *engine.GameState, actingPlayer, slot, oppIdx uint8) {
	targetPlayer := a.nplayerOpponentAt(g, actingPlayer, oppIdx)
	if targetPlayer == 255 {
		return
	}
	globalSlot := nplayerSlot(targetPlayer, slot)
	if g.LastAction.SnapSuccess {
		a.nplayerClearKnowledge(globalSlot)
	}
}

func (a *AgentState) nplayerProcessSnapOpponentMove(g *engine.GameState, actingPlayer, ownIdx uint8) {
	// Acting player moves their card at ownIdx to a target player's hand.
	globalSlot := nplayerSlot(actingPlayer, ownIdx)
	// Card moves out of actingPlayer's hand — clear its knowledge.
	a.nplayerClearKnowledge(globalSlot)
	// Target player gains a card (unknown position), tracked via hand length.
}

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

// ---------------------------------------------------------------------------
// EP-PBS helper methods
// ---------------------------------------------------------------------------

// appendOwnActive appends slotIdx to OwnActiveMask.
// If the mask is full, evicts the slot with minimum saliency first.
// No-op if slotIdx is already present.
func (a *AgentState) appendOwnActive(slotIdx uint8) {
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		if a.OwnActiveMask[i] == slotIdx {
			return // already present
		}
	}
	if a.OwnActiveMaskLen < MaxActiveMask {
		a.OwnActiveMask[a.OwnActiveMaskLen] = slotIdx
		a.OwnActiveMaskLen++
		return
	}
	a.evictOwnSaliency()
	a.OwnActiveMask[a.OwnActiveMaskLen] = slotIdx
	a.OwnActiveMaskLen++
}

// appendOppActive appends slotIdx to OppActiveMask.
// If the mask is full, evicts the oldest entry (FIFO) first.
// No-op if slotIdx is already present.
func (a *AgentState) appendOppActive(slotIdx uint8) {
	for i := uint8(0); i < a.OppActiveMaskLen; i++ {
		if a.OppActiveMask[i] == slotIdx {
			return // already present
		}
	}
	if a.OppActiveMaskLen < MaxActiveMask {
		a.OppActiveMask[a.OppActiveMaskLen] = slotIdx
		a.OppActiveMaskLen++
		return
	}
	a.evictOppFIFO()
	a.OppActiveMask[a.OppActiveMaskLen] = slotIdx
	a.OppActiveMaskLen++
}

// removeOwnActive removes slotIdx from OwnActiveMask (no-op if absent).
func (a *AgentState) removeOwnActive(slotIdx uint8) {
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		if a.OwnActiveMask[i] == slotIdx {
			for j := i; j < a.OwnActiveMaskLen-1; j++ {
				a.OwnActiveMask[j] = a.OwnActiveMask[j+1]
			}
			a.OwnActiveMaskLen--
			a.OwnActiveMask[a.OwnActiveMaskLen] = 0
			return
		}
	}
}

// removeOppActive removes slotIdx from OppActiveMask (no-op if absent).
func (a *AgentState) removeOppActive(slotIdx uint8) {
	for i := uint8(0); i < a.OppActiveMaskLen; i++ {
		if a.OppActiveMask[i] == slotIdx {
			for j := i; j < a.OppActiveMaskLen-1; j++ {
				a.OppActiveMask[j] = a.OppActiveMask[j+1]
			}
			a.OppActiveMaskLen--
			a.OppActiveMask[a.OppActiveMaskLen] = 0
			return
		}
	}
}

// evictOwnSaliency evicts the entry with minimum BucketSaliency from OwnActiveMask.
// Sets the evicted slot's tag to TagUnk and clears its bucket.
func (a *AgentState) evictOwnSaliency() {
	if a.OwnActiveMaskLen == 0 {
		return
	}
	minIdx := uint8(0)
	minSal := BucketSaliency(a.SlotBuckets[a.OwnActiveMask[0]])
	for i := uint8(1); i < a.OwnActiveMaskLen; i++ {
		sal := BucketSaliency(a.SlotBuckets[a.OwnActiveMask[i]])
		if sal < minSal {
			minSal = sal
			minIdx = i
		}
	}
	evictedSlot := a.OwnActiveMask[minIdx]
	a.SlotTags[evictedSlot] = TagUnk
	a.SlotBuckets[evictedSlot] = 0
	for j := minIdx; j < a.OwnActiveMaskLen-1; j++ {
		a.OwnActiveMask[j] = a.OwnActiveMask[j+1]
	}
	a.OwnActiveMaskLen--
	a.OwnActiveMask[a.OwnActiveMaskLen] = 0
}

// evictOppFIFO evicts the oldest entry (index 0) from OppActiveMask.
// Sets the evicted slot's tag to TagUnk.
func (a *AgentState) evictOppFIFO() {
	if a.OppActiveMaskLen == 0 {
		return
	}
	evictedSlot := a.OppActiveMask[0]
	a.SlotTags[evictedSlot] = TagUnk
	for j := uint8(0); j < a.OppActiveMaskLen-1; j++ {
		a.OppActiveMask[j] = a.OppActiveMask[j+1]
	}
	a.OppActiveMaskLen--
	a.OppActiveMask[a.OppActiveMaskLen] = 0
}

// setOwnSlotKnown transitions slotIdx to TagPrivOwn (we peeked it).
// Handles all prior-tag cases per EP-PBS transition rules.
func (a *AgentState) setOwnSlotKnown(slotIdx uint8, bucket CardBucket) {
	prevTag := a.SlotTags[slotIdx]
	a.SlotBuckets[slotIdx] = bucket
	switch prevTag {
	case TagUnk:
		a.SlotTags[slotIdx] = TagPrivOwn
		a.appendOwnActive(slotIdx)
	case TagPrivOpp:
		// Opp knew it; we now know it too → TagPub.
		a.SlotTags[slotIdx] = TagPub
		a.removeOppActive(slotIdx)
	case TagPrivOwn:
		// Re-peek: update bucket in place, mask unchanged.
	case TagPub:
		// Already public: just update bucket.
	}
}

// setOppSlotPrivOpp transitions slotIdx to TagPrivOpp (opp peeked it).
// Handles all prior-tag cases per EP-PBS transition rules.
func (a *AgentState) setOppSlotPrivOpp(slotIdx uint8) {
	prevTag := a.SlotTags[slotIdx]
	switch prevTag {
	case TagUnk:
		a.SlotTags[slotIdx] = TagPrivOpp
		a.appendOppActive(slotIdx)
	case TagPrivOwn:
		// We knew it; opp now knows it too → TagPub.
		a.SlotTags[slotIdx] = TagPub
		a.removeOwnActive(slotIdx)
	case TagPrivOpp:
		// Opp re-peeks: refresh FIFO position.
		a.removeOppActive(slotIdx)
		a.appendOppActive(slotIdx)
	case TagPub:
		// Already public.
	}
}

// eppbsSetSlotUnk forces slotIdx to TagUnk, removing it from all active masks.
func (a *AgentState) eppbsSetSlotUnk(slotIdx uint8) {
	prev := a.SlotTags[slotIdx]
	switch prev {
	case TagPrivOwn:
		a.removeOwnActive(slotIdx)
	case TagPrivOpp:
		a.removeOppActive(slotIdx)
	}
	a.SlotTags[slotIdx] = TagUnk
	a.SlotBuckets[slotIdx] = 0
}

// eppbsForceOwnSlotKnown directly sets slotIdx to TagPrivOwn (used for Replace).
// Unlike setOwnSlotKnown, this treats the slot as having a brand-new card regardless
// of prior tag, so it always results in TagPrivOwn (not TagPub).
func (a *AgentState) eppbsForceOwnSlotKnown(slotIdx uint8, bucket CardBucket) {
	prev := a.SlotTags[slotIdx]
	switch prev {
	case TagPrivOpp:
		a.removeOppActive(slotIdx)
		a.appendOwnActive(slotIdx)
	case TagUnk, TagPub:
		a.appendOwnActive(slotIdx)
	case TagPrivOwn:
		// Already in OwnActiveMask; update bucket only.
	}
	a.SlotTags[slotIdx] = TagPrivOwn
	a.SlotBuckets[slotIdx] = bucket
}

// eppbsForceOppSlotPrivOpp directly sets slotIdx to TagPrivOpp (used for Replace).
// Unlike setOppSlotPrivOpp, always results in TagPrivOpp (opp has new card we don't know).
func (a *AgentState) eppbsForceOppSlotPrivOpp(slotIdx uint8) {
	prev := a.SlotTags[slotIdx]
	switch prev {
	case TagPrivOwn:
		a.removeOwnActive(slotIdx)
		a.appendOppActive(slotIdx)
	case TagUnk, TagPub:
		a.appendOppActive(slotIdx)
	case TagPrivOpp:
		// Refresh FIFO position.
		a.removeOppActive(slotIdx)
		a.appendOppActive(slotIdx)
	}
	a.SlotTags[slotIdx] = TagPrivOpp
	a.SlotBuckets[slotIdx] = 0
}

// eppbsSwapSlots swaps the EP-PBS state (tag + bucket) between two slot indices,
// and updates all active mask references accordingly.
func (a *AgentState) eppbsSwapSlots(slotA, slotB uint8) {
	a.SlotTags[slotA], a.SlotTags[slotB] = a.SlotTags[slotB], a.SlotTags[slotA]
	a.SlotBuckets[slotA], a.SlotBuckets[slotB] = a.SlotBuckets[slotB], a.SlotBuckets[slotA]
	// Update mask references: replace slotA↔slotB in both masks.
	for i := uint8(0); i < a.OwnActiveMaskLen; i++ {
		if a.OwnActiveMask[i] == slotA {
			a.OwnActiveMask[i] = slotB
		} else if a.OwnActiveMask[i] == slotB {
			a.OwnActiveMask[i] = slotA
		}
	}
	for i := uint8(0); i < a.OppActiveMaskLen; i++ {
		if a.OppActiveMask[i] == slotA {
			a.OppActiveMask[i] = slotB
		} else if a.OppActiveMask[i] == slotB {
			a.OppActiveMask[i] = slotA
		}
	}
}
