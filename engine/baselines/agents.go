package baselines

import (
	engine "github.com/jason-s-yu/cambia/engine"
)

// Kind names one baseline policy. The values are part of the FFI contract and
// are mirrored by cfr/src/agents/go_baselines.py.
type Kind uint8

const (
	KindRandom Kind = iota
	KindRandomNoCambia
	KindRandomLateCambia
	KindImperfectGreedy
	KindMemoryHeuristic
	KindAggressiveSnap
	numKinds
)

// ValidKind reports whether k names a policy this package implements.
func ValidKind(k Kind) bool { return k < numKinds }

// defaultLateCambiaTurns mirrors RandomLateCambiaAgent's n_turns default.
const defaultLateCambiaTurns = 8

// Decision is one baseline's answer for one state.
//
// A heuristic policy names a single action. A random policy cannot: its choice
// is a uniform draw, and mean_imp is a historical metric whose recorded numbers
// depend on the exact Python RNG stream, so the draw stays on the Python side.
// Candidates is the sequence that stream draws against, in the same ascending
// index order the reference's filtered legal-action list is in, so
// `rng.choice(range(len(Candidates)))` picks the same element the reference's
// `rng.choice(filtered)` picks. Candidates aliases a buffer the Agent reuses,
// so a caller that keeps it past the next Choose must copy it.
type Decision struct {
	Action     uint16
	Uniform    bool
	Candidates []uint16
}

// Agent is one seat's baseline policy and, for the memory-carrying policies,
// its belief about the two hands it tracks.
type Agent struct {
	Kind            Kind
	Seat            uint8
	CambiaThreshold int
	LateCambiaTurns int

	opponent uint8
	inited   bool
	own      slotMemory
	opp      slotMemory
	cand     []uint16
}

// New builds a baseline for one seat. cambiaThreshold is the config's
// agents.greedy_agent.cambia_call_threshold, which ImperfectGreedyAgent and
// MemoryHeuristicAgent read; AggressiveSnapAgent uses its own constants and
// the random policies use none. lateCambiaTurns is RandomLateCambiaAgent's
// n_turns; a non-positive value takes the reference default.
func New(kind Kind, seat uint8, cambiaThreshold, lateCambiaTurns int) *Agent {
	if lateCambiaTurns <= 0 {
		lateCambiaTurns = defaultLateCambiaTurns
	}
	return &Agent{
		Kind:            kind,
		Seat:            seat,
		CambiaThreshold: cambiaThreshold,
		LateCambiaTurns: lateCambiaTurns,
		opponent:        1 - seat,
	}
}

// Reset marks the agent as belonging to a new game. Memory is rebuilt on the
// next Choose rather than here, because the reference initializes it on its
// first decision of the game and reads the hands as they stand at that moment,
// not as they were dealt.
func (a *Agent) Reset() { a.inited = false }

// aggressive Cambia thresholds, from AggressiveSnapAgent.
const (
	aggressiveHandSizeThreshold = 2
	aggressiveValueThreshold    = 4
)

// lateGameCambiaTurn is the turn at which the two threshold policies call
// Cambia regardless of what they know.
const lateGameCambiaTurn = 20

// Choose returns the action this baseline takes in g. ok is false when the
// state has no legal action at all, which the reference reports by raising.
func (a *Agent) Choose(g *engine.GameState) (Decision, bool) {
	mask := g.LegalActions()
	sample, ok := lowestLegal(&mask)
	if !ok {
		return Decision{}, false
	}

	switch a.Kind {
	case KindRandom, KindRandomNoCambia, KindRandomLateCambia:
		return a.chooseRandom(g, &mask)
	}

	a.ensureInitialized(g)

	if act, ok := a.abilityPhase(g, &mask, sample); ok {
		return Decision{Action: act}, true
	}
	if act, ok := a.snapMovePhase(&mask, sample); ok {
		return Decision{Action: act}, true
	}
	if g.Snap.Active {
		return Decision{Action: a.snapPhase(g, &mask, sample)}, true
	}
	if isLegal(&mask, engine.ActionCallCambia) && a.wantsCambia(g) {
		return Decision{Action: engine.ActionCallCambia}, true
	}
	if act, ok := a.postDraw(g, &mask, sample); ok {
		return Decision{Action: act}, true
	}
	if isLegal(&mask, engine.ActionDrawStockpile) {
		return Decision{Action: engine.ActionDrawStockpile}, true
	}
	return Decision{Action: sample}, true
}

// chooseRandom builds the candidate sequence a random policy draws against:
// every legal action, with Cambia filtered out while the policy suppresses it,
// and the unfiltered set as the fallback when the filter empties it.
func (a *Agent) chooseRandom(g *engine.GameState, mask *[3]uint64) (Decision, bool) {
	suppress := false
	switch a.Kind {
	case KindRandomNoCambia:
		suppress = true
	case KindRandomLateCambia:
		suppress = int(g.TurnNumber) < a.LateCambiaTurns
	}

	a.cand = a.cand[:0]
	for idx := uint16(0); idx < engine.NumActions; idx++ {
		if !isLegal(mask, idx) {
			continue
		}
		if suppress && idx == engine.ActionCallCambia {
			continue
		}
		a.cand = append(a.cand, idx)
	}
	if len(a.cand) == 0 {
		for idx := uint16(0); idx < engine.NumActions; idx++ {
			if isLegal(mask, idx) {
				a.cand = append(a.cand, idx)
			}
		}
	}
	if len(a.cand) == 0 {
		return Decision{}, false
	}
	return Decision{Uniform: true, Candidates: a.cand}, true
}

// ensureInitialized rebuilds the memory model from the live state on the first
// decision of a game, which is when the reference's _init_memory runs.
func (a *Agent) ensureInitialized(g *engine.GameState) {
	if a.inited {
		return
	}
	a.inited = true

	n := g.NumActivePlayers()
	if n <= 2 {
		a.opponent = 1 - a.Seat
	} else {
		a.opponent = (a.Seat + 1) % n
	}

	a.own.reset()
	a.opp.reset()

	ownLen := g.Players[a.Seat].HandLen
	for i := uint8(0); i < ownLen; i++ {
		a.own.setUnknown(i)
	}
	peek := g.Rules.InitialViewCount
	if peek > ownLen {
		peek = ownLen
	}
	for i := uint8(0); i < peek; i++ {
		c := g.Players[a.Seat].Hand[i]
		a.own.set(i, cardValue(c), cardRank(c))
	}

	oppLen := g.Players[a.opponent].HandLen
	for i := uint8(0); i < oppLen; i++ {
		a.opp.setUnknown(i)
	}
}

// peekOwn and peekOpp record a slot the rules just revealed, reading the true
// hand exactly as the reference's _update_memory_peek_* do.
func (a *Agent) peekOwn(g *engine.GameState, slot uint8) {
	if slot < g.Players[a.Seat].HandLen {
		c := g.Players[a.Seat].Hand[slot]
		a.own.set(slot, cardValue(c), cardRank(c))
	}
}

func (a *Agent) peekOpp(g *engine.GameState, slot uint8) {
	if slot < g.Players[a.opponent].HandLen {
		c := g.Players[a.opponent].Hand[slot]
		a.opp.set(slot, cardValue(c), cardRank(c))
	}
}

// abilityPhase is _handle_ability_phase_imperfect. AggressiveSnapAgent carries
// its own copy of this handler in the reference; the two are decision-identical
// on every branch, so one implementation serves all three memory policies.
func (a *Agent) abilityPhase(g *engine.GameState, mask *[3]uint64, sample uint16) (uint16, bool) {
	switch {
	case sample >= engine.ActionBasePeekOwn && sample < engine.ActionBasePeekOther:
		if slot, ok := a.own.firstUnknownSlot(); ok {
			idx := engine.EncodePeekOwn(slot)
			if isLegal(mask, idx) {
				a.peekOwn(g, slot)
				return idx, true
			}
		}
		// The reference falls back to the first legal peek-own, which is the
		// sample itself: the sample IS the lowest legal index and this branch
		// only runs when it decodes to a peek-own.
		a.peekOwn(g, uint8(sample-engine.ActionBasePeekOwn))
		return sample, true

	case sample >= engine.ActionBasePeekOther && sample < engine.ActionBaseBlindSwap:
		if slot, ok := a.opp.firstUnknownSlot(); ok {
			idx := engine.EncodePeekOther(slot)
			if isLegal(mask, idx) {
				a.peekOpp(g, slot)
				return idx, true
			}
		}
		a.peekOpp(g, uint8(sample-engine.ActionBasePeekOther))
		return sample, true

	case sample >= engine.ActionBaseBlindSwap && sample < engine.ActionBaseKingLook:
		ownHigh, haveOwn := a.own.highestKnownSlot()
		oppUnknown, haveOpp := a.opp.firstUnknownSlot()
		if haveOwn && haveOpp {
			idx := engine.EncodeBlindSwap(ownHigh, oppUnknown)
			if isLegal(mask, idx) {
				a.own.setUnknown(ownHigh)
				return idx, true
			}
		}
		// No memory update on the fallback, matching the reference.
		return sample, true

	case sample >= engine.ActionBaseKingLook && sample < engine.ActionKingSwapNo:
		// The reference's `_find_highest_known_own_slot() or 0` and its
		// `next(..., 0)` both fall back to slot 0.
		ownHigh, _ := a.own.highestKnownSlot()
		oppUnknown, _ := a.opp.firstUnknownSlot()
		idx := engine.EncodeKingLook(ownHigh, oppUnknown)
		if isLegal(mask, idx) {
			return idx, true
		}
		return sample, true

	case sample == engine.ActionKingSwapNo || sample == engine.ActionKingSwapYes:
		if g.Pending.Type == engine.PendingKingDecision && g.NumActivePlayers() == 2 {
			ownCard := engine.Card(g.Pending.Data[2])
			targetCard := engine.Card(g.Pending.Data[3])
			if ownCard != engine.EmptyCard && targetCard != engine.EmptyCard &&
				cardValue(targetCard) < cardValue(ownCard) {
				return engine.ActionKingSwapYes, true
			}
		}
		return engine.ActionKingSwapNo, true
	}
	return 0, false
}

// snapMovePhase is _handle_snap_move_imperfect: hand over the highest-valued
// own card, charging unknown slots the expected value.
func (a *Agent) snapMovePhase(mask *[3]uint64, sample uint16) (uint16, bool) {
	if sample < engine.ActionBaseSnapOpponentMove || sample >= engine.NumActions {
		return 0, false
	}
	best, ok := a.own.highestEffectiveSlot()
	if !ok {
		best = 0
	}
	targetSlot := uint8((sample - engine.ActionBaseSnapOpponentMove) % engine.MaxHandSize)
	idx := engine.EncodeSnapOpponentMove(best, targetSlot)
	if isLegal(mask, idx) {
		src, present := a.own.get(best)
		a.opp.setFrom(targetSlot, src, present)
		a.own.remove(best)
		return idx, true
	}
	return sample, true
}

// snapPhase is the snap-window branch: snap a known match, else pass. The
// aggressive policy also snaps a known opponent match, which is the one place
// its snap handling differs.
func (a *Agent) snapPhase(g *engine.GameState, mask *[3]uint64, sample uint16) uint16 {
	top := g.DiscardTop()
	haveTop := top != engine.EmptyCard
	topRank := cardRank(top)

	for slot := uint8(0); slot < engine.MaxHandSize; slot++ {
		idx := engine.EncodeSnapOwn(slot)
		if isLegal(mask, idx) && a.own.rankMatches(slot, topRank, haveTop) {
			return idx
		}
	}
	if a.Kind == KindAggressiveSnap {
		for slot := uint8(0); slot < engine.MaxHandSize; slot++ {
			idx := engine.EncodeSnapOpponent(slot)
			if isLegal(mask, idx) && a.opp.rankMatches(slot, topRank, haveTop) {
				return idx
			}
		}
	}
	if isLegal(mask, engine.ActionPassSnap) {
		return engine.ActionPassSnap
	}
	return sample
}

// wantsCambia is the Cambia branch of each policy, reached only when calling
// Cambia is legal.
func (a *Agent) wantsCambia(g *engine.GameState) bool {
	known := a.own.knownCount()
	estimate := a.own.estimatedValue()
	if a.Kind == KindAggressiveSnap {
		return a.own.size() <= aggressiveHandSizeThreshold ||
			(known >= 3 && estimate <= float64(aggressiveValueThreshold+4)) ||
			int(g.TurnNumber) >= lateGameCambiaTurn
	}
	if known >= 3 && estimate <= float64(a.CambiaThreshold+4) {
		return true
	}
	return int(g.TurnNumber) >= lateGameCambiaTurn
}

// postDraw is the discard-or-replace branch. It runs only when a discard or a
// replace is legal, which is what the reference's `any(isinstance(...))` test
// over the legal set amounts to.
func (a *Agent) postDraw(g *engine.GameState, mask *[3]uint64, sample uint16) (uint16, bool) {
	if !hasDiscardOrReplace(mask) {
		return 0, false
	}

	drawn := engine.EmptyCard
	if g.Pending.Type == engine.PendingDiscard {
		drawn = engine.Card(g.Pending.Data[0])
	}
	if drawn == engine.EmptyCard {
		return a.plainDiscard(mask, sample), true
	}

	drawnValue := cardValue(drawn)
	drawnRank := cardRank(drawn)

	switch a.Kind {
	case KindImperfectGreedy:
		// The best strict improvement over a known slot, first-wins on a tie
		// because the reference compares reductions with a strict `>`.
		bestSlot := uint8(0)
		bestReduction := 0
		haveBest := false
		for i := range a.own.entries {
			e := a.own.entries[i]
			if !e.known || drawnValue >= e.value {
				continue
			}
			if e.value-drawnValue > bestReduction {
				bestReduction = e.value - drawnValue
				bestSlot = e.slot
				haveBest = true
			}
		}
		if !haveBest && drawnValue <= a.CambiaThreshold {
			if slot, ok := a.own.firstUnknownSlot(); ok {
				bestSlot = slot
				haveBest = true
			}
		}
		if haveBest {
			idx := engine.EncodeReplace(bestSlot)
			if isLegal(mask, idx) {
				a.own.set(bestSlot, drawnValue, drawnRank)
				return idx, true
			}
		}

	case KindMemoryHeuristic, KindAggressiveSnap:
		if slot, ok := a.own.highestKnownSlot(); ok {
			e, _ := a.own.get(slot)
			if drawnValue < e.value {
				idx := engine.EncodeReplace(slot)
				if isLegal(mask, idx) {
					a.own.set(slot, drawnValue, drawnRank)
					return idx, true
				}
			}
		}
		if a.Kind == KindMemoryHeuristic && drawnValue <= 3 {
			if slot, ok := a.own.firstUnknownSlot(); ok {
				idx := engine.EncodeReplace(slot)
				if isLegal(mask, idx) {
					a.own.set(slot, drawnValue, drawnRank)
					return idx, true
				}
			}
		}
	}

	// Take any ability the drawn card carries. The aggressive policy tests the
	// peek-other ranks first in the reference, but that test is subsumed by the
	// ability test below and reaches the same action.
	if hasAbilityRank(drawnRank) && isLegal(mask, engine.ActionDiscardWithAbility) {
		return engine.ActionDiscardWithAbility, true
	}
	return a.plainDiscard(mask, sample), true
}

// plainDiscard is the reference's
// `ActionDiscard(use_ability=False) if legal else list(legal_actions)[0]`.
func (a *Agent) plainDiscard(mask *[3]uint64, sample uint16) uint16 {
	if isLegal(mask, engine.ActionDiscardNoAbility) {
		return engine.ActionDiscardNoAbility
	}
	return sample
}

// hasAbilityRank is the reference's `drawn_card.rank in [SEVEN, EIGHT, NINE,
// TEN, KING]`. Note that it excludes Jack and Queen, whose blind swap the
// engine does treat as an ability.
func hasAbilityRank(rank uint8) bool {
	switch rank {
	case engine.RankSeven, engine.RankEight, engine.RankNine, engine.RankTen, engine.RankKing:
		return true
	}
	return false
}

// isLegal reports whether action idx is set in the legal-action bitmask.
func isLegal(mask *[3]uint64, idx uint16) bool {
	if idx >= engine.NumActions {
		return false
	}
	return mask[idx/64]>>(idx%64)&1 == 1
}

// lowestLegal returns the lowest legal action index, which is what the
// reference's `next(iter(legal_actions))` reads off the ascending decoded list.
func lowestLegal(mask *[3]uint64) (uint16, bool) {
	for idx := uint16(0); idx < engine.NumActions; idx++ {
		if mask[idx/64]>>(idx%64)&1 == 1 {
			return idx, true
		}
	}
	return 0, false
}

// hasDiscardOrReplace reports whether any discard or replace action is legal.
func hasDiscardOrReplace(mask *[3]uint64) bool {
	if isLegal(mask, engine.ActionDiscardNoAbility) || isLegal(mask, engine.ActionDiscardWithAbility) {
		return true
	}
	for slot := uint8(0); slot < engine.MaxHandSize; slot++ {
		if isLegal(mask, engine.EncodeReplace(slot)) {
			return true
		}
	}
	return false
}
