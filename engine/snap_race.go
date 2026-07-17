package engine

import "fmt"

// snap_race.go implements the race-ON snap model (HouseRules.SnapRace = true):
// a true N-way race, distinct from the race-OFF sequential discarder-first model
// in snap.go. Per the frozen 2026-07-16 snap ruling (cambia-564):
//
//   - Simultaneous imperfect-info commit: every eligible snapper commits a choice
//     (pass / snap-own / snap-opponent) without the commit mutating any hand, so a
//     later committer observes the same state a race-OFF pre-snap observer would;
//     no committer sees another's commit before choosing.
//   - Uniform-random winner: among the willing committers (kind != pass), one is
//     drawn uniformly at random via the game RNG. This is the deterministic
//     turn-based model of "whoever reacts first wins": the reaction-time race is
//     an internal chance event, outcome-sampled per trajectory exactly like the
//     initial deal, so no explicit chance node is added to the tree.
//   - Losers pay the penalty: every other willing committer draws the snap penalty
//     (they committed to the race and lost), matching the live service's behavior
//     of penalizing late/losing snap attempts.
//   - Window closes on the single winner: at most one snap succeeds per discard.
//
// Commit recording and resolution are intercepted by the action dispatchers
// (ApplyAction / ApplyNPlayerAction) BEFORE the race-OFF snapOwn/snapOpponent
// resolvers, so those resolvers and the entire race-OFF path stay byte-identical.

// DecodeSnapCommit decodes a raw committed snap action index (in either the
// 2-player or the N-player action space; the two snap ranges are disjoint) into
// its intent. ok is false if the index is not a snap action.
func DecodeSnapCommit(action uint16) (kind, slot, oppRel uint8, ok bool) {
	if action == ActionPassSnap || action == NPlayerActionPassSnap {
		return SnapCommitPass, 0, 0, true
	}
	if t, isOwn := ActionIsSnapOwn(action); isOwn {
		return SnapCommitOwn, t, 0, true
	}
	if o, isOpp := ActionIsSnapOpponent(action); isOpp {
		return SnapCommitOpp, o, 0, true
	}
	if t, isOwn := NPlayerDecodeSnapOwn(action); isOwn {
		return SnapCommitOwn, t, 0, true
	}
	if s, rel, isOpp := NPlayerDecodeSnapOpponent(action); isOpp {
		return SnapCommitOpp, s, rel, true
	}
	return 0, 0, 0, false
}

// recordSnapCommit stores one committer's choice without mutating any hand, then
// advances to the next committer. advanceSnapper drives resolveSnapRace once the
// final committer has committed. Works for both action spaces via DecodeSnapCommit.
func (g *GameState) recordSnapCommit(actionIdx uint16) error {
	if !g.Snap.Active {
		return fmt.Errorf("snap phase is not active")
	}
	kind, _, _, ok := DecodeSnapCommit(actionIdx)
	if !ok {
		return fmt.Errorf("snap race commit: unhandled snap action index %d", actionIdx)
	}
	if kind == SnapCommitOpp && !g.Rules.AllowOpponentSnapping {
		return fmt.Errorf("opponent snapping is not allowed by house rules")
	}

	j := g.Snap.CurrentSnapperIdx
	g.Snap.Commits[j] = actionIdx

	// Record the commit as the last action (a commit resolves nothing yet).
	g.LastAction.ActionIdx = actionIdx
	g.LastAction.ActingPlayer = g.Snap.Snappers[j]
	g.LastAction.RevealedCard = EmptyCard
	g.LastAction.RevealedIdx = 0
	g.LastAction.RevealedOwner = 0
	g.LastAction.SnapSuccess = false
	g.LastAction.SnapPenalty = 0

	g.advanceSnapper()
	return nil
}

// resolveSnapRace draws the uniform-random winner among willing committers,
// resolves that winner's snap, penalizes the losing willing committers, and
// ends the phase (or leaves a PendingSnapMove for a winning opponent snap).
// Called by advanceSnapper after the last committer commits when SnapRace is on.
func (g *GameState) resolveSnapRace() {
	n := g.Snap.NumSnappers

	// Collect willing committer slots (kind != pass), in snapper order.
	var willing [MaxPlayers]uint8
	wc := uint8(0)
	for j := uint8(0); j < n; j++ {
		kind, _, _, _ := DecodeSnapCommit(g.Snap.Commits[j])
		if kind != SnapCommitPass {
			willing[wc] = j
			wc++
		}
	}
	if wc == 0 {
		// Everyone passed: no snap, no penalty.
		g.endSnapPhase()
		return
	}

	// Uniform-random winner among the willing committers.
	//
	// CORRECTNESS FENCE: this draws the winner from the game RNG as an internal,
	// outcome-sampled chance transition (no explicit chance node in the tree). That
	// is correct for MCCFR trajectory sampling, but it is NOT enumerable by an exact
	// tree builder that walks chance branches explicitly (e.g. the Python
	// tiny_solver): such a builder would treat this stochastic transition as a
	// sampled-deterministic step and corrupt any exact NashConv computed on a
	// race-ON tree. Exact solving of race-ON requires first exposing this draw as an
	// enumerable chance point. The Python tiny_solver guards against SnapRace=true.
	winSlot := willing[g.randN(uint64(wc))]

	// Penalize the losing willing committers first. drawPenalty only appends to a
	// hand (it never shifts existing indices), so the winner's committed slot
	// indices remain valid regardless of this ordering.
	for k := uint8(0); k < wc; k++ {
		j := willing[k]
		if j == winSlot {
			continue
		}
		loser := g.Snap.Snappers[j]
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(loser)
	}

	// Resolve the winner's committed snap.
	winP := g.Snap.Snappers[winSlot]
	kind, slot, oppRel, _ := DecodeSnapCommit(g.Snap.Commits[winSlot])
	rawIdx := g.Snap.Commits[winSlot]
	pendingMove := false
	switch kind {
	case SnapCommitOwn:
		g.resolveWinnerSnapOwn(winP, slot, rawIdx)
	case SnapCommitOpp:
		pendingMove = g.resolveWinnerSnapOpp(winP, oppRel, slot, rawIdx)
	}

	if pendingMove {
		// Winning opponent snap: the winner must now move a card into the vacated
		// slot. The phase stays active with PendingSnapMove taking priority; the
		// winner's SnapOpponentMove action ends the phase (SnapRace branch in
		// snapOpponentMove / nplayerSnapOpponentMove).
		return
	}
	g.endSnapPhase()
}

// resolveWinnerSnapOwn resolves a winning snap-own commit for player p. It does
// not advance the snapper or end the phase (resolveSnapRace owns phase control).
// Mirrors snapOwn's own-card resolution body.
func (g *GameState) resolveWinnerSnapOwn(p, idx uint8, rawIdx uint16) {
	g.LastAction.ActionIdx = rawIdx
	g.LastAction.ActingPlayer = p

	handLen := g.Players[p].HandLen
	if idx >= handLen {
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(p)
		return
	}

	card := g.Players[p].Hand[idx]
	if card.Rank() == g.Snap.DiscardedRank {
		g.removeCardFromHand(p, idx)
		g.DiscardPile[g.DiscardLen] = card
		g.DiscardLen++
		g.LastAction.SnapSuccess = true
		g.LastAction.RevealedCard = card
		g.LastAction.RevealedIdx = idx
		g.LastAction.RevealedOwner = p
		return
	}
	g.LastAction.SnapSuccess = false
	g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
	g.drawPenalty(p)
}

// resolveWinnerSnapOpp resolves a winning snap-opponent commit for player p,
// targeting the oppRel-th opponent's card at slot. Returns true if the snap
// succeeded and a PendingSnapMove was set (the winner must move a card into the
// vacated slot). Mirrors the resolution body of snapOpponent / nplayerSnapOpponent.
func (g *GameState) resolveWinnerSnapOpp(p, oppRel, slot uint8, rawIdx uint16) bool {
	g.LastAction.ActionIdx = rawIdx
	g.LastAction.ActingPlayer = p

	opponent, ok := g.opponentByRel(p, oppRel)
	if !ok || g.Players[p].HandLen == 0 || slot >= g.Players[opponent].HandLen {
		g.LastAction.SnapSuccess = false
		g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
		g.drawPenalty(p)
		return false
	}

	card := g.Players[opponent].Hand[slot]
	if card.Rank() == g.Snap.DiscardedRank {
		g.removeCardFromHand(opponent, slot)
		g.DiscardPile[g.DiscardLen] = card
		g.DiscardLen++
		g.LastAction.SnapSuccess = true
		g.LastAction.RevealedCard = card
		g.LastAction.RevealedIdx = slot
		g.LastAction.RevealedOwner = opponent
		g.Pending.Type = PendingSnapMove
		g.Pending.PlayerID = p
		g.Pending.Data[0] = opponent
		g.Pending.Data[1] = slot
		return true
	}
	g.LastAction.SnapSuccess = false
	g.LastAction.SnapPenalty = g.Rules.PenaltyDrawCount
	g.drawPenalty(p)
	return false
}

// opponentByRel returns the oppRel-th opponent of player p in ascending player
// order (skipping p), matching Opponents(p)[oppRel] without allocating.
func (g *GameState) opponentByRel(p, oppRel uint8) (uint8, bool) {
	nPlayers := g.Rules.numPlayers()
	seen := uint8(0)
	for i := uint8(0); i < nPlayers; i++ {
		if i == p {
			continue
		}
		if seen == oppRel {
			return i, true
		}
		seen++
	}
	return 0, false
}
