package engine

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// TournamentHouseRules returns the enforced house rules for circuit/tournament mode.
// Per T1: allowDrawFromDiscardPile=true, allowReplaceAbilities=true, lockCallerHand=false.
// All other settings inherit from DefaultHouseRules.
func TournamentHouseRules() HouseRules {
	hr := DefaultHouseRules()
	hr.AllowDrawFromDiscard = true
	hr.AllowReplaceAbilities = true
	hr.LockCallerHand = false
	return hr
}

// ForfeitRoundScore is what a seat that did not play a round out is worth: 41 points, the +2
// sigma statistical maximum for a blind hand, which punishes the absence without corrupting the
// lobby's rating (RULES.md T5, MATCHMAKING.md 8).
//
// It is the default for CircuitConfig.MissedRoundScore and the score the service records for a
// forfeited seat, so a missed circuit round and a forfeited quick-play seat cost the same.
const ForfeitRoundScore = 41

// CircuitFormat represents a tournament circuit length preset.
type CircuitFormat string

const (
	CircuitQuick        CircuitFormat = "quick"        // 8 rounds
	CircuitStandard     CircuitFormat = "standard"     // 12 rounds
	CircuitChampionship CircuitFormat = "championship" // 20 rounds
)

// CircuitConfig holds configuration for a multi-round circuit tournament.
type CircuitConfig struct {
	Format           CircuitFormat
	NumPlayers       int
	NumRounds        int // 0 = auto from Format. Must be multiple of NumPlayers.
	PlayerIDs        []int
	MissedRoundScore int           // Default ForfeitRoundScore
	AbandonThreshold int           // Default 2 consecutive misses
	DisconnectGrace  time.Duration // Default 60s (informational for service layer)
}

// CircuitRoundResult records the outcome of a single round.
type CircuitRoundResult struct {
	RoundNum       int
	PlayerScores   map[int]int // Raw hand scores per player
	Placements     []int       // Player IDs sorted by score ascending (Cambia caller wins ties)
	CambiaCallerID int         // -1 if none called Cambia
	Subsidies      map[int]int // Aggression bonus per player (negative = good)
	DealerID       int
	FirstActorID   int
	Forfeited      map[int]bool // Players who missed this round
}

// CircuitPlayerState tracks a player's progress through the circuit.
type CircuitPlayerState struct {
	PlayerID          int
	CumulativeScore   int            // Post-subsidy cumulative total
	RawCumulative     int            // Pre-subsidy cumulative (tiebreaker #1)
	RoundScores       []int          // Per-round raw scores
	RoundPlacements   []int          // Per-round placement (1-indexed)
	H2HRecord         map[int][2]int // opp_id -> [wins, losses] (tiebreaker #2)
	BestRound         int            // Lowest single-round raw score (tiebreaker #3)
	ConsecutiveMisses int
	Abandoned         bool
}

// CircuitState tracks the full state of an ongoing circuit tournament.
type CircuitState struct {
	Config       CircuitConfig
	Players      []CircuitPlayerState
	Rounds       []CircuitRoundResult
	CurrentRound int
	DealerSeat   int // Index into Config.PlayerIDs array
	Completed    bool
}

// NewCircuit creates and validates a new circuit tournament state.
func NewCircuit(config CircuitConfig) (*CircuitState, error) {
	// Auto-set NumRounds from Format
	if config.NumRounds == 0 {
		if config.Format == "" {
			config.Format = CircuitStandard
		}
		switch config.Format {
		case CircuitQuick:
			config.NumRounds = 8
		case CircuitStandard:
			config.NumRounds = 12
		case CircuitChampionship:
			config.NumRounds = 20
		default:
			config.NumRounds = 12
		}
	}

	if config.NumPlayers < 2 {
		return nil, fmt.Errorf("circuit requires at least 2 players, got %d", config.NumPlayers)
	}
	if config.NumRounds <= 0 {
		return nil, fmt.Errorf("NumRounds must be > 0, got %d", config.NumRounds)
	}
	if config.NumRounds%config.NumPlayers != 0 {
		return nil, fmt.Errorf("NumRounds (%d) must be a multiple of NumPlayers (%d)", config.NumRounds, config.NumPlayers)
	}
	if len(config.PlayerIDs) != config.NumPlayers {
		return nil, fmt.Errorf("len(PlayerIDs) (%d) must equal NumPlayers (%d)", len(config.PlayerIDs), config.NumPlayers)
	}

	// Apply defaults
	if config.MissedRoundScore == 0 {
		config.MissedRoundScore = ForfeitRoundScore
	}
	if config.AbandonThreshold == 0 {
		config.AbandonThreshold = 2
	}
	if config.DisconnectGrace == 0 {
		config.DisconnectGrace = 60 * time.Second
	}

	players := make([]CircuitPlayerState, config.NumPlayers)
	for i, pid := range config.PlayerIDs {
		h2h := make(map[int][2]int)
		for _, other := range config.PlayerIDs {
			if other != pid {
				h2h[other] = [2]int{0, 0}
			}
		}
		players[i] = CircuitPlayerState{
			PlayerID:  pid,
			BestRound: math.MaxInt32,
			H2HRecord: h2h,
		}
	}

	return &CircuitState{
		Config:  config,
		Players: players,
	}, nil
}

// playerIndex returns the index of a player in cs.Players by ID, or -1.
func (cs *CircuitState) playerIndex(playerID int) int {
	for i, p := range cs.Players {
		if p.PlayerID == playerID {
			return i
		}
	}
	return -1
}

// RecordRound records the results of a completed round.
func (cs *CircuitState) RecordRound(scores map[int]int, cambiaCallerID int) error {
	// Validate all non-abandoned players have scores
	for _, p := range cs.Players {
		if p.Abandoned {
			continue
		}
		if _, ok := scores[p.PlayerID]; !ok {
			return fmt.Errorf("missing score for player %d", p.PlayerID)
		}
	}

	// Collect active player IDs for this round
	var activePlayers []int
	for _, p := range cs.Players {
		if !p.Abandoned {
			activePlayers = append(activePlayers, p.PlayerID)
		}
	}

	// Sort by score ascending; Cambia caller wins ties
	sort.SliceStable(activePlayers, func(i, j int) bool {
		si := scores[activePlayers[i]]
		sj := scores[activePlayers[j]]
		if si != sj {
			return si < sj
		}
		// Tie: Cambia caller goes first (lower placement = better)
		if activePlayers[i] == cambiaCallerID {
			return true
		}
		if activePlayers[j] == cambiaCallerID {
			return false
		}
		return activePlayers[i] < activePlayers[j]
	})

	// Build subsidy table based on placement. The format-aware schedule and tie handling
	// live in one place, ComputeAggressionSubsidy (RULES.md T3): H2H (<=2p) -3/0, FFA-4
	// (<=4p) -5/-2/0/0, 5+ -5/-2/-1/0. There is no separate inline table here (cambia-1009).
	n := len(activePlayers)

	// competitionPlacements is 0-indexed with tied players sharing the leading index of
	// their score group (e.g. [0, 1, 1, 3]), which is what ComputeAggressionSubsidy expects
	// so its tie rule (Cambia caller wins ties; two tied non-callers both get the higher
	// bonus) applies uniformly across every tied player in the group.
	competitionPlacements := make([]int, n)
	for idx, pid := range activePlayers {
		if idx > 0 && scores[pid] == scores[activePlayers[idx-1]] {
			competitionPlacements[idx] = competitionPlacements[idx-1]
		} else {
			competitionPlacements[idx] = idx
		}
	}

	callerIdx := -1
	for idx, pid := range activePlayers {
		if pid == cambiaCallerID {
			callerIdx = idx
			break
		}
	}

	subsidyByActiveIdx := ComputeAggressionSubsidy(n, competitionPlacements, callerIdx)
	subsidies := make(map[int]int, n)
	for idx, pid := range activePlayers {
		subsidies[pid] = subsidyByActiveIdx[idx]
	}

	// placement tracks each player's 0-based sorted position (Cambia caller first on a
	// tie), independent of the subsidy schedule above; it feeds RoundPlacements/H2H below.
	placement := make(map[int]int) // playerID -> 0-based placement
	for idx, pid := range activePlayers {
		placement[pid] = idx
	}

	// Update player states
	for i := range cs.Players {
		p := &cs.Players[i]
		if p.Abandoned {
			continue
		}
		score, ok := scores[p.PlayerID]
		if !ok {
			continue
		}
		p.RawCumulative += score
		sub := subsidies[p.PlayerID]
		p.CumulativeScore += score + sub
		p.RoundScores = append(p.RoundScores, score)
		if score < p.BestRound {
			p.BestRound = score
		}
		p.ConsecutiveMisses = 0
		// Record placement (1-indexed)
		pl := placement[p.PlayerID] + 1
		p.RoundPlacements = append(p.RoundPlacements, pl)
	}

	// Update H2H records
	for ii, pid := range activePlayers {
		for jj, opid := range activePlayers {
			if jj <= ii {
				continue
			}
			si := scores[pid]
			sj := scores[opid]
			if si < sj {
				// pid wins
				cs.updateH2H(pid, opid, true)
			} else if sj < si {
				// opid wins
				cs.updateH2H(pid, opid, false)
			} else {
				// Tied - Cambia caller wins
				if pid == cambiaCallerID {
					cs.updateH2H(pid, opid, true)
				} else if opid == cambiaCallerID {
					cs.updateH2H(pid, opid, false)
				}
				// True tie (neither called Cambia) = no update
			}
		}
	}

	// Build result
	result := CircuitRoundResult{
		RoundNum:       cs.CurrentRound + 1,
		PlayerScores:   scores,
		Placements:     activePlayers,
		CambiaCallerID: cambiaCallerID,
		Subsidies:      subsidies,
		DealerID:       cs.NextDealerSeat(),
		FirstActorID:   cs.NextFirstActor(),
		Forfeited:      make(map[int]bool),
	}
	cs.Rounds = append(cs.Rounds, result)

	cs.CurrentRound++
	cs.DealerSeat = (cs.DealerSeat + 1) % len(cs.Config.PlayerIDs)
	if cs.CurrentRound >= cs.Config.NumRounds {
		cs.Completed = true
	}

	return nil
}

// updateH2H updates head-to-head record for a pair. pidWins=true means pid beats opid.
func (cs *CircuitState) updateH2H(pid, opid int, pidWins bool) {
	pi := cs.playerIndex(pid)
	oi := cs.playerIndex(opid)
	if pi < 0 || oi < 0 {
		return
	}

	if pidWins {
		rec := cs.Players[pi].H2HRecord[opid]
		rec[0]++
		cs.Players[pi].H2HRecord[opid] = rec

		rec2 := cs.Players[oi].H2HRecord[pid]
		rec2[1]++
		cs.Players[oi].H2HRecord[pid] = rec2
	} else {
		rec := cs.Players[pi].H2HRecord[opid]
		rec[1]++
		cs.Players[pi].H2HRecord[opid] = rec

		rec2 := cs.Players[oi].H2HRecord[pid]
		rec2[0]++
		cs.Players[oi].H2HRecord[pid] = rec2
	}
}

// RecordMissedRound scores a player as having missed a round (41 points, no subsidy).
func (cs *CircuitState) RecordMissedRound(playerID int) error {
	idx := cs.playerIndex(playerID)
	if idx < 0 {
		return fmt.Errorf("player %d not found", playerID)
	}
	p := &cs.Players[idx]

	missed := cs.Config.MissedRoundScore
	p.RawCumulative += missed
	p.CumulativeScore += missed
	p.RoundScores = append(p.RoundScores, missed)
	if missed < p.BestRound {
		p.BestRound = missed
	}
	p.ConsecutiveMisses++

	if p.ConsecutiveMisses >= cs.Config.AbandonThreshold {
		p.Abandoned = true
		// Score all remaining rounds as missed
		remaining := cs.Config.NumRounds - cs.CurrentRound - 1
		for r := 0; r < remaining; r++ {
			p.RawCumulative += missed
			p.CumulativeScore += missed
			p.RoundScores = append(p.RoundScores, missed)
		}
	}

	return nil
}

// RecordReconnection resets a player's consecutive miss counter.
func (cs *CircuitState) RecordReconnection(playerID int) {
	idx := cs.playerIndex(playerID)
	if idx < 0 {
		return
	}
	cs.Players[idx].ConsecutiveMisses = 0
}

// NextDealerSeat returns the player ID of the current dealer.
func (cs *CircuitState) NextDealerSeat() int {
	return cs.Config.PlayerIDs[cs.DealerSeat]
}

// NextFirstActor returns the player ID who acts first (left of dealer).
func (cs *CircuitState) NextFirstActor() int {
	return cs.Config.PlayerIDs[(cs.DealerSeat+1)%len(cs.Config.PlayerIDs)]
}

// resolveH2HTieGroup reorders a slice of players already level on cumulative and
// raw score (T1 tie-breakers #1-#2) by tie-breaker #2, the head-to-head record:
// the stored pairwise win/loss tally the tied players hold against each other
// (H2HRecord), not a recount of per-round placements or a sum against the whole
// field.
//
// For a group of exactly two this is a direct comparison of their mutual
// record. For a group of three or more it is a mini-table: each player's total
// wins counted only against the other players in this group. Ranking by that
// per-player total is always a strict weak ordering (it sorts a number), so it
// cannot itself cycle; a subgroup that remains level after the mini-table --
// including a genuinely cyclic result, where each player in the subgroup beat
// one tied opponent and lost to another, which is exactly what makes their
// within-group win totals come out equal -- falls through to tie-breaker #3
// (BestRound), then PlayerID, among just that subgroup.
func resolveH2HTieGroup(group []CircuitPlayerState) {
	groupWins := make(map[int]int, len(group))
	for _, p := range group {
		wins := 0
		for _, opp := range group {
			if opp.PlayerID == p.PlayerID {
				continue
			}
			wins += p.H2HRecord[opp.PlayerID][0]
		}
		groupWins[p.PlayerID] = wins
	}

	sort.SliceStable(group, func(i, j int) bool {
		pi := group[i]
		pj := group[j]
		// 2. Head-to-head mini-table wins, descending.
		wi := groupWins[pi.PlayerID]
		wj := groupWins[pj.PlayerID]
		if wi != wj {
			return wi > wj
		}
		// 3. BestRound ascending.
		if pi.BestRound != pj.BestRound {
			return pi.BestRound < pj.BestRound
		}
		// 4. PlayerID ascending (final tiebreak).
		return pi.PlayerID < pj.PlayerID
	})
}

// GetStandings returns a sorted copy of player states by standings.
func (cs *CircuitState) GetStandings() []CircuitPlayerState {
	standings := make([]CircuitPlayerState, len(cs.Players))
	copy(standings, cs.Players)

	// Group first by tie-breakers #1-#2 (CumulativeScore, RawCumulative); PlayerID
	// is only a placeholder order within a group, corrected below by resolveH2HTieGroup.
	sort.SliceStable(standings, func(i, j int) bool {
		pi := standings[i]
		pj := standings[j]

		// 1. CumulativeScore ascending
		if pi.CumulativeScore != pj.CumulativeScore {
			return pi.CumulativeScore < pj.CumulativeScore
		}
		// 2. RawCumulative ascending
		if pi.RawCumulative != pj.RawCumulative {
			return pi.RawCumulative < pj.RawCumulative
		}
		return pi.PlayerID < pj.PlayerID
	})

	// Resolve tie-breaker #2 (head-to-head) within each group still level on
	// CumulativeScore and RawCumulative.
	start := 0
	for start < len(standings) {
		end := start + 1
		for end < len(standings) &&
			standings[end].CumulativeScore == standings[start].CumulativeScore &&
			standings[end].RawCumulative == standings[start].RawCumulative {
			end++
		}
		if end-start >= 2 {
			resolveH2HTieGroup(standings[start:end])
		}
		start = end
	}

	return standings
}

// IsComplete returns true if all rounds have been played.
func (cs *CircuitState) IsComplete() bool {
	return cs.Completed
}
