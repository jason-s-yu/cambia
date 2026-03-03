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
	NumRounds        int           // 0 = auto from Format. Must be multiple of NumPlayers.
	PlayerIDs        []int
	MissedRoundScore int           // Default 41
	AbandonThreshold int           // Default 2 consecutive misses
	DisconnectGrace  time.Duration // Default 60s (informational for service layer)
}

// CircuitRoundResult records the outcome of a single round.
type CircuitRoundResult struct {
	RoundNum       int
	PlayerScores   map[int]int  // Raw hand scores per player
	Placements     []int        // Player IDs sorted by score ascending (Cambia caller wins ties)
	CambiaCallerID int          // -1 if none called Cambia
	Subsidies      map[int]int  // Aggression bonus per player (negative = good)
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
	DealerSeat   int  // Index into Config.PlayerIDs array
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
		config.MissedRoundScore = 41
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

	// Build subsidy table based on placement
	subsidies := make(map[int]int)
	n := len(activePlayers)

	// Determine subsidy schedule
	var scheduleBase []int
	if n >= 5 {
		scheduleBase = []int{-5, -2, -1}
	} else {
		scheduleBase = []int{-5, -2}
	}
	subsidyByPlacement := make([]int, n) // index=placement(0-based), value=subsidy
	for i := range subsidyByPlacement {
		if i < len(scheduleBase) {
			subsidyByPlacement[i] = scheduleBase[i]
		}
		// rest are 0
	}

	// Assign subsidies with tie handling:
	// If two players are tied (same score), and neither called Cambia, BOTH get the higher placement's bonus.
	// Cambia caller wins the tie and gets their earned placement bonus.
	placement := make(map[int]int) // playerID -> 0-based placement
	for idx, pid := range activePlayers {
		placement[pid] = idx
	}

	for idx, pid := range activePlayers {
		score := scores[pid]
		sub := subsidyByPlacement[idx]

		// Check if there's a tie partner who also didn't call cambia
		for jdx, opid := range activePlayers {
			if jdx == idx {
				continue
			}
			if scores[opid] == score && pid != cambiaCallerID && opid != cambiaCallerID {
				// Both tied, neither called Cambia — both get higher placement bonus
				higherPlacement := idx
				if jdx < higherPlacement {
					higherPlacement = jdx
				}
				sub = subsidyByPlacement[higherPlacement]
				break
			}
		}
		subsidies[pid] = sub
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
				// Tied — Cambia caller wins
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

// h2hTotalWins returns total H2H wins for a player across all opponents.
func h2hTotalWins(p CircuitPlayerState) int {
	total := 0
	for _, rec := range p.H2HRecord {
		total += rec[0]
	}
	return total
}

// GetStandings returns a sorted copy of player states by standings.
func (cs *CircuitState) GetStandings() []CircuitPlayerState {
	standings := make([]CircuitPlayerState, len(cs.Players))
	copy(standings, cs.Players)

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
		// 3. H2H wins descending
		wi := h2hTotalWins(pi)
		wj := h2hTotalWins(pj)
		if wi != wj {
			return wi > wj
		}
		// 4. BestRound ascending
		if pi.BestRound != pj.BestRound {
			return pi.BestRound < pj.BestRound
		}
		// 5. PlayerID ascending (final tiebreak)
		return pi.PlayerID < pj.PlayerID
	})

	return standings
}

// IsComplete returns true if all rounds have been played.
func (cs *CircuitState) IsComplete() bool {
	return cs.Completed
}
