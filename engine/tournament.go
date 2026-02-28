package engine

import (
	"errors"
	"math"
	"sort"
)

// TournamentConfig configures a tournament.
type TournamentConfig struct {
	// Mode: "series", "single_elimination", or "double_elimination"
	Mode string
	// NumRounds is the number of rounds for series mode.
	NumRounds int
	// GamesPerMatchup is best-of-N (default 1).
	GamesPerMatchup int
	// ScoringMethod: "cumulative_score", "wins", or "lowest_cumulative"
	ScoringMethod string
	// SeedingMethod: "random" or "by_score"
	SeedingMethod string
	// PlayerIDs lists all participants.
	PlayerIDs []int
}

// PlayerStanding tracks a player's tournament performance.
type PlayerStanding struct {
	PlayerID  int
	Score     int
	Wins      int
	Losses    int
	Eliminated bool
}

// Matchup represents a single game between two players.
type Matchup struct {
	Player1 int
	Player2 int
	Winner  int   // -1 if not yet played; equals Player1 or Player2 on completion
	Scores  []int // per-player scores from the game
	Bye     bool  // true if Player2 is a bye (Player1 auto-advances)
	Played  bool
}

// BracketRound holds the matchups for one round of an elimination bracket.
type BracketRound struct {
	Matchups []Matchup
}

// TournamentState is the full mutable state of a tournament.
type TournamentState struct {
	Config       TournamentConfig
	Bracket      []BracketRound   // populated for elimination modes
	Standings    []PlayerStanding
	CurrentRound int
	Completed    bool

	// internal: series scheduling
	seriesSchedule [][][2]int // [round][matchup][p1, p2]

	// internal: double-elim losers bracket
	losersRounds []BracketRound
	losersRound  int
}

// NewTournament creates and initialises a TournamentState.
func NewTournament(config TournamentConfig) *TournamentState {
	if config.GamesPerMatchup <= 0 {
		config.GamesPerMatchup = 1
	}
	if config.ScoringMethod == "" {
		config.ScoringMethod = "wins"
	}
	if config.SeedingMethod == "" {
		config.SeedingMethod = "random"
	}

	t := &TournamentState{
		Config: config,
	}

	// Build standings.
	for _, pid := range config.PlayerIDs {
		t.Standings = append(t.Standings, PlayerStanding{PlayerID: pid})
	}

	switch config.Mode {
	case "single_elimination":
		t.buildSingleElimBracket()
	case "double_elimination":
		t.buildDoubleElimBracket()
	default: // "series"
		t.buildSeriesSchedule()
	}
	return t
}

// ---------------------------------------------------------------------------
// Build helpers
// ---------------------------------------------------------------------------

func ceilLog2(n int) int {
	if n <= 1 {
		return 0
	}
	return int(math.Ceil(math.Log2(float64(n))))
}

func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// buildSingleElimBracket initialises the first-round matchups with byes.
func (t *TournamentState) buildSingleElimBracket() {
	players := t.Config.PlayerIDs
	n := len(players)
	if n <= 1 {
		t.Completed = true
		return
	}

	// Number of slots in the bracket (next power of two ≥ n).
	slots := 1
	for slots < n {
		slots <<= 1
	}
	numByes := slots - n

	// Round 1: pair players; top seeds get byes.
	round1 := BracketRound{}
	byesAssigned := 0
	hi := n - 1
	lo := 0
	for lo < hi {
		m := Matchup{Player1: players[lo]}
		if byesAssigned < numByes {
			// Player lo gets a bye.
			m.Bye = true
			m.Player2 = -1
			m.Winner = players[lo]
			m.Played = true
			byesAssigned++
			lo++
		} else {
			m.Player2 = players[hi]
			hi--
			lo++
		}
		round1.Matchups = append(round1.Matchups, m)
	}
	// If odd remaining after pairing (shouldn't happen with the above logic but guard).
	if lo == hi {
		round1.Matchups = append(round1.Matchups, Matchup{
			Player1: players[lo],
			Bye:     true,
			Player2: -1,
			Winner:  players[lo],
			Played:  true,
		})
	}

	t.Bracket = []BracketRound{round1}

	// Pre-allocate subsequent rounds as empty.
	totalRounds := ceilLog2(n)
	for r := 1; r < totalRounds; r++ {
		t.Bracket = append(t.Bracket, BracketRound{})
	}
}

// buildDoubleElimBracket initialises winners and losers brackets.
func (t *TournamentState) buildDoubleElimBracket() {
	// Winners bracket is the same as single elim round 1.
	t.buildSingleElimBracket()
	// Losers bracket starts empty; rounds are added dynamically.
	t.losersRounds = []BracketRound{}
	t.losersRound = 0
}

// buildSeriesSchedule creates a round-robin-style schedule.
func (t *TournamentState) buildSeriesSchedule() {
	players := t.Config.PlayerIDs
	n := len(players)
	rounds := t.Config.NumRounds
	if rounds <= 0 {
		rounds = 1
	}

	schedule := make([][][2]int, rounds)
	for r := 0; r < rounds; r++ {
		var matchups [][2]int
		for i := 0; i < n-1; i++ {
			for j := i + 1; j < n; j++ {
				matchups = append(matchups, [2]int{players[i], players[j]})
			}
		}
		schedule[r] = matchups
	}
	t.seriesSchedule = schedule
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// NextMatchup returns the next unplayed matchup.
// Returns (player1, player2, isBye, nil) on success.
// Returns (-1, -1, false, nil) if all matchups in the current round are played
// (caller should advance with RecordResult or check IsComplete).
// Returns an error only on unexpected state.
func (t *TournamentState) NextMatchup() (int, int, bool, error) {
	if t.Completed {
		return -1, -1, false, errors.New("tournament is complete")
	}

	switch t.Config.Mode {
	case "single_elimination":
		return t.nextSingleElimMatchup()
	case "double_elimination":
		return t.nextDoubleElimMatchup()
	default:
		return t.nextSeriesMatchup()
	}
}

func (t *TournamentState) nextSeriesMatchup() (int, int, bool, error) {
	if t.CurrentRound >= len(t.seriesSchedule) {
		return -1, -1, false, nil
	}
	for _, pair := range t.seriesSchedule[t.CurrentRound] {
		if !t.isSeriesMatchupPlayed(pair[0], pair[1]) {
			return pair[0], pair[1], false, nil
		}
	}
	return -1, -1, false, nil
}

// isSeriesMatchupPlayed reports whether (p1,p2) has been recorded in the current series round.
func (t *TournamentState) isSeriesMatchupPlayed(p1, p2 int) bool {
	if t.CurrentRound >= len(t.Bracket) {
		return false
	}
	for _, m := range t.Bracket[t.CurrentRound].Matchups {
		if (m.Player1 == p1 && m.Player2 == p2) || (m.Player1 == p2 && m.Player2 == p1) {
			return m.Played
		}
	}
	return false
}

func (t *TournamentState) nextSingleElimMatchup() (int, int, bool, error) {
	if t.CurrentRound >= len(t.Bracket) {
		return -1, -1, false, nil
	}
	round := &t.Bracket[t.CurrentRound]
	for i := range round.Matchups {
		m := &round.Matchups[i]
		if !m.Played {
			return m.Player1, m.Player2, m.Bye, nil
		}
	}
	return -1, -1, false, nil
}

func (t *TournamentState) nextDoubleElimMatchup() (int, int, bool, error) {
	// Check winners bracket first.
	p1, p2, bye, err := t.nextSingleElimMatchup()
	if err != nil {
		return p1, p2, bye, err
	}
	if p1 != -1 {
		return p1, p2, bye, nil
	}
	// Then losers bracket.
	if t.losersRound < len(t.losersRounds) {
		r := &t.losersRounds[t.losersRound]
		for i := range r.Matchups {
			m := &r.Matchups[i]
			if !m.Played {
				return m.Player1, m.Player2, m.Bye, nil
			}
		}
	}
	return -1, -1, false, nil
}

// RecordResult records the outcome of a matchup.
// scores[i] is the game score for player i (lower is better in Cambia).
// For elimination, winner = player with lower score.
func (t *TournamentState) RecordResult(player1, player2 int, scores []int) error {
	if t.Completed {
		return errors.New("tournament is complete")
	}
	switch t.Config.Mode {
	case "single_elimination":
		return t.recordSingleElim(player1, player2, scores, false)
	case "double_elimination":
		return t.recordDoubleElim(player1, player2, scores)
	default:
		return t.recordSeries(player1, player2, scores)
	}
}

// determineWinner returns the winner based on scores (lower score wins in Cambia).
func determineWinner(p1, p2 int, scores []int) int {
	if len(scores) < 2 {
		return p1
	}
	if scores[0] <= scores[1] {
		return p1
	}
	return p2
}

func (t *TournamentState) standingFor(pid int) *PlayerStanding {
	for i := range t.Standings {
		if t.Standings[i].PlayerID == pid {
			return &t.Standings[i]
		}
	}
	return nil
}

func (t *TournamentState) recordSeries(p1, p2 int, scores []int) error {
	winner := determineWinner(p1, p2, scores)
	loser := p2
	if winner == p2 {
		loser = p1
	}

	s1 := t.standingFor(p1)
	s2 := t.standingFor(p2)
	if s1 != nil && len(scores) > 0 {
		s1.Score += scores[0]
		if t.Config.ScoringMethod == "lowest_cumulative" {
			// lower is better; no special inversion needed, GetStandings sorts appropriately
		}
	}
	if s2 != nil && len(scores) > 1 {
		s2.Score += scores[1]
	}
	ws := t.standingFor(winner)
	ls := t.standingFor(loser)
	if ws != nil {
		ws.Wins++
	}
	if ls != nil {
		ls.Losses++
	}

	// Record in bracket structure.
	for t.CurrentRound >= len(t.Bracket) {
		t.Bracket = append(t.Bracket, BracketRound{})
	}
	t.Bracket[t.CurrentRound].Matchups = append(t.Bracket[t.CurrentRound].Matchups, Matchup{
		Player1: p1,
		Player2: p2,
		Winner:  winner,
		Scores:  scores,
		Played:  true,
	})

	// Check if current round is complete.
	if t.isSeriesRoundComplete(t.CurrentRound) {
		t.CurrentRound++
		if t.CurrentRound >= t.Config.NumRounds {
			t.Completed = true
		}
	}
	return nil
}

func (t *TournamentState) isSeriesRoundComplete(round int) bool {
	if round >= len(t.seriesSchedule) {
		return true
	}
	expected := len(t.seriesSchedule[round])
	if round >= len(t.Bracket) {
		return false
	}
	played := 0
	for _, m := range t.Bracket[round].Matchups {
		if m.Played {
			played++
		}
	}
	return played >= expected
}

func (t *TournamentState) recordSingleElim(p1, p2 int, scores []int, inLosers bool) error {
	bracket := &t.Bracket
	currentRound := t.CurrentRound
	if inLosers {
		bracket = &t.losersRounds
		currentRound = t.losersRound
	}

	if currentRound >= len(*bracket) {
		return errors.New("no active round in bracket")
	}
	round := &(*bracket)[currentRound]

	// Find the matchup.
	idx := -1
	for i := range round.Matchups {
		m := &round.Matchups[i]
		if m.Played {
			continue
		}
		if (m.Player1 == p1 && m.Player2 == p2) || (m.Player1 == p2 && m.Player2 == p1) {
			idx = i
			break
		}
	}
	if idx < 0 {
		return errors.New("matchup not found in current round")
	}

	m := &round.Matchups[idx]
	winner := determineWinner(m.Player1, m.Player2, scores)
	if m.Bye {
		winner = m.Player1
	}
	m.Winner = winner
	m.Scores = scores
	m.Played = true

	loser := m.Player2
	if winner == m.Player2 {
		loser = m.Player1
	}

	ws := t.standingFor(winner)
	ls := t.standingFor(loser)
	if ws != nil {
		ws.Wins++
	}
	if ls != nil {
		ls.Losses++
		if !inLosers {
			// In single_elimination, one loss = eliminated.
			ls.Eliminated = true
		}
	}

	// Check if round is complete; if so, build next round.
	if t.isRoundComplete(round) {
		if !inLosers {
			t.advanceSingleElimRound()
		}
	}
	return nil
}

func (t *TournamentState) isRoundComplete(round *BracketRound) bool {
	for i := range round.Matchups {
		if !round.Matchups[i].Played {
			return false
		}
	}
	return true
}

func (t *TournamentState) advanceSingleElimRound() {
	if t.CurrentRound >= len(t.Bracket) {
		return
	}
	round := &t.Bracket[t.CurrentRound]

	// Collect winners.
	var winners []int
	for _, m := range round.Matchups {
		if m.Winner >= 0 {
			winners = append(winners, m.Winner)
		}
	}

	t.CurrentRound++

	if len(winners) <= 1 {
		// Tournament over.
		t.Completed = true
		return
	}

	if t.CurrentRound >= len(t.Bracket) {
		t.Bracket = append(t.Bracket, BracketRound{})
	}

	nextRound := &t.Bracket[t.CurrentRound]
	for i := 0; i+1 < len(winners); i += 2 {
		nextRound.Matchups = append(nextRound.Matchups, Matchup{
			Player1: winners[i],
			Player2: winners[i+1],
			Winner:  -1,
		})
	}
	// Odd winner gets a bye.
	if len(winners)%2 == 1 {
		w := winners[len(winners)-1]
		nextRound.Matchups = append(nextRound.Matchups, Matchup{
			Player1: w,
			Player2: -1,
			Bye:     true,
			Winner:  w,
			Played:  true,
		})
	}
}

func (t *TournamentState) recordDoubleElim(p1, p2 int, scores []int) error {
	// Try winners bracket first.
	if t.CurrentRound < len(t.Bracket) {
		round := &t.Bracket[t.CurrentRound]
		for i := range round.Matchups {
			m := &round.Matchups[i]
			if !m.Played &&
				((m.Player1 == p1 && m.Player2 == p2) || (m.Player1 == p2 && m.Player2 == p1)) {

				winner := determineWinner(m.Player1, m.Player2, scores)
				loser := m.Player2
				if winner == m.Player2 {
					loser = m.Player1
				}
				m.Winner = winner
				m.Scores = scores
				m.Played = true

				ws := t.standingFor(winner)
				ls := t.standingFor(loser)
				if ws != nil {
					ws.Wins++
				}
				if ls != nil {
					ls.Losses++
					// Drop to losers bracket (not yet eliminated).
					t.dropToLosers(loser)
				}

				if t.isRoundComplete(round) {
					t.advanceSingleElimRound()
					t.advanceDoubleElimLosers()
				}
				return nil
			}
		}
	}

	// Try losers bracket.
	if t.losersRound < len(t.losersRounds) {
		r := &t.losersRounds[t.losersRound]
		for i := range r.Matchups {
			m := &r.Matchups[i]
			if !m.Played &&
				((m.Player1 == p1 && m.Player2 == p2) || (m.Player1 == p2 && m.Player2 == p1)) {

				winner := determineWinner(m.Player1, m.Player2, scores)
				loser := m.Player2
				if winner == m.Player2 {
					loser = m.Player1
				}
				m.Winner = winner
				m.Scores = scores
				m.Played = true

				ws := t.standingFor(winner)
				ls := t.standingFor(loser)
				if ws != nil {
					ws.Wins++
				}
				if ls != nil {
					ls.Losses++
					ls.Eliminated = true // second loss
				}

				if t.isRoundComplete(r) {
					t.losersRound++
					t.advanceDoubleElimLosers()
				}
				return nil
			}
		}
	}

	return errors.New("matchup not found")
}

func (t *TournamentState) dropToLosers(pid int) {
	// Add player to next available losers round.
	for len(t.losersRounds) <= t.losersRound {
		t.losersRounds = append(t.losersRounds, BracketRound{})
	}
	// Pair with the first unpaired loser in the same losers round.
	lr := &t.losersRounds[t.losersRound]
	for i := range lr.Matchups {
		if lr.Matchups[i].Player2 < 0 && !lr.Matchups[i].Played {
			lr.Matchups[i].Player2 = pid
			return
		}
	}
	// No open slot; start new matchup.
	lr.Matchups = append(lr.Matchups, Matchup{
		Player1: pid,
		Player2: -1,
		Winner:  -1,
	})
}

func (t *TournamentState) advanceDoubleElimLosers() {
	// Pair remaining losers-bracket survivors into next round.
	if t.losersRound >= len(t.losersRounds) {
		return
	}
	r := &t.losersRounds[t.losersRound]
	// Check if all current matchups are complete.
	for i := range r.Matchups {
		if !r.Matchups[i].Played && r.Matchups[i].Player2 >= 0 {
			return // still matchups to play
		}
	}
	// Collect winners from current losers round.
	var survivors []int
	for _, m := range r.Matchups {
		if m.Winner >= 0 {
			survivors = append(survivors, m.Winner)
		}
	}
	if len(survivors) <= 1 {
		// Check if grand final needed.
		t.checkDoubleElimComplete(survivors)
		return
	}
	t.losersRound++
	for len(t.losersRounds) <= t.losersRound {
		t.losersRounds = append(t.losersRounds, BracketRound{})
	}
	nr := &t.losersRounds[t.losersRound]
	for i := 0; i+1 < len(survivors); i += 2 {
		nr.Matchups = append(nr.Matchups, Matchup{
			Player1: survivors[i],
			Player2: survivors[i+1],
			Winner:  -1,
		})
	}
}

func (t *TournamentState) checkDoubleElimComplete(losersSurvivors []int) {
	// Check if winners bracket is done too.
	if t.CurrentRound >= len(t.Bracket) || len(t.Bracket[t.CurrentRound].Matchups) == 0 {
		t.Completed = true
	}
}

// GetStandings returns standings sorted by the configured scoring method.
func (t *TournamentState) GetStandings() []PlayerStanding {
	out := make([]PlayerStanding, len(t.Standings))
	copy(out, t.Standings)

	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		// Eliminated players go to the bottom.
		if a.Eliminated != b.Eliminated {
			return !a.Eliminated
		}
		switch t.Config.ScoringMethod {
		case "lowest_cumulative":
			if a.Score != b.Score {
				return a.Score < b.Score // lower is better
			}
		case "cumulative_score":
			if a.Score != b.Score {
				return a.Score > b.Score // higher is better
			}
		default: // "wins"
			if a.Wins != b.Wins {
				return a.Wins > b.Wins
			}
		}
		return a.PlayerID < b.PlayerID
	})
	return out
}

// IsComplete returns true when the tournament has finished.
func (t *TournamentState) IsComplete() bool {
	return t.Completed
}
