// Command subsidysim measures raw-score / placement statistics for Cambia
// games under a simple threshold-heuristic policy, for the cambia-1010
// aggression-subsidy derivation. Data collection only: this program computes
// no subsidy recommendation, it only emits measured distributions.
//
// Two legs:
//
//   - Leg 1 (symmetric, N=4): all four seats use the same theta. Sweeps
//     theta in {6,9,12}. Written to results_4p.json.
//
//   - Leg 2 (best-response, N in {2,4}): the field seats are fixed at
//     theta=9 and one deviant seat sweeps theta_dev in {5..14}, answering
//     whether a marginally looser (or tighter) call threshold is profitable
//     against a fixed field. The deviant's physical seat position rotates
//     across games (game i -> seat i mod N) so first-actor advantage does
//     not confound the sweep. Written to results_br.json.
//
// Policy per seat (identical shape for every seat; only theta differs):
//   - At start of turn: call Cambia if (sum of known own card values +
//     6.22 * count of unknown own cards) <= theta; otherwise draw from the
//     stockpile.
//   - Post-draw: replace the highest-value KNOWN own card if the drawn
//     card's value is lower than it; otherwise discard. When discarding a
//     drawn card that has the peek-own ability (7/8) drawn from the
//     stockpile, use the ability to reveal an unknown own card. All other
//     abilities (peek-other, blind-swap, king-look) are skipped (plain
//     discard, ability never triggered).
//   - Snap decisions: always pass (this policy never attempts a snap),
//     so hand sizes stay fixed at CardsPerPlayer for the whole game and no
//     penalty draws ever occur.
//
// Because no ability besides peek-own is ever invoked and no snap is ever
// attempted, no player's hand is ever touched by another player: each
// seat's knowledge state is fully local and is tracked here (not in the
// engine) as a per-slot known/unknown flag plus last-known value.
//
// Placement tie rule (RULES.md §6): the Cambia caller wins any tie they are
// part of outright (ranked strictly above the other tied player(s)); ties
// not involving the caller are broken by seat index for a well-defined
// strict 1..N placement order (needed to bucket "E/SD raw score by
// placement" and "adjacent-placement gap" below). Whether an actual score
// tie occurred is reported separately (p_exact_tie_for_first / gap<=k
// mass), so the tie-break used for bucketing does not hide how often ties
// really happen.
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	engine "github.com/jason-s-yu/cambia/engine"
)

const (
	eCard           = 6.22 // approx E[card value] over the standard 54-card deck
	maxGameTurns    = 20000
	maxLowLevelStep = 500000 // hard circuit breaker against runaway games (bug guard)
)

// ---------------------------------------------------------------------------
// Knowledge tracking (per game, per seat)
// ---------------------------------------------------------------------------

type knowledge struct {
	known [engine.MaxPlayers][engine.MaxHandSize]bool
	val   [engine.MaxPlayers][engine.MaxHandSize]int8
}

func (k *knowledge) knownSumUnknown(seat uint8, handLen uint8) (sum int, unknown int) {
	for i := uint8(0); i < handLen; i++ {
		if k.known[seat][i] {
			sum += int(k.val[seat][i])
		} else {
			unknown++
		}
	}
	return
}

func (k *knowledge) maxKnownSlot(seat uint8, handLen uint8) (idx uint8, v int8, ok bool) {
	for i := uint8(0); i < handLen; i++ {
		if k.known[seat][i] {
			if !ok || k.val[seat][i] > v {
				v = k.val[seat][i]
				idx = i
				ok = true
			}
		}
	}
	return
}

func (k *knowledge) firstUnknownSlot(seat uint8, handLen uint8) uint8 {
	for i := uint8(0); i < handLen; i++ {
		if !k.known[seat][i] {
			return i
		}
	}
	return 0
}

// ---------------------------------------------------------------------------
// Game simulation
// ---------------------------------------------------------------------------

// gameResult holds the per-game measurements collected for aggregation.
// scores/placements are length numPlayers, indexed by physical seat.
type gameResult struct {
	numPlayers  int
	callerSeat  int // -1 if no caller (should not happen given maxGameTurns headroom)
	deviantSeat int // -1 when not applicable (leg 1); physical seat of the deviant (leg 2)
	scores      []int
	placements  []int // 1-indexed, strict order (see tie-break note above)
	turns       int
	tieForFirst bool // 2+ players share the minimum raw score
}

func baseRules() engine.HouseRules {
	r := engine.DefaultHouseRules()
	// DefaultHouseRules() sets MaxGameTurns=46, tuned for 2P; both legs here
	// (N=2 and N=4) need enough headroom that the threshold policy always
	// converges to a natural Cambia call rather than being cut off by the
	// turn cap (NumPlayers is overridden per-call in simulateGame).
	r.MaxGameTurns = maxGameTurns
	return r
}

// simulateGame plays one full game under the threshold heuristic policy,
// one theta per physical seat (len(thetas) determines player count), and
// returns the measured result. deviantSeat is stamped onto the result
// verbatim for the caller's convenience (leg 2 bookkeeping); pass -1 when
// not applicable.
func simulateGame(seed uint64, thetas []float64, deviantSeat int, rules engine.HouseRules) gameResult {
	n := len(thetas)
	rules.NumPlayers = uint8(n)

	g := engine.NewGame(seed, rules)
	g.Deal()

	var kn knowledge
	for p := 0; p < n; p++ {
		ps := uint8(p)
		cnt := g.Players[ps].InitialPeekCount
		for i := uint8(0); i < cnt; i++ {
			idx := g.Players[ps].InitialPeek[i]
			kn.known[ps][idx] = true
			kn.val[ps][idx] = int8(g.Players[ps].Hand[idx].Value())
		}
	}

	steps := 0
	for !g.IsTerminal() {
		steps++
		if steps > maxLowLevelStep {
			panic(fmt.Sprintf("subsidysim: runaway game (seed=%d n=%d steps=%d)", seed, n, steps))
		}

		switch g.DecisionCtx() {
		case engine.CtxStartTurn:
			acting := g.CurrentPlayer
			handLen := g.Players[acting].HandLen
			sum, unknown := kn.knownSumUnknown(acting, handLen)
			est := float64(sum) + eCard*float64(unknown)
			theta := thetas[acting]
			if g.CambiaCaller == -1 && est <= theta {
				must(g.ApplyNPlayerAction(engine.NPlayerActionCallCambia))
			} else {
				must(g.ApplyNPlayerAction(engine.NPlayerActionDrawStockpile))
			}

		case engine.CtxPostDraw:
			acting := g.Pending.PlayerID
			handLen := g.Players[acting].HandLen
			drawn := engine.Card(g.Pending.Data[0])
			idx, maxVal, ok := kn.maxKnownSlot(acting, handLen)
			if ok && int8(drawn.Value()) < maxVal {
				must(g.ApplyNPlayerAction(engine.NPlayerEncodeReplace(idx)))
				kn.known[acting][idx] = true
				kn.val[acting][idx] = int8(drawn.Value())
			} else if drawn.Ability() == engine.AbilityPeekOwn {
				must(g.ApplyNPlayerAction(engine.NPlayerActionDiscardWithAbility))
			} else {
				must(g.ApplyNPlayerAction(engine.NPlayerActionDiscardNoAbility))
			}

		case engine.CtxAbilitySelect:
			if g.Pending.Type != engine.PendingPeekOwn {
				panic(fmt.Sprintf("subsidysim: unexpected pending ability type %d (policy only ever triggers peek-own)", g.Pending.Type))
			}
			acting := g.Pending.PlayerID
			handLen := g.Players[acting].HandLen
			slot := kn.firstUnknownSlot(acting, handLen)
			kn.known[acting][slot] = true
			kn.val[acting][slot] = int8(g.Players[acting].Hand[slot].Value())
			must(g.ApplyNPlayerAction(engine.NPlayerEncodePeekOwn(slot)))

		case engine.CtxSnapDecision:
			must(g.ApplyNPlayerAction(engine.NPlayerActionPassSnap))

		case engine.CtxSnapMove:
			panic("subsidysim: unreachable CtxSnapMove under always-pass snap policy")

		case engine.CtxTerminal:
			// Loop condition already exits; nothing to do.
		}
	}

	scores := make([]int, n)
	minScore := math.MaxInt32
	for p := 0; p < n; p++ {
		ps := uint8(p)
		s := 0
		hl := g.Players[ps].HandLen
		for i := uint8(0); i < hl; i++ {
			s += int(g.Players[ps].Hand[i].Value())
		}
		scores[p] = s
		if s < minScore {
			minScore = s
		}
	}
	tieCount := 0
	for _, s := range scores {
		if s == minScore {
			tieCount++
		}
	}

	res := gameResult{
		numPlayers:  n,
		callerSeat:  int(g.CambiaCaller),
		deviantSeat: deviantSeat,
		scores:      scores,
		turns:       int(g.TurnNumber),
		tieForFirst: tieCount >= 2,
	}
	res.placements = computePlacements(scores, res.callerSeat)
	return res
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

// computePlacements assigns a strict 1..N placement to each seat: ascending
// by raw score, with the Cambia caller (if part of a tied group) ranked
// first among the tied seats (RULES.md §6), and any remaining tie among
// non-caller seats broken deterministically by seat index.
func computePlacements(scores []int, callerSeat int) []int {
	n := len(scores)
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(i, j int) bool {
		si, sj := scores[order[i]], scores[order[j]]
		if si != sj {
			return si < sj
		}
		if order[i] == callerSeat {
			return true
		}
		if order[j] == callerSeat {
			return false
		}
		return order[i] < order[j]
	})
	placements := make([]int, n)
	for idx, seat := range order {
		placements[seat] = idx + 1
	}
	return placements
}

// ---------------------------------------------------------------------------
// Shared sanity assertions (used by both legs' smoke tests)
// ---------------------------------------------------------------------------

func assertGameSane(res gameResult, seed uint64, label string) error {
	n := res.numPlayers
	seen := make([]bool, n+1)
	sum := 0
	for _, pl := range res.placements {
		if pl < 1 || pl > n || seen[pl] {
			return fmt.Errorf("%s seed=%d: invalid placement set %v", label, seed, res.placements)
		}
		seen[pl] = true
		sum += pl
	}
	want := n * (n + 1) / 2
	if sum != want {
		return fmt.Errorf("%s seed=%d: placements do not sum to %d: %v", label, seed, want, res.placements)
	}

	if res.callerSeat >= 0 {
		minScore := res.scores[0]
		for _, s := range res.scores {
			if s < minScore {
				minScore = s
			}
		}
		if res.scores[res.callerSeat] == minScore && res.placements[res.callerSeat] != 1 {
			return fmt.Errorf("%s seed=%d: caller %d tied/won for lowest (score %d, scores %v) but placement=%d, want 1",
				label, seed, res.callerSeat, res.scores[res.callerSeat], res.scores, res.placements[res.callerSeat])
		}
		if res.scores[res.callerSeat] != minScore && res.placements[res.callerSeat] == 1 {
			return fmt.Errorf("%s seed=%d: caller %d does not have lowest hand (score %d, min %d) but was placed 1st",
				label, seed, res.callerSeat, res.scores[res.callerSeat], minScore)
		}
	}

	for _, s := range res.scores {
		if s < -8 || s > 160 {
			return fmt.Errorf("%s seed=%d: score out of sane range: %v", label, seed, res.scores)
		}
	}
	if res.turns <= 0 {
		return fmt.Errorf("%s seed=%d: non-positive turns %d", label, seed, res.turns)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Aggregation helpers (shared)
// ---------------------------------------------------------------------------

type meanSD struct {
	Mean float64 `json:"mean"`
	SD   float64 `json:"sd"`
	N    int     `json:"n"`
}

func computeMeanSD(xs []float64) meanSD {
	n := len(xs)
	if n == 0 {
		return meanSD{}
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean := sum / float64(n)
	if n < 2 {
		return meanSD{Mean: mean, SD: 0, N: n}
	}
	var ss float64
	for _, x := range xs {
		d := x - mean
		ss += d * d
	}
	sd := math.Sqrt(ss / float64(n-1))
	return meanSD{Mean: mean, SD: sd, N: n}
}

func percentile(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sorted[0]
	}
	rank := p / 100 * float64(n-1)
	lo := int(math.Floor(rank))
	hi := int(math.Ceil(rank))
	if lo == hi {
		return sorted[lo]
	}
	frac := rank - float64(lo)
	return sorted[lo]*(1-frac) + sorted[hi]*frac
}

type gapStats struct {
	P10  float64 `json:"p10"`
	P25  float64 `json:"p25"`
	P50  float64 `json:"p50"`
	P75  float64 `json:"p75"`
	P90  float64 `json:"p90"`
	PLE1 float64 `json:"p_le_1"`
	PLE2 float64 `json:"p_le_2"`
	PLE3 float64 `json:"p_le_3"`
	PLE5 float64 `json:"p_le_5"`
	N    int     `json:"n"`
}

func computeGapStats(gaps []int) gapStats {
	n := len(gaps)
	if n == 0 {
		return gapStats{}
	}
	fs := make([]float64, n)
	le1, le2, le3, le5 := 0, 0, 0, 0
	for i, g := range gaps {
		fs[i] = float64(g)
		if g <= 1 {
			le1++
		}
		if g <= 2 {
			le2++
		}
		if g <= 3 {
			le3++
		}
		if g <= 5 {
			le5++
		}
	}
	sort.Float64s(fs)
	return gapStats{
		P10:  percentile(fs, 10),
		P25:  percentile(fs, 25),
		P50:  percentile(fs, 50),
		P75:  percentile(fs, 75),
		P90:  percentile(fs, 90),
		PLE1: float64(le1) / float64(n),
		PLE2: float64(le2) / float64(n),
		PLE3: float64(le3) / float64(n),
		PLE5: float64(le5) / float64(n),
		N:    n,
	}
}

func policyDescription() string {
	return "draw stockpile; replace highest-value known own card if drawn card is lower, else discard " +
		"(peek-own ability used on discard when drawn card is 7/8, all other abilities skipped, never snap); " +
		"call Cambia at start of turn when sum(known own values) + 6.22*count(unknown own cards) <= theta (per-seat theta)"
}

func resolveOutPath(filename string) string {
	if _, err := os.Stat(filepath.Join("cmd", "subsidysim")); err == nil {
		return filepath.Join("cmd", "subsidysim", filename)
	}
	return filename
}

func writeJSON(path string, v interface{}) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "marshal error (%s): %v\n", path, err)
		os.Exit(1)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "write error (%s): %v\n", path, err)
		os.Exit(1)
	}
	abs, _ := filepath.Abs(path)
	fmt.Printf("wrote %s\n", abs)
}

// ===========================================================================
// Leg 1: symmetric N=4 theta sweep
// ===========================================================================

const leg1N = 4

func leg1Thetas(theta float64) []float64 {
	return []float64{theta, theta, theta, theta}
}

func leg1MainSeed(cellIdx, gameIdx int) uint64 {
	return uint64(1_000_000*(cellIdx+1) + gameIdx + 1)
}

func leg1SmokeSeed(cellIdx, gameIdx int) uint64 {
	// Disjoint from the leg-1 main-run seed space (offset by 900_000_000).
	return uint64(900_000_000 + 1_000_000*(cellIdx+1) + gameIdx + 1)
}

func runLeg1SmokeTest(thetas []float64, rules engine.HouseRules) error {
	const smokeGames = 100
	for cellIdx, theta := range thetas {
		for i := 0; i < smokeGames; i++ {
			seed := leg1SmokeSeed(cellIdx, i)
			res := simulateGame(seed, leg1Thetas(theta), -1, rules)
			if err := assertGameSane(res, seed, fmt.Sprintf("leg1 theta=%v", theta)); err != nil {
				return err
			}
		}
	}
	return nil
}

type leg1CellOutput struct {
	Theta               float64             `json:"theta"`
	Games               int                 `json:"games"`
	PCall               float64             `json:"p_call"`
	CallerGames         int                 `json:"caller_games"`
	CallerPlacementDist map[string]float64  `json:"caller_placement_dist"`
	ScoreByPlacement    map[string]meanSD   `json:"score_by_placement"`
	CallerRawScore      meanSD              `json:"caller_raw_score"`
	NonCallerRawScore   meanSD              `json:"noncaller_raw_score"`
	AdjacentGapOverall  gapStats            `json:"adjacent_gap_overall"`
	AdjacentGapByPair   map[string]gapStats `json:"adjacent_gap_by_pair"`
	PExactTieForFirst   float64             `json:"p_exact_tie_for_first"`
	MeanTurns           float64             `json:"mean_turns"`
	SDTurns             float64             `json:"sd_turns"`
	FractionHitMaxTurns float64             `json:"fraction_hit_max_game_turns"`
}

func aggregateLeg1Cell(theta float64, results []gameResult) leg1CellOutput {
	n := len(results)
	out := leg1CellOutput{
		Theta:               theta,
		Games:               n,
		CallerPlacementDist: map[string]float64{"1": 0, "2": 0, "3": 0, "4": 0},
		ScoreByPlacement:    map[string]meanSD{},
		AdjacentGapByPair:   map[string]gapStats{},
	}

	callerGames := 0
	callerPlacementCount := map[int]int{}
	scoresByPlacement := map[int][]float64{1: {}, 2: {}, 3: {}, 4: {}}
	var callerScores, nonCallerScores []float64
	var turns []float64
	gapAll := []int{}
	gapByPair := map[string][]int{"1_2": {}, "2_3": {}, "3_4": {}}
	tieCount := 0
	cappedCount := 0

	for _, r := range results {
		turns = append(turns, float64(r.turns))
		if r.turns >= maxGameTurns {
			cappedCount++
		}
		if r.tieForFirst {
			tieCount++
		}

		var seatByPlacement [leg1N + 1]int
		for seat, pl := range r.placements {
			seatByPlacement[pl] = seat
			scoresByPlacement[pl] = append(scoresByPlacement[pl], float64(r.scores[seat]))
		}

		if r.callerSeat >= 0 {
			callerGames++
			callerPl := r.placements[r.callerSeat]
			callerPlacementCount[callerPl]++
			callerScores = append(callerScores, float64(r.scores[r.callerSeat]))
			for seat := 0; seat < leg1N; seat++ {
				if seat != r.callerSeat {
					nonCallerScores = append(nonCallerScores, float64(r.scores[seat]))
				}
			}
		}

		for k := 1; k <= 3; k++ {
			gap := r.scores[seatByPlacement[k+1]] - r.scores[seatByPlacement[k]]
			gapAll = append(gapAll, gap)
			pairKey := fmt.Sprintf("%d_%d", k, k+1)
			gapByPair[pairKey] = append(gapByPair[pairKey], gap)
		}
	}

	out.CallerGames = callerGames
	if n > 0 {
		out.PCall = float64(callerGames) / float64(n)
		out.PExactTieForFirst = float64(tieCount) / float64(n)
		out.FractionHitMaxTurns = float64(cappedCount) / float64(n)
	}
	if callerGames > 0 {
		for pl := 1; pl <= 4; pl++ {
			out.CallerPlacementDist[fmt.Sprintf("%d", pl)] = float64(callerPlacementCount[pl]) / float64(callerGames)
		}
	}
	for pl := 1; pl <= 4; pl++ {
		out.ScoreByPlacement[fmt.Sprintf("%d", pl)] = computeMeanSD(scoresByPlacement[pl])
	}
	out.CallerRawScore = computeMeanSD(callerScores)
	out.NonCallerRawScore = computeMeanSD(nonCallerScores)
	out.AdjacentGapOverall = computeGapStats(gapAll)
	for k, v := range gapByPair {
		out.AdjacentGapByPair[k] = computeGapStats(v)
	}
	tMS := computeMeanSD(turns)
	out.MeanTurns = tMS.Mean
	out.SDTurns = tMS.SD

	return out
}

type leg1MetaOutput struct {
	GeneratedAt  string  `json:"generated_at"`
	Ticket       string  `json:"ticket"`
	Leg          string  `json:"leg"`
	EngineModule string  `json:"engine_module"`
	NumPlayers   int     `json:"num_players"`
	ECardApprox  float64 `json:"e_card_approx"`
	MaxGameTurns int     `json:"max_game_turns"`
	SeedScheme   string  `json:"seed_scheme"`
	SmokeGames   int     `json:"smoke_games_per_cell"`
	Policy       string  `json:"policy"`
	RunSeconds   float64 `json:"run_seconds"`
}

type leg1Output struct {
	Meta  leg1MetaOutput   `json:"meta"`
	Cells []leg1CellOutput `json:"cells"`
}

func runLeg1(numGames int, rules engine.HouseRules) {
	thetas := []float64{6, 9, 12}
	fmt.Printf("=== leg 1 (symmetric, N=4): thetas=%v, games/cell=%d ===\n", thetas, numGames)

	smokeStart := time.Now()
	if err := runLeg1SmokeTest(thetas, rules); err != nil {
		fmt.Fprintf(os.Stderr, "LEG1 SMOKE TEST FAILED: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("leg1 smoke test PASSED in %v\n", time.Since(smokeStart))

	start := time.Now()
	cellResults := make([][]gameResult, len(thetas))
	var wg sync.WaitGroup
	for cellIdx, theta := range thetas {
		wg.Add(1)
		go func(cellIdx int, theta float64) {
			defer wg.Done()
			results := make([]gameResult, numGames)
			for i := 0; i < numGames; i++ {
				seed := leg1MainSeed(cellIdx, i)
				results[i] = simulateGame(seed, leg1Thetas(theta), -1, rules)
			}
			cellResults[cellIdx] = results
		}(cellIdx, theta)
	}
	wg.Wait()
	elapsed := time.Since(start)
	fmt.Printf("leg1 main run completed in %v\n", elapsed)

	out := leg1Output{
		Meta: leg1MetaOutput{
			GeneratedAt:  time.Now().UTC().Format(time.RFC3339),
			Ticket:       "cambia-1010",
			Leg:          "4-player symmetric",
			EngineModule: "github.com/jason-s-yu/cambia/engine",
			NumPlayers:   leg1N,
			ECardApprox:  eCard,
			MaxGameTurns: maxGameTurns,
			SeedScheme:   "seed = 1_000_000*(cellIndex+1) + gameIndex + 1, cellIndex in order of theta list [6,9,12]",
			SmokeGames:   100,
			Policy:       policyDescription(),
			RunSeconds:   elapsed.Seconds(),
		},
	}
	for cellIdx, theta := range thetas {
		out.Cells = append(out.Cells, aggregateLeg1Cell(theta, cellResults[cellIdx]))
	}

	writeJSON(resolveOutPath("results_4p.json"), out)

	for _, c := range out.Cells {
		fmt.Printf("leg1 theta=%v games=%d P(call)=%.4f P(caller 1st)=%.4f median_gap=%.3f mean_turns=%.2f P(tie1st)=%.4f\n",
			c.Theta, c.Games, c.PCall, c.CallerPlacementDist["1"], c.AdjacentGapOverall.P50, c.MeanTurns, c.PExactTieForFirst)
	}
}

// ===========================================================================
// Leg 2: best-response asymmetric sweep (N in {2,4}, one deviant seat)
// ===========================================================================

var brThetaDevs = []float64{5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

const brThetaField = 9.0

func brMainSeed(n int, thetaIdx int, gameIdx int) uint64 {
	return uint64(10_000_000*n + 100_000*thetaIdx + gameIdx + 1)
}

func brSmokeSeed(n int, thetaIdx int, gameIdx int) uint64 {
	// Fixed offset well above the largest possible main-run seed
	// (10_000_000*4 + 100_000*9 + 100_000 = 41_000_000) to stay disjoint.
	return uint64(90_000_000 + 10_000_000*n + 100_000*thetaIdx + gameIdx + 1)
}

func brThetasForGame(n int, deviantSeat int, thetaDev float64) []float64 {
	thetas := make([]float64, n)
	for s := 0; s < n; s++ {
		if s == deviantSeat {
			thetas[s] = thetaDev
		} else {
			thetas[s] = brThetaField
		}
	}
	return thetas
}

func runLeg2SmokeTest(numPlayersList []int, rules engine.HouseRules) error {
	const smokeGames = 100
	for _, n := range numPlayersList {
		for thetaIdx, thetaDev := range brThetaDevs {
			for i := 0; i < smokeGames; i++ {
				seed := brSmokeSeed(n, thetaIdx, i)
				deviantSeat := i % n
				res := simulateGame(seed, brThetasForGame(n, deviantSeat, thetaDev), deviantSeat, rules)
				label := fmt.Sprintf("legBR N=%d thetaDev=%v", n, thetaDev)
				if err := assertGameSane(res, seed, label); err != nil {
					return err
				}
				if res.deviantSeat != deviantSeat {
					return fmt.Errorf("%s seed=%d: deviantSeat not preserved (got %d want %d)", label, seed, res.deviantSeat, deviantSeat)
				}
			}
		}
	}
	return nil
}

type roleStats struct {
	PlacementDist map[string]float64 `json:"placement_dist"`
	RawScore      meanSD             `json:"raw_score"`
}

type brCellOutput struct {
	NumPlayers       int       `json:"num_players"`
	ThetaField       float64   `json:"theta_field"`
	ThetaDev         float64   `json:"theta_dev"`
	Games            int       `json:"games"`
	PAnyCall         float64   `json:"p_any_call"`
	PDeviantIsCaller float64   `json:"p_deviant_is_caller"`
	Deviant          roleStats `json:"deviant"`
	Field            roleStats `json:"field"`
	MeanTurns        float64   `json:"mean_turns"`
	SDTurns          float64   `json:"sd_turns"`
}

func aggregateBRCell(n int, thetaDev float64, results []gameResult) brCellOutput {
	ng := len(results)
	out := brCellOutput{
		NumPlayers: n,
		ThetaField: brThetaField,
		ThetaDev:   thetaDev,
		Games:      ng,
	}

	anyCall := 0
	devIsCaller := 0
	devPlacementCount := map[int]int{}
	fieldPlacementCount := map[int]int{}
	var devScores, fieldScores []float64
	var turns []float64

	for _, r := range results {
		turns = append(turns, float64(r.turns))
		if r.callerSeat >= 0 {
			anyCall++
		}
		if r.deviantSeat >= 0 && r.callerSeat == r.deviantSeat {
			devIsCaller++
		}
		for seat := 0; seat < n; seat++ {
			if seat == r.deviantSeat {
				devScores = append(devScores, float64(r.scores[seat]))
				devPlacementCount[r.placements[seat]]++
			} else {
				fieldScores = append(fieldScores, float64(r.scores[seat]))
				fieldPlacementCount[r.placements[seat]]++
			}
		}
	}

	if ng > 0 {
		out.PAnyCall = float64(anyCall) / float64(ng)
		out.PDeviantIsCaller = float64(devIsCaller) / float64(ng)
	}

	devDist := map[string]float64{}
	fieldDist := map[string]float64{}
	fieldSamples := ng * (n - 1)
	for pl := 1; pl <= n; pl++ {
		key := fmt.Sprintf("%d", pl)
		if ng > 0 {
			devDist[key] = float64(devPlacementCount[pl]) / float64(ng)
		} else {
			devDist[key] = 0
		}
		if fieldSamples > 0 {
			fieldDist[key] = float64(fieldPlacementCount[pl]) / float64(fieldSamples)
		} else {
			fieldDist[key] = 0
		}
	}

	out.Deviant = roleStats{PlacementDist: devDist, RawScore: computeMeanSD(devScores)}
	out.Field = roleStats{PlacementDist: fieldDist, RawScore: computeMeanSD(fieldScores)}
	tms := computeMeanSD(turns)
	out.MeanTurns = tms.Mean
	out.SDTurns = tms.SD

	return out
}

type brMetaOutput struct {
	GeneratedAt    string    `json:"generated_at"`
	Ticket         string    `json:"ticket"`
	Leg            string    `json:"leg"`
	EngineModule   string    `json:"engine_module"`
	NumPlayers     []int     `json:"num_players_swept"`
	ThetaField     float64   `json:"theta_field"`
	ThetaDevSweep  []float64 `json:"theta_dev_sweep"`
	ECardApprox    float64   `json:"e_card_approx"`
	MaxGameTurns   int       `json:"max_game_turns"`
	SeedScheme     string    `json:"seed_scheme"`
	RotationScheme string    `json:"rotation_scheme"`
	SmokeGames     int       `json:"smoke_games_per_cell"`
	Policy         string    `json:"policy"`
	RunSeconds     float64   `json:"run_seconds"`
}

type brOutput struct {
	Meta  brMetaOutput   `json:"meta"`
	Cells []brCellOutput `json:"cells"`
}

func runLeg2(numGames int, rules engine.HouseRules) {
	numPlayersList := []int{2, 4}
	fmt.Printf("=== leg 2 (best-response): N=%v, theta_field=%v, theta_dev=%v, games/cell=%d ===\n",
		numPlayersList, brThetaField, brThetaDevs, numGames)

	smokeStart := time.Now()
	if err := runLeg2SmokeTest(numPlayersList, rules); err != nil {
		fmt.Fprintf(os.Stderr, "LEG2 SMOKE TEST FAILED: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("leg2 smoke test PASSED in %v\n", time.Since(smokeStart))

	start := time.Now()
	type cellKey struct {
		nIdx, thetaIdx int
	}
	cellResults := make(map[cellKey][]gameResult)
	var mu sync.Mutex
	var wg sync.WaitGroup
	for nIdx, n := range numPlayersList {
		for thetaIdx, thetaDev := range brThetaDevs {
			wg.Add(1)
			go func(nIdx, n, thetaIdx int, thetaDev float64) {
				defer wg.Done()
				results := make([]gameResult, numGames)
				for i := 0; i < numGames; i++ {
					seed := brMainSeed(n, thetaIdx, i)
					deviantSeat := i % n
					results[i] = simulateGame(seed, brThetasForGame(n, deviantSeat, thetaDev), deviantSeat, rules)
				}
				mu.Lock()
				cellResults[cellKey{nIdx, thetaIdx}] = results
				mu.Unlock()
			}(nIdx, n, thetaIdx, thetaDev)
		}
	}
	wg.Wait()
	elapsed := time.Since(start)
	fmt.Printf("leg2 main run completed in %v\n", elapsed)

	out := brOutput{
		Meta: brMetaOutput{
			GeneratedAt:    time.Now().UTC().Format(time.RFC3339),
			Ticket:         "cambia-1010",
			Leg:            "best-response asymmetric sweep (2P and 4P)",
			EngineModule:   "github.com/jason-s-yu/cambia/engine",
			NumPlayers:     numPlayersList,
			ThetaField:     brThetaField,
			ThetaDevSweep:  brThetaDevs,
			ECardApprox:    eCard,
			MaxGameTurns:   maxGameTurns,
			SeedScheme:     "seed = 10_000_000*N + 100_000*thetaDevIndex + gameIndex + 1 (thetaDevIndex is the position of theta_dev in [5,6,7,8,9,10,11,12,13,14]); smoke seeds add a fixed +90_000_000 offset, disjoint from main-run and leg-1 seed spaces",
			RotationScheme: "deviant seat for game i (0-indexed within its cell) = i mod N, so the deviant occupies every physical seat (and every turn-order position) equally often within each cell",
			SmokeGames:     100,
			Policy:         policyDescription(),
			RunSeconds:     elapsed.Seconds(),
		},
	}
	for nIdx, n := range numPlayersList {
		for thetaIdx, thetaDev := range brThetaDevs {
			out.Cells = append(out.Cells, aggregateBRCell(n, thetaDev, cellResults[cellKey{nIdx, thetaIdx}]))
		}
	}

	writeJSON(resolveOutPath("results_br.json"), out)

	for _, c := range out.Cells {
		fmt.Printf("leg2 N=%d theta_dev=%v games=%d P(any_call)=%.4f P(dev caller)=%.4f dev_P(1st)=%.4f dev_mean_score=%.3f mean_turns=%.2f\n",
			c.NumPlayers, c.ThetaDev, c.Games, c.PAnyCall, c.PDeviantIsCaller, c.Deviant.PlacementDist["1"], c.Deviant.RawScore.Mean, c.MeanTurns)
	}
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

func envInt(name string, def int) int {
	if v := os.Getenv(name); v != "" {
		var out int
		if _, err := fmt.Sscanf(v, "%d", &out); err == nil {
			return out
		}
	}
	return def
}

func main() {
	rules := baseRules()

	leg1Games := envInt("SUBSIDYSIM_GAMES", 100000)
	runLeg1(leg1Games, rules)

	fmt.Println()
	leg2Games := envInt("SUBSIDYSIM_BR_GAMES", 100000)
	runLeg2(leg2Games, rules)
}
