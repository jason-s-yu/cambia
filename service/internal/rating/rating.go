package rating

import (
	"math"
	"sort"

	"github.com/google/uuid"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

// We'll keep ephemeral track of each user's (mu, phi, sigma).
type glickoState struct {
	mu    float64
	phi   float64
	sigma float64
}

// FinalizeRatings runs a multi-iteration Glicko-2 update on the entire group of players
// based on their final "score" (lower is better in Cambia). This single function is typically
// called once at game end to produce updated rating fields for each player.
//
//  1. We convert final scores into a fraction from 0..1, where 1 is best rank and 0 is worst.
//  2. We call MultiIterationGlicko2 to refine each user's rating across multiple iterations, so
//     that phi and sigma can converge a bit closer than in a single pass.
//
// Note: for a true persistent Glicko-2, we store each user's phi, sigma in the DB, then
// feed them into the next match.
// TODO: Modify returns ephemeral updated ELO only
func FinalizeRatings(players []models.User, scoresMap map[uuid.UUID]int) []models.User {
	// 1) Build a rank-based fraction for each user
	type userScore struct {
		UserID uuid.UUID
		Score  int
	}
	var arr []userScore
	for _, p := range players {
		arr = append(arr, userScore{p.ID, scoresMap[p.ID]})
	}
	sort.Slice(arr, func(i, j int) bool {
		return arr[i].Score < arr[j].Score // ascending
	})

	// We'll assign fractional scores: top rank => 1.0, last => 0.0, ties share fraction
	rankFrac := make(map[uuid.UUID]float64, len(arr))
	i := 0
	for i < len(arr) {
		j := i + 1
		for j < len(arr) && arr[j].Score == arr[i].Score {
			j++
		}
		// players i..j-1 are tied
		// midRank fraction => 1 - (avgRank / (count-1))
		avgRank := float64(i+(j-1)) / 2
		fr := 1.0 - (avgRank / float64(len(arr)-1))
		for k := i; k < j; k++ {
			rankFrac[arr[k].UserID] = fr
		}
		i = j
	}

	// Build slices for Glicko
	scores := make([]float64, len(players))
	userIndex := make(map[uuid.UUID]int)
	for i, p := range players {
		userIndex[p.ID] = i
	}
	for _, p := range players {
		idx := userIndex[p.ID]
		scores[idx] = rankFrac[p.ID]
	}

	return MultiIterationGlicko2(players, scores, 10) // 10 iterations for demonstration
}

// MultiIterationGlicko2 repeatedly applies Glicko2 updates for the given players
// and their 0..1 "scores" for a single game. We treat "opponent" as the average rating
// of the rest. Real Glicko2 might sum over each pairing, but here's a simpler approach.
//
//   - players: slice of user info
//   - scores:  parallel slice of the same length with final fraction for each user
//   - iterations: number of times we re-run the Glicko update to refine phi, sigma
//
// We return the updated players with new Elo in .Elo1v1 (for demonstration).
// In a production system, you'd store updated phi, sigma in your DB for next time.
func MultiIterationGlicko2(players []models.User, scores []float64, iterations int) []models.User {
	states := make([]glickoState, len(players))

	// Initialize from their Elo. In production, you'd load prior phi/sigma from DB.
	for i, u := range players {
		states[i].mu = (float64(u.Elo1v1) - DefaultMu) / GlickoScale
		states[i].phi = DefaultPhi / GlickoScale
		states[i].sigma = 0.06
	}

	for iter := 0; iter < iterations; iter++ {
		// Compute the average rating for "everyone else"
		var total float64
		for i := range states {
			elo := states[i].mu*GlickoScale + DefaultMu
			total += elo
		}
		// Single pass update
		newStates := make([]glickoState, len(players))
		for i := range players {
			oldMu := states[i].mu
			oldPhi := states[i].phi
			oldSigma := states[i].sigma

			myElo := oldMu*GlickoScale + DefaultMu
			opponentElo := (total - myElo) / float64(len(players)-1)

			oppMu := (opponentElo - DefaultMu) / GlickoScale
			oppPhi := DefaultPhi / GlickoScale
			oppSigma := 0.06

			// single-match update
			score := scores[i]
			ns := doGlickoUpdate(oldMu, oldPhi, oldSigma, oppMu, oppPhi, oppSigma, score)
			newStates[i] = ns
		}
		states = newStates
	}

	// After iterations, convert back to Elo
	for i := range players {
		newElo := states[i].mu*GlickoScale + DefaultMu
		players[i].Elo1v1 = int(math.Round(newElo))
	}
	return players
}

// doGlickoUpdate is a helper that updates (mu, phi, sigma) vs an average "opponent" in one match
func doGlickoUpdate(mu, phi, sigma, oppMu, oppPhi, oppSigma, score float64) glickoState {
	gVal := g(oppPhi)
	EVal := E(mu, oppMu, oppPhi)
	v := 1.0 / (gVal * gVal * EVal * (1 - EVal))
	delta := v * gVal * (score - EVal)

	// volatility iteration
	a := math.Log(sigma * sigma)
	A := a
	var B float64
	if delta*delta > phi*phi+v {
		B = math.Log(delta*delta - phi*phi - v)
	} else {
		k := 1.0
		for f(a-k*Tau, phi, v, delta, A) < 0 {
			k++
		}
		B = a - k*Tau
	}
	fA := func(x float64) float64 {
		return f(x, phi, v, delta, A)
	}

	fB := fA(B)
	for i := 0; i < 100; i++ {
		fAVal := fA(A)
		if math.Abs(fAVal) < Epsilon {
			break
		}
		A1 := A
		A = A1 - fAVal*(A1-B)/(fAVal-fB)
		fB = fA(B)
		if math.Abs(A-B) < Epsilon {
			break
		}
	}
	newSigma := math.Exp(A / 2)
	phiStar := math.Sqrt(phi*phi + newSigma*newSigma)
	phiPrime := 1.0 / math.Sqrt((1.0/(phiStar*phiStar))+(1.0/v))
	muPrime := mu + phiPrime*phiPrime*gVal*(score-EVal)

	return glickoState{
		mu:    muPrime,
		phi:   phiPrime,
		sigma: newSigma,
	}
}

// Additional convenience for 1v1 direct update, used by test
func Update1v1(winner, loser models.User) (models.User, models.User) {
	// players := []models.User{winner, loser}
	// winner => score=1, loser => score=0
	scores := map[models.User]float64{
		winner: 1.0,
		loser:  0.0,
	}

	// transform to arrays
	arr := make([]models.User, 2)
	arr[0] = winner
	arr[1] = loser
	sarr := []float64{scores[winner], scores[loser]}

	arr = MultiIterationGlicko2(arr, sarr, 10)
	return arr[0], arr[1]
}
