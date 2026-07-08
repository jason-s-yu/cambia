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

// doGlickoUpdate is a helper that updates (mu, phi, sigma) vs an average "opponent" in one match.
// It delegates to updateGlickoMulti so the variance, volatility, and rating math live in one place.
// oppSigma is accepted for call-site symmetry but unused: the Glicko2 update depends only on the
// opponent's mu and phi.
func doGlickoUpdate(mu, phi, sigma, oppMu, oppPhi, oppSigma, score float64) glickoState {
	_ = oppSigma
	r := Glicko2Rating{Mu: mu, Phi: phi, Sigma: sigma}
	opp := Glicko2Rating{Mu: oppMu, Phi: oppPhi, Sigma: oppSigma}
	nr := updateGlickoMulti(r, []Glicko2Rating{opp}, []float64{score})
	return glickoState{mu: nr.Mu, phi: nr.Phi, sigma: nr.Sigma}
}

// Update1v1 applies one Glicko2 rating-period update for a single decisive 1v1 result:
// the winner scores 1 against the loser, the loser scores 0 against the winner. Both
// updates read the pre-match opponent rating, and a fresh user (zero Phi1v1/Sigma1v1)
// falls back to the baseline deviation and volatility via ratingFromUser.
func Update1v1(winner, loser models.User) (models.User, models.User) {
	wR := ratingFromUser(winner)
	lR := ratingFromUser(loser)

	newW := updateGlicko(wR, lR, 1.0)
	newL := updateGlicko(lR, wR, 0.0)

	winner.Elo1v1 = int(math.Round(newW.ToElo()))
	winner.Phi1v1 = newW.Phi * GlickoScale
	winner.Sigma1v1 = newW.Sigma

	loser.Elo1v1 = int(math.Round(newL.ToElo()))
	loser.Phi1v1 = newL.Phi * GlickoScale
	loser.Sigma1v1 = newL.Sigma

	return winner, loser
}
