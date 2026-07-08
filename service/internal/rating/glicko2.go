// internal/rating/glicko2.go
package rating

import (
	"log"
	"math"

	"github.com/jason-s-yu/cambia/service/internal/models"
)

const (
	// GlickoScale is the multiplier used for converting between Elo and Glicko2's mu.
	GlickoScale = 173.7178
	// DefaultMu is the baseline rating (1500) in Glicko2 terms.
	DefaultMu = 1500.0
	// DefaultPhi is the baseline rating deviation (RD) in Glicko2 terms (350).
	DefaultPhi = 350.0
	// Tau is the constraint on volatility changes.
	Tau = 0.5
	// Epsilon is the tolerance used in iteration stopping conditions.
	Epsilon = 0.000001
)

// Glicko2Rating holds the transformed rating (Mu), rating deviation (Phi),
// and volatility (Sigma) for a single user in Glicko2 space.
type Glicko2Rating struct {
	Mu    float64
	Phi   float64
	Sigma float64
}

// NewGlicko2Rating creates a new Glicko2Rating from a standard Elo, rating deviation, and volatility.
//
// elo is the user's current rating in standard "1500-based" scale.
// rd is the user's rating deviation in the same scale (e.g., 350).
// sigma is the user's volatility (typically around 0.06).
func NewGlicko2Rating(elo, rd, sigma float64) Glicko2Rating {
	return Glicko2Rating{
		Mu:    (elo - DefaultMu) / GlickoScale,
		Phi:   rd / GlickoScale,
		Sigma: sigma,
	}
}

// ToElo converts a Glicko2Rating's Mu back to a standard 1500-based Elo scale.
func (r Glicko2Rating) ToElo() float64 {
	return r.Mu*GlickoScale + DefaultMu
}

// ratingFromUser builds a Glicko2Rating from a user's stored 1v1 fields, substituting
// the baseline deviation and volatility when a user has no prior rating (zero-valued
// Phi1v1/Sigma1v1). Without this guard a fresh user would yield phi=0 or sigma=0, which
// makes ln(sigma^2) and the deviation update undefined.
func ratingFromUser(u models.User) Glicko2Rating {
	phi := u.Phi1v1
	if phi <= 0 {
		phi = DefaultPhi
	}
	sigma := u.Sigma1v1
	if sigma <= 0 {
		sigma = 0.06
	}
	return NewGlicko2Rating(float64(u.Elo1v1), phi, sigma)
}

// SingleOrMultiPlayerGlicko2 applies a single-step Glicko2 update for a group of users
// given their final scores in [0..1]. For multi-player, we approximate the "opponent rating"
// as the average of all other players' Elos.
//
// allUsers is the slice of user objects with fields Elo1v1, Phi1v1, Sigma1v1
// scores is a parallel slice of float64 in [0..1] representing each user's final fraction
// returns a new slice of updated users
func SingleOrMultiPlayerGlicko2(allUsers []models.User, scores []float64) []models.User {
	if len(allUsers) != len(scores) {
		log.Printf("Mismatch in user vs. score count. No rating update performed.")
		return allUsers
	}

	var totalMu float64
	for i := range allUsers {
		totalMu += float64(allUsers[i].Elo1v1)
	}
	avgElo := totalMu / float64(len(allUsers))

	updated := make([]models.User, len(allUsers))
	for i, u := range allUsers {
		r := ratingFromUser(u)

		oppElo := (avgElo*float64(len(allUsers)) - float64(u.Elo1v1)) / float64(len(allUsers)-1)
		oppR := NewGlicko2Rating(oppElo, DefaultPhi, 0.06)

		newR := updateGlicko(r, oppR, scores[i])
		newElo := newR.ToElo()
		u.Elo1v1 = int(math.Round(newElo))
		u.Phi1v1 = newR.Phi * GlickoScale
		u.Sigma1v1 = newR.Sigma
		updated[i] = u
	}
	return updated
}

// updateGlicko performs a single-match Glicko2 update with volatility for a user r
// against an opponent rOpp, given the final score in [0..1]. It is the one-opponent
// case of updateGlickoMulti.
func updateGlicko(r, rOpp Glicko2Rating, score float64) Glicko2Rating {
	return updateGlickoMulti(r, []Glicko2Rating{rOpp}, []float64{score})
}

// updateGlickoMulti applies a single Glicko2 rating-period update (Glickman steps 3-8)
// for player r against a set of opponents with the given scores in [0..1]. Variance v
// and the delta sum are accumulated across all opponents, then a single volatility,
// deviation, and rating update is applied. With an empty opponent set the rating is
// unchanged except that the deviation grows by the volatility (step 6 with no games).
func updateGlickoMulti(r Glicko2Rating, opps []Glicko2Rating, scores []float64) Glicko2Rating {
	if len(opps) == 0 {
		phiStar := math.Sqrt(r.Phi*r.Phi + r.Sigma*r.Sigma)
		return Glicko2Rating{Mu: r.Mu, Phi: phiStar, Sigma: r.Sigma}
	}

	// Step 3 (variance) and step 4 (delta) accumulated over all opponents.
	var invV, deltaSum float64
	for j := range opps {
		gj := g(opps[j].Phi)
		Ej := E(r.Mu, opps[j].Mu, opps[j].Phi)
		invV += gj * gj * Ej * (1 - Ej)
		deltaSum += gj * (scores[j] - Ej)
	}
	v := 1.0 / invV
	delta := v * deltaSum

	// Step 5: new volatility via the Illinois algorithm.
	newSigma := glickoVolatility(r.Phi, v, delta, r.Sigma)

	// Steps 6-7: pre-rating-period deviation, then new deviation.
	phiStar := math.Sqrt(r.Phi*r.Phi + newSigma*newSigma)
	phiPrime := 1.0 / math.Sqrt(1.0/(phiStar*phiStar)+1.0/v)

	// Step 8: new rating. mu' = mu + phi'^2 * sum_j g(phi_j) * (s_j - E_j).
	muPrime := r.Mu + phiPrime*phiPrime*deltaSum

	return Glicko2Rating{
		Mu:    muPrime,
		Phi:   phiPrime,
		Sigma: newSigma,
	}
}

// glickoVolatility solves for the new volatility sigma' using the Illinois algorithm
// from the Glickman paper (step 5). The constant a = ln(sigma^2) is closed over so the
// root-finder never mutates it; A and B bracket the root and the false-position estimate
// C replaces one endpoint each iteration.
func glickoVolatility(phi, v, delta, sigma float64) float64 {
	a := math.Log(sigma * sigma)
	fn := func(x float64) float64 {
		return f(x, phi, v, delta, a)
	}

	A := a
	var B float64
	if delta*delta > phi*phi+v {
		B = math.Log(delta*delta - phi*phi - v)
	} else {
		k := 1.0
		for fn(a-k*Tau) < 0 {
			k++
		}
		B = a - k*Tau
	}

	fA := fn(A)
	fB := fn(B)
	for math.Abs(B-A) > Epsilon {
		C := A + (A-B)*fA/(fB-fA)
		fC := fn(C)
		if fC*fB <= 0 {
			A, fA = B, fB
		} else {
			fA = fA / 2
		}
		B, fB = C, fC
	}
	return math.Exp(A / 2)
}

// g is the G(phi) factor from Glicko2, applying the standard formula 1/sqrt(1+3phi^2/pi^2).
func g(phi float64) float64 {
	return 1.0 / math.Sqrt(1.0+3.0*phi*phi/math.Pi/math.Pi)
}

// E is the expected score formula in Glicko2 space, E(mu,mu2,phi2)=1/(1+exp[-g(phi2)*(mu-mu2)])
func E(mu, mu2, phi2 float64) float64 {
	return 1.0 / (1.0 + math.Exp(-g(phi2)*(mu-mu2)))
}

// f is the Glicko2 volatility root-finding function used in the iterative volatility update.
func f(x, phi, v, delta, a float64) float64 {
	ex := math.Exp(x)
	num := ex * (delta*delta - phi*phi - v - ex)
	den := 2.0 * (phi*phi + v + ex) * (phi*phi + v + ex)
	return (num / den) - ((x - a) / (Tau * Tau))
}
