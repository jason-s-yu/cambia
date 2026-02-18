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
		oldPhi := u.Phi1v1
		oldSigma := u.Sigma1v1
		r := NewGlicko2Rating(float64(u.Elo1v1), oldPhi, oldSigma)

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
// against an opponent rOpp, given the final score in [0..1].
func updateGlicko(r, rOpp Glicko2Rating, score float64) Glicko2Rating {
	gVal := g(rOpp.Phi)
	EVal := E(r.Mu, rOpp.Mu, rOpp.Phi)

	v := 1.0 / (gVal * gVal * EVal * (1 - EVal))
	delta := v * gVal * (score - EVal)

	a := math.Log(r.Sigma * r.Sigma)
	A := a
	var B float64
	if delta*delta > r.Phi*r.Phi+v {
		B = math.Log(delta*delta - r.Phi*r.Phi - v)
	} else {
		k := 1.0
		for f(a-k*Tau, r.Phi, v, delta, A) < 0 {
			k++
		}
		B = a - k*Tau
	}

	fA := func(x float64) float64 {
		return f(x, r.Phi, v, delta, A)
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
	phiStar := math.Sqrt(r.Phi*r.Phi + newSigma*newSigma)
	phiPrime := 1.0 / math.Sqrt(1.0/(phiStar*phiStar)+1.0/v)
	muPrime := r.Mu + phiPrime*phiPrime*gVal*(score-EVal)

	return Glicko2Rating{
		Mu:    muPrime,
		Phi:   phiPrime,
		Sigma: newSigma,
	}
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
