package models

import "github.com/google/uuid"

type User struct {
	ID       uuid.UUID `json:"id"`
	Email    string    `json:"email"`
	Password string    `json:"password,omitempty"`
	Username string    `json:"username"`

	IsEphemeral bool `json:"is_ephemeral"`
	IsAdmin     bool `json:"is_admin"`

	Elo1v1  int `json:"elo_1v1"`
	Elo4p   int `json:"elo_4p"`
	Elo7p8p int `json:"elo_7p8p"`

	// Glicko2 for 1v1
	Phi1v1   float64 `json:"phi_1v1"`
	Sigma1v1 float64 `json:"sigma_1v1"`
}
