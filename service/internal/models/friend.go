package models

import "github.com/google/uuid"

type Friend struct {
	User1ID uuid.UUID `json:"user1_id"`
	User2ID uuid.UUID `json:"user2_id"`
	Status  string    `json:"status"` // 'pending', 'accepted'
}
