package models

import "github.com/google/uuid"

type Card struct {
	ID    uuid.UUID `json:"id"`
	Suit  string    `json:"suit"`
	Rank  string    `json:"rank"`
	Value int       `json:"value"`
}
