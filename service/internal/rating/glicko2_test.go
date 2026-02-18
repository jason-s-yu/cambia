package rating

import (
	"testing"

	"github.com/jason-s-yu/cambia/service/internal/models"
)

func TestUpdate1v1(t *testing.T) {
	winner := models.User{Elo1v1: 1500}
	loser := models.User{Elo1v1: 1500}

	newW, newL := Update1v1(winner, loser)
	if newW.Elo1v1 <= 1500 {
		t.Errorf("winner's rating should have gone up, got %d", newW.Elo1v1)
	}
	if newL.Elo1v1 >= 1500 {
		t.Errorf("loser's rating should have gone down, got %d", newL.Elo1v1)
	}
}
