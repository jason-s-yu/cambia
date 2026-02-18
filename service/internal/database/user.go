package database

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5"
	"github.com/jason-s-yu/cambia/service/internal/auth"
	"github.com/jason-s-yu/cambia/service/internal/models"
)

func CreateUser(ctx context.Context, user *models.User) error {
	if user.ID == uuid.Nil {
		id, err := uuid.NewRandom()
		if err != nil {
			return fmt.Errorf("failed to generate user id: %w", err)
		}
		user.ID = id
	}

	hash, err := auth.CreateHash(user.Password, auth.Params)
	if err != nil {
		return fmt.Errorf("failed to hash password: %w", err)
	}
	user.Password = hash

	q := `INSERT INTO users (id, email, password, username, is_ephemeral, is_admin)
	      VALUES ($1, $2, $3, $4, $5, $6)`

	err = pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, execErr := tx.Exec(ctx, q,
			user.ID, user.Email, user.Password, user.Username,
			user.IsEphemeral, user.IsAdmin,
		)
		return execErr
	})
	if err != nil {
		return fmt.Errorf("failed to insert user: %w", err)
	}
	return nil
}

func GetUserByEmail(ctx context.Context, email string) (*models.User, error) {
	var u models.User
	q := `
	SELECT id, email, password, username, is_ephemeral, is_admin,
	       elo_1v1, elo_4p, elo_7p8p,
	       phi_1v1, sigma_1v1
	FROM users
	WHERE email=$1
	`
	err := DB.QueryRow(ctx, q, email).Scan(
		&u.ID, &u.Email, &u.Password, &u.Username,
		&u.IsEphemeral, &u.IsAdmin,
		&u.Elo1v1, &u.Elo4p, &u.Elo7p8p,
		&u.Phi1v1, &u.Sigma1v1,
	)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

func GetUserByID(ctx context.Context, id uuid.UUID) (*models.User, error) {
	var u models.User
	q := `
	SELECT id, email, password, username, is_ephemeral, is_admin,
	       elo_1v1, elo_4p, elo_7p8p,
	       phi_1v1, sigma_1v1
	FROM users
	WHERE id=$1
	`
	err := DB.QueryRow(ctx, q, id).Scan(
		&u.ID, &u.Email, &u.Password, &u.Username,
		&u.IsEphemeral, &u.IsAdmin,
		&u.Elo1v1, &u.Elo4p, &u.Elo7p8p,
		&u.Phi1v1, &u.Sigma1v1,
	)
	if err != nil {
		return nil, err
	}
	return &u, nil
}

func AuthenticateUser(ctx context.Context, email, password string) (string, error) {
	user, err := GetUserByEmail(ctx, email)
	if err != nil {
		return "", fmt.Errorf("user not found or db error: %w", err)
	}

	match, err := auth.ComparePasswordAndHash(password, user.Password)
	if err != nil || !match {
		return "", fmt.Errorf("invalid credentials")
	}

	token, err := auth.CreateJWT(user.ID.String())
	if err != nil {
		return "", fmt.Errorf("failed to create jwt: %w", err)
	}

	return token, nil
}

// SaveUserGlicko1v1 stores the user's ELO, phi, and sigma in the DB
func SaveUserGlicko1v1(ctx context.Context, u *models.User) error {
	q := `
	UPDATE users
	SET elo_1v1=$1, phi_1v1=$2, sigma_1v1=$3
	WHERE id=$4
	`
	return pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, err := tx.Exec(ctx, q, u.Elo1v1, u.Phi1v1, u.Sigma1v1, u.ID)
		return err
	})
}

// UpdateUserCredentials updates a user's email/password and ephemeral flag in DB
func UpdateUserCredentials(ctx context.Context, u *models.User) error {
	hashed, err := auth.CreateHash(u.Password, auth.Params)
	if err != nil {
		return err
	}

	q := `UPDATE users SET email = $1, password = $2, is_ephemeral = $3 WHERE id = $4`
	err = pgx.BeginTxFunc(ctx, DB, pgx.TxOptions{}, func(tx pgx.Tx) error {
		_, e := tx.Exec(ctx, q, u.Email, hashed, u.IsEphemeral, u.ID)
		return e
	})
	if err != nil {
		return fmt.Errorf("failed to update user credentials: %w", err)
	}
	return nil
}
