// internal/auth/session.go
package auth

import (
	"crypto/ed25519"
	"fmt"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// privateKey and publicKey are used for signing and verifying JWT tokens.
var (
	privateKey ed25519.PrivateKey
	publicKey  ed25519.PublicKey

	// TOKEN_EXPIRE_TIME_SEC indicates how many seconds until JWT expiration (0 => never).
	TOKEN_EXPIRE_TIME_SEC int
)

// parseTokenExpireTime reads the TOKEN_EXPIRE_TIME env var and sets TOKEN_EXPIRE_TIME_SEC accordingly.
func parseTokenExpireTime() {
	duration := os.Getenv("TOKEN_EXPIRE_TIME")
	if duration == "never" || duration == "0" || duration == "" {
		TOKEN_EXPIRE_TIME_SEC = 0
	} else {
		d, err := time.ParseDuration(duration)
		if err != nil {
			fmt.Printf("failed to parse token expire time: %v\n", err)
			os.Exit(1)
		}
		TOKEN_EXPIRE_TIME_SEC = int(d.Seconds())
	}
}

// Init generates a fresh ed25519 key pair at runtime and sets the token expiration.
func Init() {
	var err error
	publicKey, privateKey, err = ed25519.GenerateKey(nil)
	if err != nil {
		fmt.Printf("failed to generate ed25519 key pair: %v\n", err)
		os.Exit(1)
	}
	parseTokenExpireTime()
}

// InitFromPath reads ed25519 private/public keys from file and sets the token expiration.
func InitFromPath(privatePath, publicPath string) error {
	privateKeyData, err := os.ReadFile(privatePath)
	if err != nil {
		return fmt.Errorf("failed to read private key file: %w", err)
	}
	publicKeyData, err := os.ReadFile(publicPath)
	if err != nil {
		return fmt.Errorf("failed to read public key file: %w", err)
	}

	privateKey = ed25519.PrivateKey(privateKeyData)
	publicKey = ed25519.PublicKey(publicKeyData)
	parseTokenExpireTime()
	return nil
}

// CreateJWT creates a signed JWT token with "sub" = userID, exp = now + 72h by default
// (unless overridden by TOKEN_EXPIRE_TIME_SEC).
func CreateJWT(userID string) (string, error) {
	claims := jwt.MapClaims{
		"sub": userID,
	}

	if TOKEN_EXPIRE_TIME_SEC > 0 {
		claims["exp"] = time.Now().Add(time.Duration(TOKEN_EXPIRE_TIME_SEC) * time.Second).Unix()
	} else {
		// "never" means no exp claim
	}

	token := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims)
	return token.SignedString(privateKey)
}

// AuthenticateJWT verifies a JWT string, returns the "sub" field if valid, else an error.
func AuthenticateJWT(tokenString string) (string, error) {
	t, err := jwt.Parse(tokenString, func(t *jwt.Token) (interface{}, error) {
		if _, ok := t.Method.(*jwt.SigningMethodEd25519); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", t.Header["alg"])
		}
		return publicKey, nil
	})

	if err != nil {
		return "", fmt.Errorf("jwt parse error: %w", err)
	}
	if !t.Valid {
		return "", fmt.Errorf("invalid token")
	}

	claims, ok := t.Claims.(jwt.MapClaims)
	if !ok {
		return "", fmt.Errorf("invalid jwt claims")
	}

	userID, ok := claims["sub"].(string)
	if !ok {
		return "", fmt.Errorf("missing sub in jwt")
	}

	return userID, nil
}
