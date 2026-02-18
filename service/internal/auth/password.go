// internal/auth/password.go
package auth

import (
	"crypto/rand"
	"crypto/subtle"
	"encoding/base64"
	"errors"
	"fmt"
	"runtime"
	"strings"

	"golang.org/x/crypto/argon2"
)

// ErrInvalidHash indicates that the stored password hash is in an invalid format.
var ErrInvalidHash = errors.New("the encoded hash is not in the correct format")

// ErrIncompatibleVersion indicates that the Argon2 version is incompatible.
var ErrIncompatibleVersion = errors.New("incompatible version of argon2")

// params holds Argon2id hashing parameters.
type params struct {
	memory      uint32
	iterations  uint32
	parallelism uint8
	saltLength  uint32
	keyLength   uint32
}

// Params is a default set of Argon2id parameters used for hashing.
var Params = &params{
	memory:      64 * 1024,
	iterations:  5,
	parallelism: uint8(runtime.NumCPU() / 2),
	saltLength:  16,
	keyLength:   32,
}

// generateRandomBytes returns n random bytes or an error.
func generateRandomBytes(n uint32) ([]byte, error) {
	b := make([]byte, n)
	_, err := rand.Read(b)
	return b, err
}

// CreateHash uses Argon2id to create a hashed representation of the password.
//
// It returns a string encoded with Argon2 version, parameters, salt, and derived key.
func CreateHash(password string, p *params) (string, error) {
	salt, err := generateRandomBytes(p.saltLength)
	if err != nil {
		return "", err
	}

	hash := argon2.IDKey([]byte(password), salt, p.iterations, p.memory, p.parallelism, p.keyLength)

	b64Salt := base64.RawStdEncoding.EncodeToString(salt)
	b64Hash := base64.RawStdEncoding.EncodeToString(hash)

	encodedHash := fmt.Sprintf("$argon2id$v=%d$m=%d,t=%d,p=%d$%s$%s",
		argon2.Version, p.memory, p.iterations, p.parallelism, b64Salt, b64Hash)
	return encodedHash, nil
}

// ComparePasswordAndHash checks if the provided password matches the Argon2id hash.
//
// It returns true if the password is correct, or false if incorrect. If decoding the
// hash fails or Argon2 versions mismatch, an error is returned.
func ComparePasswordAndHash(password, encodedHash string) (bool, error) {
	p, salt, hash, err := DecodeHash(encodedHash)
	if err != nil {
		return false, err
	}

	newHash := argon2.IDKey([]byte(password), salt, p.iterations, p.memory, p.parallelism, p.keyLength)
	if subtle.ConstantTimeCompare(hash, newHash) == 1 {
		return true, nil
	}
	return false, nil
}

// DecodeHash parses an Argon2id encoded hash and returns its parameters, salt, and key.
func DecodeHash(encodedHash string) (*params, []byte, []byte, error) {
	vals := strings.Split(encodedHash, "$")
	if len(vals) != 6 {
		return nil, nil, nil, ErrInvalidHash
	}

	var version int
	_, err := fmt.Sscanf(vals[2], "v=%d", &version)
	if err != nil {
		return nil, nil, nil, err
	}
	if version != argon2.Version {
		return nil, nil, nil, ErrIncompatibleVersion
	}

	p := &params{}
	_, err = fmt.Sscanf(vals[3], "m=%d,t=%d,p=%d", &p.memory, &p.iterations, &p.parallelism)
	if err != nil {
		return nil, nil, nil, err
	}

	salt, err := base64.RawStdEncoding.Strict().DecodeString(vals[4])
	if err != nil {
		return nil, nil, nil, err
	}
	p.saltLength = uint32(len(salt))

	key, err := base64.RawStdEncoding.Strict().DecodeString(vals[5])
	if err != nil {
		return nil, nil, nil, err
	}
	p.keyLength = uint32(len(key))

	return p, salt, key, nil
}
