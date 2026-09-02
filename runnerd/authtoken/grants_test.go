package authtoken

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

// goldenNode is the fixture node the Python mint writes; every other fixture
// node in this file is generated in-process. Fixture nodes are node-a and
// node-b by name and are described by their grant alone.
const goldenNode = "node-a"

// clock is the injected time source: no test here reads the wall clock.
type clock struct{ t time.Time }

func (c *clock) now() time.Time            { return c.t }
func (c *clock) advance(d time.Duration)   { c.t = c.t.Add(d) }
func newClock(t time.Time) *clock          { return &clock{t: t} }
func at(y int, m time.Month, d int) *clock { return newClock(time.Date(y, m, d, 0, 0, 0, 0, time.UTC)) }

// mintGrant signs an enrollment grant the way cfr/src/harness/nashnet.py does.
// mutate lets a test bend one claim without restating the rest.
func mintGrant(t *testing.T, opPriv ed25519.PrivateKey, nodePub ed25519.PublicKey,
	iat time.Time, life time.Duration, mutate func(jwt.MapClaims)) string {
	t.Helper()
	nodeID, err := DeriveNodeID(nodePub)
	if err != nil {
		t.Fatal(err)
	}
	claims := jwt.MapClaims{
		"aud":         AudienceEnroll,
		"sub":         SubjectForNode(nodeID),
		"node_pubkey": base64.RawURLEncoding.EncodeToString(nodePub),
		"caps":        map[string]any{"max_slots": 2, "kinds": []string{"train"}},
		"iat":         iat.Unix(),
		"exp":         iat.Add(life).Unix(),
	}
	if mutate != nil {
		mutate(claims)
	}
	return mintEdDSA(t, opPriv, claims)
}

// enroll writes a grant for a fresh node keypair into dir and returns the node
// id and its private key.
func enroll(t *testing.T, dir string, opPriv ed25519.PrivateKey, c *clock,
	life time.Duration) (string, ed25519.PrivateKey) {
	t.Helper()
	nodePub, nodePriv := genKeypair(t)
	nodeID, err := DeriveNodeID(nodePub)
	if err != nil {
		t.Fatal(err)
	}
	grant := mintGrant(t, opPriv, nodePub, c.now(), life, nil)
	writeFile(t, filepath.Join(dir, nodeID+GrantFileSuffix), grant)
	return nodeID, nodePriv
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func newStore(t *testing.T, dir string, opPub ed25519.PublicKey, c *clock,
	ttl time.Duration) *GrantStore {
	t.Helper()
	s, err := NewGrantStore(GrantStoreConfig{
		Dir: dir, OperatorKey: opPub, KeysetTTL: ttl, Now: c.now,
	})
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// nodeToken mints a node call token valid at the store's current clock.
func nodeToken(t *testing.T, priv ed25519.PrivateKey, nodeID string, c *clock) string {
	t.Helper()
	return mintEdDSA(t, priv, nodeClaims(SubjectForNode(nodeID), c.now(), time.Minute))
}

func TestDeriveNodeIDIsKeyDerivedAndPathSafe(t *testing.T) {
	pubA, _ := genKeypair(t)
	pubB, _ := genKeypair(t)
	idA, err := DeriveNodeID(pubA)
	if err != nil {
		t.Fatal(err)
	}
	idB, _ := DeriveNodeID(pubB)
	if idA == idB {
		t.Fatal("two keys derived the same node id")
	}
	if len(idA) != len(NodeIDPrefix)+nodeIDHexLen {
		t.Fatalf("node id %q has the wrong length", idA)
	}
	if _, err := NodeIDFromSubject(SubjectForNode(idA)); err != nil {
		t.Fatalf("a derived id must be a valid subject: %v", err)
	}
	if _, err := DeriveNodeID(ed25519.PublicKey("short")); !errors.Is(err, ErrGrantInvalid) {
		t.Fatalf("err = %v, want ErrGrantInvalid", err)
	}
}

func TestGrantResolvesANodeTokenEndToEnd(t *testing.T) {
	dir := t.TempDir()
	opPub, opPriv := genKeypair(t)
	c := at(2026, time.September, 1)
	nodeID, nodePriv := enroll(t, dir, opPriv, c, DefaultGrantLifetime)
	store := newStore(t, dir, opPub, c, DefaultKeysetTTL)

	g, err := store.Grant(nodeID)
	if err != nil {
		t.Fatalf("Grant: %v", err)
	}
	if g.NodeID != nodeID || g.Caps.MaxSlots == nil || *g.Caps.MaxSlots != 2 {
		t.Fatalf("grant = %+v", g)
	}
	got, _, err := store.VerifyNode(nodeToken(t, nodePriv, nodeID, c))
	if err != nil {
		t.Fatalf("VerifyNode: %v", err)
	}
	if got != nodeID {
		t.Fatalf("node id = %q, want %q", got, nodeID)
	}
	// A token signed by a key that is not this node's granted key is refused
	// even though its subject resolves.
	_, otherPriv := genKeypair(t)
	if _, _, err := store.VerifyNode(nodeToken(t, otherPriv, nodeID, c)); err == nil {
		t.Fatal("a token signed by another key should be refused")
	}
}

func TestGrantRefusedWhenTheFileNameIsNotTheDerivedID(t *testing.T) {
	dir := t.TempDir()
	opPub, opPriv := genKeypair(t)
	c := at(2026, time.September, 1)
	store := newStore(t, dir, opPub, c, DefaultKeysetTTL)

	nodeAPub, _ := genKeypair(t)
	nodeBPub, _ := genKeypair(t)
	idA, _ := DeriveNodeID(nodeAPub)
	idB, _ := DeriveNodeID(nodeBPub)

	t.Run("node-a's grant placed under node-b's name", func(t *testing.T) {
		writeFile(t, filepath.Join(dir, idB+GrantFileSuffix),
			mintGrant(t, opPriv, nodeAPub, c.now(), DefaultGrantLifetime, nil))
		if _, err := store.Grant(idB); !errors.Is(err, ErrGrantInvalid) {
			t.Fatalf("err = %v, want ErrGrantInvalid", err)
		}
	})

	t.Run("a subject that its own key does not derive", func(t *testing.T) {
		grant := mintGrant(t, opPriv, nodeAPub, c.now(), DefaultGrantLifetime,
			func(m jwt.MapClaims) { m["sub"] = SubjectForNode(idB) })
		writeFile(t, filepath.Join(dir, idA+GrantFileSuffix), grant)
		if _, err := store.Grant(idA); !errors.Is(err, ErrGrantInvalid) {
			t.Fatalf("err = %v, want ErrGrantInvalid", err)
		}
	})
}

func TestGrantRefusedWhenExpiredUnsignedOrOverLong(t *testing.T) {
	dir := t.TempDir()
	opPub, opPriv := genKeypair(t)
	_, foreignPriv := genKeypair(t)
	c := at(2026, time.September, 1)
	store := newStore(t, dir, opPub, c, DefaultKeysetTTL)
	nodePub, _ := genKeypair(t)
	nodeID, _ := DeriveNodeID(nodePub)
	path := filepath.Join(dir, nodeID+GrantFileSuffix)

	cases := []struct {
		name  string
		grant string
	}{
		{"expired before the clock", mintGrant(t, opPriv, nodePub,
			c.now().Add(-2*time.Hour), time.Hour, nil)},
		{"signed by a key that is not the operator's",
			mintGrant(t, foreignPriv, nodePub, c.now(), DefaultGrantLifetime, nil)},
		{"unsigned (alg none)", func() string {
			tok := jwt.NewWithClaims(jwt.SigningMethodNone, jwt.MapClaims{
				"aud": AudienceEnroll, "sub": SubjectForNode(nodeID),
				"node_pubkey": base64.RawURLEncoding.EncodeToString(nodePub),
				"iat":         c.now().Unix(), "exp": c.now().Add(time.Hour).Unix(),
			})
			s, err := tok.SignedString(jwt.UnsafeAllowNoneSignatureType)
			if err != nil {
				t.Fatal(err)
			}
			return s
		}()},
		{"carrying the operator audience instead of the enrollment one",
			mintGrant(t, opPriv, nodePub, c.now(), DefaultGrantLifetime,
				func(m jwt.MapClaims) { m["aud"] = Audience })},
		{"no exp", mintGrant(t, opPriv, nodePub, c.now(), DefaultGrantLifetime,
			func(m jwt.MapClaims) { delete(m, "exp") })},
		{"a lifetime past the ceiling", mintGrant(t, opPriv, nodePub, c.now(),
			DefaultGrantLifetime+24*time.Hour, nil)},
		{"an unknown cap field", mintGrant(t, opPriv, nodePub, c.now(),
			DefaultGrantLifetime, func(m jwt.MapClaims) {
				m["caps"] = map[string]any{"max_slot": 2}
			})},
		{"not a JWS at all", "this is not a token"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			writeFile(t, path, tc.grant)
			if _, err := store.Grant(nodeID); err == nil {
				t.Fatal("Grant should have refused the file")
			}
		})
	}

	// The ceiling is a configured value, not a constant: a store told to accept
	// longer grants accepts the same file.
	longer, err := NewGrantStore(GrantStoreConfig{
		Dir: dir, OperatorKey: opPub, Now: c.now,
		MaxGrantLifetime: DefaultGrantLifetime + 48*time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, path, mintGrant(t, opPriv, nodePub, c.now(),
		DefaultGrantLifetime+24*time.Hour, nil))
	if _, err := longer.Grant(nodeID); err != nil {
		t.Fatalf("a raised ceiling should accept the grant: %v", err)
	}
}

func TestGrantExpiryIsRecheckedAtVerificationNotOnlyAtLoad(t *testing.T) {
	dir := t.TempDir()
	opPub, opPriv := genKeypair(t)
	c := at(2026, time.September, 1)
	nodePub, nodePriv := genKeypair(t)
	nodeID, _ := DeriveNodeID(nodePub)
	writeFile(t, filepath.Join(dir, nodeID+GrantFileSuffix),
		mintGrant(t, opPriv, nodePub, c.now(), 10*time.Minute, nil))
	// A cache TTL far longer than the grant, so the second call is served from
	// the cache and the refusal can only come from the re-check.
	store := newStore(t, dir, opPub, c, time.Hour)

	c.advance(time.Minute)
	if _, _, err := store.VerifyNode(nodeToken(t, nodePriv, nodeID, c)); err != nil {
		t.Fatalf("VerifyNode inside the grant window: %v", err)
	}
	c.advance(30 * time.Minute)
	_, _, err := store.VerifyNode(nodeToken(t, nodePriv, nodeID, c))
	if !errors.Is(err, ErrGrantExpired) {
		t.Fatalf("err = %v, want ErrGrantExpired", err)
	}
}

func TestTombstoneRefusesTheNextVerification(t *testing.T) {
	dir := t.TempDir()
	opPub, opPriv := genKeypair(t)
	c := at(2026, time.September, 1)
	nodeID, nodePriv := enroll(t, dir, opPriv, c, DefaultGrantLifetime)
	// An hour-long cache TTL: the tombstone must win without waiting for it.
	store := newStore(t, dir, opPub, c, time.Hour)

	tok := nodeToken(t, nodePriv, nodeID, c)
	if _, _, err := store.VerifyNode(tok); err != nil {
		t.Fatalf("VerifyNode before revocation: %v", err)
	}
	writeFile(t, filepath.Join(dir, nodeID+TombstoneFileSuffix), "revoked\n")
	if _, _, err := store.VerifyNode(tok); !errors.Is(err, ErrRevoked) {
		t.Fatalf("err = %v, want ErrRevoked", err)
	}
	if _, err := store.Grant(nodeID); !errors.Is(err, ErrRevoked) {
		t.Fatalf("Grant err = %v, want ErrRevoked", err)
	}
}

func TestKeysetCacheServesWithinTTLAndReloadsAfterIt(t *testing.T) {
	dir := t.TempDir()
	opPub, opPriv := genKeypair(t)
	c := at(2026, time.September, 1)
	nodeID, _ := enroll(t, dir, opPriv, c, DefaultGrantLifetime)
	store := newStore(t, dir, opPub, c, DefaultKeysetTTL)

	if _, err := store.Grant(nodeID); err != nil {
		t.Fatalf("Grant: %v", err)
	}
	// Removing the grant kills the identity at the next load, and the cache TTL
	// is exactly how long that takes (D60).
	if err := os.Remove(filepath.Join(dir, nodeID+GrantFileSuffix)); err != nil {
		t.Fatal(err)
	}
	c.advance(DefaultKeysetTTL - time.Second)
	if _, err := store.Grant(nodeID); err != nil {
		t.Fatalf("within the TTL the cached grant should still serve: %v", err)
	}
	c.advance(2 * time.Second)
	if _, err := store.Grant(nodeID); !errors.Is(err, ErrUnknownNode) {
		t.Fatalf("err = %v, want ErrUnknownNode", err)
	}
}

func TestGrantStoreRefusesAnUnsafeNodeID(t *testing.T) {
	dir := t.TempDir()
	opPub, _ := genKeypair(t)
	store := newStore(t, dir, opPub, at(2026, time.September, 1), DefaultKeysetTTL)
	for _, id := range []string{"", "../etc/passwd", "a/b", "n-x\x00"} {
		if _, err := store.Grant(id); !errors.Is(err, ErrInvalidSubject) {
			t.Fatalf("Grant(%q) err = %v, want ErrInvalidSubject", id, err)
		}
	}
}

func TestNewGrantStoreValidatesItsConfig(t *testing.T) {
	pub, _ := genKeypair(t)
	if _, err := NewGrantStore(GrantStoreConfig{OperatorKey: pub}); err == nil {
		t.Fatal("an empty nodes directory should be refused")
	}
	if _, err := NewGrantStore(GrantStoreConfig{Dir: t.TempDir()}); err == nil {
		t.Fatal("a missing operator key should be refused")
	}
}

// goldenMeta mirrors runnerd/authtoken/testdata/nashnet/golden.json, written by
// cfr/tests/test_harness_nashnet.py.
type goldenMeta struct {
	NodeName          string `json:"node_name"`
	NodeID            string `json:"node_id"`
	Subject           string `json:"subject"`
	GrantFile         string `json:"grant_file"`
	OperatorPubkeyHex string `json:"operator_pubkey_hex"`
	NodeSeedHex       string `json:"node_seed_hex"`
	IAT               int64  `json:"iat"`
	Exp               int64  `json:"exp"`
	LifetimeSeconds   int64  `json:"lifetime_seconds"`
	Caps              Caps   `json:"caps"`
}

// TestGoldenPythonMintedGrantVerifies pins the two implementations of the grant
// together: the fixture is minted by the client module, and this test is the
// coordinator's only reader of it. A drift in either claim set fails here or in
// the Python suite rather than at enrollment time.
func TestGoldenPythonMintedGrantVerifies(t *testing.T) {
	dir := filepath.Join("testdata", "nashnet")
	raw, err := os.ReadFile(filepath.Join(dir, "golden.json"))
	if err != nil {
		t.Fatalf("read golden metadata (regenerate with the cfr suite): %v", err)
	}
	var meta goldenMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	if meta.NodeName != goldenNode {
		t.Fatalf("fixture node = %q, want %q", meta.NodeName, goldenNode)
	}
	opPub, err := hex.DecodeString(meta.OperatorPubkeyHex)
	if err != nil {
		t.Fatal(err)
	}
	seed, err := hex.DecodeString(meta.NodeSeedHex)
	if err != nil {
		t.Fatal(err)
	}
	nodePriv := ed25519.NewKeyFromSeed(seed)

	// A clock one hour into the grant's window: the fixture is time-pinned, so
	// the test never depends on when it runs.
	c := newClock(time.Unix(meta.IAT, 0).UTC().Add(time.Hour))
	store := newStore(t, dir, opPub, c, DefaultKeysetTTL)

	g, err := store.Grant(meta.NodeID)
	if err != nil {
		t.Fatalf("the Python-minted grant must verify: %v", err)
	}
	if got, _ := DeriveNodeID(g.PublicKey); got != meta.NodeID {
		t.Fatalf("derived id = %q, want %q", got, meta.NodeID)
	}
	if g.ExpiresAt.Sub(g.IssuedAt) != DefaultGrantLifetime {
		t.Fatalf("grant lifetime = %s, want %s (the two defaults must agree)",
			g.ExpiresAt.Sub(g.IssuedAt), DefaultGrantLifetime)
	}
	if int64(DefaultGrantLifetime.Seconds()) != meta.LifetimeSeconds {
		t.Fatalf("fixture lifetime %ds does not match DefaultGrantLifetime",
			meta.LifetimeSeconds)
	}
	if g.Caps.MaxSlots == nil || *g.Caps.MaxSlots != *meta.Caps.MaxSlots ||
		g.Caps.MaxVRAMGB["cuda"] != meta.Caps.MaxVRAMGB["cuda"] ||
		len(g.Caps.Kinds) != len(meta.Caps.Kinds) ||
		len(g.Caps.Labels) != len(meta.Caps.Labels) ||
		g.Caps.MaxLeaseBytes == nil || *g.Caps.MaxLeaseBytes != *meta.Caps.MaxLeaseBytes {
		t.Fatalf("caps = %+v, want %+v", g.Caps, meta.Caps)
	}

	// The node the fixture enrolls can drive its own routes.
	nodeID, _, err := store.VerifyNode(nodeToken(t, nodePriv, meta.NodeID, c))
	if err != nil {
		t.Fatalf("VerifyNode with the fixture node key: %v", err)
	}
	if nodeID != meta.NodeID || SubjectForNode(nodeID) != meta.Subject {
		t.Fatalf("node id = %q, want %q", nodeID, meta.NodeID)
	}
	if meta.GrantFile != meta.NodeID+GrantFileSuffix {
		t.Fatalf("fixture grant file %q is not the derived name", meta.GrantFile)
	}
}
