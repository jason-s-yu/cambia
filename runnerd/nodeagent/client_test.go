package nodeagent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
)

// TestCertificatePinRefusedBeforeBody covers AC8's first half: a coordinator
// certificate whose fingerprint does not match aborts the TLS handshake, so
// the request body never reaches the wire and the coordinator handles nothing.
func TestCertificatePinRefusedBeforeBody(t *testing.T) {
	stub := newStubCoordinator(t)
	cfg := Config{
		Coordinator: Coordinator{URL: stub.url(), CertSHA256: strings.Repeat("ab", 32)},
		BaseDir:     t.TempDir(),
	}
	if err := cfg.normalize(); err != nil {
		t.Fatalf("normalize: %v", err)
	}
	client, err := NewClient(cfg, testSigner(t))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	_, err = client.Register(context.Background(), nashnet.RegisterRequest{NodeID: "node-a"})
	if err == nil {
		t.Fatalf("register succeeded against a coordinator with the wrong certificate")
	}
	if !strings.Contains(err.Error(), ErrCertificatePin.Error()) {
		t.Fatalf("error = %v, want a fingerprint mismatch", err)
	}
	if n := stub.requestCount(); n != 0 {
		t.Fatalf("the coordinator handled %d requests; the pin must fail before any body is sent", n)
	}
}

// TestPinnedClientReachesCoordinator is the positive control for the pin: the
// same construction with the right fingerprint completes a round trip.
func TestPinnedClientReachesCoordinator(t *testing.T) {
	stub := newStubCoordinator(t)
	cfg := Config{
		Coordinator: Coordinator{URL: stub.url(), CertSHA256: stub.fingerprint()},
		BaseDir:     t.TempDir(),
	}
	if err := cfg.normalize(); err != nil {
		t.Fatalf("normalize: %v", err)
	}
	client, err := NewClient(cfg, testSigner(t))
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	resp, err := client.Register(context.Background(), nashnet.RegisterRequest{NodeID: "node-a"})
	if err != nil {
		t.Fatalf("register: %v", err)
	}
	if resp.NodeEpoch == 0 {
		t.Fatalf("register returned no node epoch")
	}
}

// TestClientRefusesAbsoluteDownloadPath keeps a claim response from steering a
// node at a third host: only coordinator-relative paths are fetched.
func TestClientRefusesAbsoluteDownloadPath(t *testing.T) {
	stub := newStubCoordinator(t)
	cfg := Config{
		Coordinator: Coordinator{URL: stub.url(), CertSHA256: stub.fingerprint()},
		BaseDir:     t.TempDir(),
	}
	if err := cfg.normalize(); err != nil {
		t.Fatal(err)
	}
	client, err := NewClient(cfg, testSigner(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Download(context.Background(), "tok", "https://elsewhere.example/x", 0, nil); err == nil {
		t.Fatalf("an absolute download url was accepted")
	}
}

// TestIsFencedClassification pins the D36 rule: both 401 and
// 409 lease_superseded stop a job, and nothing else does.
func TestIsFencedClassification(t *testing.T) {
	cases := []struct {
		err  error
		want bool
	}{
		{&APIError{Status: 401}, true},
		{&APIError{Status: 401, Code: "unauthorized"}, true},
		{&APIError{Status: 409, Code: nashnet.CodeLeaseSuperseded}, true},
		{&APIError{Status: 409, Code: nashnet.CodeManifestOutOfOrder}, false},
		{&APIError{Status: 409, Code: nashnet.CodeBlobsMissing}, false},
		{&APIError{Status: 500}, false},
		{errors.New("connection refused"), false},
	}
	for _, tc := range cases {
		if got := IsFenced(tc.err); got != tc.want {
			t.Errorf("IsFenced(%v) = %v, want %v", tc.err, got, tc.want)
		}
	}
}
