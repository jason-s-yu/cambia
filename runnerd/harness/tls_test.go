package harness

import (
	"crypto/ed25519"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
)

func TestNewServerRequiresVerifier(t *testing.T) {
	disp := NewDispatcher(nil, StubEnvironment{}, t.TempDir(), 1, 16, 0)
	if _, err := NewServer(ServerConfig{Dispatcher: disp}); err == nil {
		t.Fatal("NewServer without a verifier must error (JWT mandatory)")
	}
	if _, err := NewServer(ServerConfig{Verifier: verifierForTest(t)}); err == nil {
		t.Fatal("NewServer without a dispatcher must error")
	}
}

func TestPlaintextRejectedOnTLSListener(t *testing.T) {
	r := newRig(t, rigConfig{})
	// The rig's server is TLS-only (httptest.NewTLSServer). A plaintext request to
	// the same address must fail the handshake.
	plainURL := strings.Replace(r.baseURL, "https://", "http://", 1)
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(plainURL + "/harness/health")
	// A plaintext request must not be served normally. Go's TLS server answers a
	// plaintext request with a 400 ("Client sent an HTTP request to an HTTPS
	// server") rather than a transport error; either outcome proves the endpoint
	// is not reachable in cleartext.
	if err != nil {
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("plaintext GET got %d, want a transport error or 400 (never served)", resp.StatusCode)
	}
}

func TestListenAndServeTLSRequiresCertKey(t *testing.T) {
	r := newRig(t, rigConfig{})
	srv, err := NewServer(ServerConfig{Dispatcher: r.disp, Verifier: verifierForTest(t)})
	if err != nil {
		t.Fatal(err)
	}
	// A missing cert/key pair is a hard failure: there is no plaintext fallback.
	errCh := make(chan error, 1)
	go func() {
		errCh <- srv.ListenAndServeTLS("127.0.0.1:0", "/nonexistent/cert.pem", "/nonexistent/key.pem")
	}()
	select {
	case err := <-errCh:
		if err == nil {
			t.Fatal("ListenAndServeTLS with a missing cert should return an error")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("ListenAndServeTLS did not return; it must not fall back to plaintext")
	}
}

func verifierForTest(t *testing.T) *authtoken.Verifier {
	t.Helper()
	pub, _, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	return authtoken.NewVerifier(pub)
}
