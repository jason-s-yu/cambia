package nodeagent

import (
	"context"
	"io"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// NodeTransport is the whole of the claim, lease, upload, commit, and result
// protocol as one Go interface (D65). It has two implementations: the
// fingerprint-pinned HTTPS Client a remote node runs, and the coordinator's
// own in-process loopback, which its embedded node runs. Neither shortcuts
// validation: the loopback reaches the same handlers, the same lease fence,
// and the same commit transaction as a request off the wire.
//
// It lives here rather than in runnerd/nashnet because the interface has to
// name the manifest shapes, and nashnet is a leaf package that may import no
// other runnerd package (nashnet/leaf_test.go). The coordinator already
// depends on this package for the launch template, so implementing the
// interface here costs it no new edge.
type NodeTransport interface {
	// Register declares the node and adopts the returned policy and epoch (D3).
	Register(ctx context.Context, req nashnet.RegisterRequest) (nashnet.RegisterResponse, error)
	// Heartbeat refreshes the volatile facts and the gate report while idle (D3).
	Heartbeat(ctx context.Context, nodeID string, req nashnet.HeartbeatRequest) (nashnet.HeartbeatResponse, error)
	// Events holds the back-channel long poll (D45).
	Events(ctx context.Context, nodeID string, waitSeconds int) (nashnet.EventsResponse, error)
	// Claim holds the claim long poll (D2). A handout returns the claim and a
	// nil hold; a 204 returns a nil claim and the hold that explains the wait.
	Claim(ctx context.Context, req nashnet.ClaimRequest) (*nashnet.ClaimResponse, *nashnet.ClaimHold, error)
	// Progress renews the lease and drives the coordinator's projection (D5).
	Progress(ctx context.Context, leaseID, leaseToken string, req nashnet.ProgressRequest) (nashnet.ProgressResponse, error)
	// Nack returns a claim the node cannot honor (D8).
	Nack(ctx context.Context, leaseID, leaseToken string, req nashnet.NackRequest) error
	// Result posts the terminal commit (D6).
	Result(ctx context.Context, leaseID, leaseToken string, req nashnet.ResultRequest) (nashnet.ResultResponse, error)
	// Probe asks which digests the lease can already prove (D50 step 1).
	Probe(ctx context.Context, leaseID, leaseToken string, digests []string) (ProbeResponse, error)
	// BlobOffset is the resume probe (D50 step 2).
	BlobOffset(ctx context.Context, leaseID, leaseToken, digest string) (int64, error)
	// AppendChunk uploads one chunk at an exact offset (D50 step 3).
	AppendChunk(ctx context.Context, leaseID, leaseToken, digest string, start, end, total int64, chunk []byte) (int64, error)
	// ManifestHead reads the coordinator's current folded head (D51).
	ManifestHead(ctx context.Context, leaseID, leaseToken string) (ManifestHead, error)
	// Commit posts a fast-forward manifest commit (D51).
	Commit(ctx context.Context, leaseID, leaseToken string, req quarantine.CommitRequest) (quarantine.CommitResponse, error)
	// AppendLog appends raw log bytes at an exact offset (D54).
	AppendLog(ctx context.Context, leaseID, leaseToken string, offset int64, body []byte) error
	// Download streams a lease-scoped GET (the snapshot bundle of D48 or one
	// seed entry of D53) into w, resuming at offset.
	Download(ctx context.Context, leaseToken, rawPath string, offset int64, w io.Writer) (int64, error)
}
