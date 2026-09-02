package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"testing"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
	"github.com/jason-s-yu/cambia/runnerd/nodeagent"
)

// TestManifestHeadDecodesIntoTheNodeShape pins the coordinator's answer against
// the type the node actually decodes into, so the two halves of the protocol
// cannot drift on a field name. runnerd/nodeagent imports nothing from this
// package, so reading its wire types here costs no cycle.
func TestManifestHeadDecodesIntoTheNodeShape(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "wire-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	base := "/nashnet/leases/" + claimed.LeaseID

	content := []byte("wire bytes\n")
	digest := r.uploadBlob(t, claimed.LeaseToken, claimed.LeaseID, content)
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/manifest",
		quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 1, Final: true,
			Entries: []quarantine.Entry{{Path: "metrics.jsonl", Digest: digest,
				Size: int64(len(content)), MTime: r.clock.now().UnixNano()}},
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("commit: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	resp = r.doLease(t, claimed.LeaseToken, http.MethodGet, base+"/manifest", nil)
	var head nodeagent.ManifestHead
	decodeInto(t, resp, &head)
	if head.Seq != 1 || head.Digest == "" {
		t.Fatalf("head = %+v, want the committed seq and its digest", head)
	}
	if !head.Final {
		t.Fatal("the head must tell a restarted agent the final manifest is already in")
	}
	if len(head.Entries) != 1 || head.Entries[0].Path != "metrics.jsonl" {
		t.Fatalf("head entries = %+v, want the promoted entry", head.Entries)
	}

	// The probe answer decodes into the node's own shape too.
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/blobs/probe",
		nodeagent.ProbeRequest{Digests: []string{digest}})
	var probe nodeagent.ProbeResponse
	decodeInto(t, resp, &probe)
	if len(probe.Have) != 1 || probe.Have[0] != digest {
		t.Fatalf("probe = %+v, want the uploaded digest under have", probe)
	}
}

// TestZeroByteBlobIsABodylessPatch pins the one upload the range grammar cannot
// express: a zero-byte artifact arrives as bytes */0 with no body, and the
// coordinator creates the blob against the digest the empty string hashes to.
func TestZeroByteBlobIsABodylessPatch(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "empty-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	sum := sha256.Sum256(nil)
	digest := hex.EncodeToString(sum[:])

	resp := r.raw(t, http.MethodPatch,
		"/nashnet/leases/"+claimed.LeaseID+"/blobs/"+digest, nil, map[string]string{
			nashnet.HeaderLeaseToken: claimed.LeaseToken,
			"Content-Range":          "bytes */0",
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("zero-byte PATCH: got %d, want 200", resp.StatusCode)
	}
	var chunk nodeagent.ChunkResponse
	decodeInto(t, resp, &chunk)
	if chunk.CommittedOffset != 0 {
		t.Fatalf("committed_offset = %d, want 0", chunk.CommittedOffset)
	}

	// The empty blob is now provable, so a manifest may name it.
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/manifest", quarantine.CommitRequest{
			ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 1,
			Entries: []quarantine.Entry{{Path: "metrics.jsonl", Digest: digest,
				Size: 0, MTime: r.clock.now().UnixNano()}},
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("commit naming the empty blob: got %d, want 200", resp.StatusCode)
	}
	var committed quarantine.CommitResponse
	decodeInto(t, resp, &committed)
	if len(committed.Promoted) != 1 {
		t.Fatalf("promoted = %v, want the zero-byte entry", committed.Promoted)
	}
}

// TestNeverLaunchedTerminalNeedsNoJournal is the coordinator half of the node's
// prepare-failure path: a job that never launched has no artifacts to commit,
// so its terminal is accepted rather than held behind the artifact gate of D6.
func TestNeverLaunchedTerminalNeedsNoJournal(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "prepare-fail-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: claimed.LeaseEpoch, Phase: nashnet.PhasePreparing})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("progress: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/result", nashnet.ResultRequest{
			LeaseEpoch: claimed.LeaseEpoch, State: nashnet.ResultFailed,
			LastError: "config render rejected at the pinned commit",
		})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("never-launched terminal: got %d, want 200", resp.StatusCode)
	}
	var recorded nashnet.ResultResponse
	decodeInto(t, resp, &recorded)
	if recorded.State != nashnet.ResultFailed {
		t.Fatalf("recorded state = %q, want failed", recorded.State)
	}
	view, _ := r.disp.resolveView("prepare-fail-job")
	if view.State != StateFailed {
		t.Fatalf("job state = %q, want failed", view.State)
	}

	// A launched lease still needs its final manifest.
	r.queueJob(t, JobSpec{Name: "launched-job"})
	_, second := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if second == nil {
		t.Fatal("expected a second claim")
	}
	resp = r.doLease(t, second.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+second.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: second.LeaseEpoch, Phase: nashnet.PhaseRunning, PID: 7788})
	resp.Body.Close()
	resp = r.doLease(t, second.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+second.LeaseID+"/result", nashnet.ResultRequest{
			LeaseEpoch: second.LeaseEpoch, State: nashnet.ResultCrashed,
		})
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("launched terminal with no manifest: got %d, want 409", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeArtifactsIncomplete {
		t.Fatalf("launched terminal code = %q, want artifacts_incomplete", code)
	}
}
