package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/authtoken"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// TestHeartbeatRefreshesAnIdleNode covers the idle-liveness route: it refreshes
// the volatile facts without bumping the epoch, and an unregistered node is told
// to register rather than silently created.
func TestHeartbeatRefreshesAnIdleNode(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	before := r.register(t, r.nodeA, 2)

	resp := r.doNode(t, r.nodeA, http.MethodPost, "/nashnet/nodes/"+r.nodeA.id+"/heartbeat",
		nashnet.HeartbeatRequest{SlotsFree: 1, GateReport: admitReport(1)})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("heartbeat: got %d, want 200", resp.StatusCode)
	}
	var out nashnet.HeartbeatResponse
	decodeInto(t, resp, &out)
	if out.NodeEpoch != before {
		t.Fatalf("heartbeat bumped the epoch to %d, want %d", out.NodeEpoch, before)
	}

	resp = r.doNode(t, r.nodeB, http.MethodPost, "/nashnet/nodes/"+r.nodeB.id+"/heartbeat",
		nashnet.HeartbeatRequest{SlotsFree: 1})
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("heartbeat from an unregistered node: got %d, want 409", resp.StatusCode)
	}
	resp.Body.Close()
}

// TestBlobProbeResumeAndAbandon walks the three upload-support routes: the
// have/want probe over the provable set, the resume offset, the offset fence,
// and abandoning a part.
func TestBlobProbeResumeAndAbandon(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "blob-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	base := "/nashnet/leases/" + claimed.LeaseID
	content := []byte("checkpoint bytes that arrive in two chunks")
	sum := sha256.Sum256(content)
	digest := hex.EncodeToString(sum[:])

	resp := r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/blobs/probe",
		map[string]any{"digests": []string{digest}})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("probe: got %d, want 200", resp.StatusCode)
	}
	var probe struct{ Have, Want []string }
	decodeInto(t, resp, &probe)
	if len(probe.Want) != 1 || len(probe.Have) != 0 {
		t.Fatalf("probe over the provable set = have %v want %v", probe.Have, probe.Want)
	}

	// First half, then a resume probe, then the second half.
	half := len(content) / 2
	r.patchChunk(t, claimed, digest, content[:half], 0, len(content))
	resp = r.doLease(t, claimed.LeaseToken, http.MethodHead, base+"/blobs/"+digest, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("resume probe: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	if got := resp.Header.Get(HeaderOffset); got != strconv.Itoa(half) {
		t.Fatalf("resume offset = %q, want %d", got, half)
	}

	// An append at the wrong offset is refused with the true one.
	bad := r.patchChunkRaw(t, claimed, digest, content[half:], 0, len(content))
	if bad.StatusCode != http.StatusConflict {
		t.Fatalf("wrong offset: got %d, want 409", bad.StatusCode)
	}
	var body nashnet.ErrorBody
	decodeInto(t, bad, &body)
	if body.Error != nashnet.CodeOffsetMismatch || body.Offset != int64(half) {
		t.Fatalf("wrong offset body = %+v, want offset_mismatch at %d", body, half)
	}

	r.patchChunk(t, claimed, digest, content[half:], half, len(content))
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/blobs/probe",
		map[string]any{"digests": []string{digest}})
	decodeInto(t, resp, &probe)
	if len(probe.Have) != 1 {
		t.Fatalf("a verified blob is still want: have %v want %v", probe.Have, probe.Want)
	}

	// A malformed digest never reaches the filesystem.
	resp = r.doLease(t, claimed.LeaseToken, http.MethodDelete, base+"/blobs/NOTHEX", nil)
	if resp.StatusCode != http.StatusUnprocessableEntity {
		t.Fatalf("malformed digest: got %d, want 422", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeInvalidDigest {
		t.Fatalf("malformed digest code = %q, want invalid_digest", code)
	}

	// An abandoned part is gone.
	other := []byte("a part that is abandoned")
	osum := sha256.Sum256(other)
	odigest := hex.EncodeToString(osum[:])
	r.patchChunk(t, claimed, odigest, other[:4], 0, len(other))
	resp = r.doLease(t, claimed.LeaseToken, http.MethodDelete, base+"/blobs/"+odigest, nil)
	if resp.StatusCode != http.StatusNoContent {
		t.Fatalf("abandon: got %d, want 204", resp.StatusCode)
	}
	resp.Body.Close()
	resp = r.doLease(t, claimed.LeaseToken, http.MethodHead, base+"/blobs/"+odigest, nil)
	resp.Body.Close()
	if got := resp.Header.Get(HeaderOffset); got != "0" {
		t.Fatalf("offset after abandon = %q, want 0", got)
	}
}

// TestManifestHeadFencesTheChain covers GET /manifest and the fast-forward
// fence: the node reads the coordinator's head and a replayed or out-of-order
// seq is refused with the head to re-diff against.
func TestManifestHeadFencesTheChain(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "manifest-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	base := "/nashnet/leases/" + claimed.LeaseID

	resp := r.doLease(t, claimed.LeaseToken, http.MethodGet, base+"/manifest", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("manifest head: got %d, want 200", resp.StatusCode)
	}
	var head struct {
		Seq    int64  `json:"seq"`
		Digest string `json:"digest"`
	}
	decodeInto(t, resp, &head)
	if head.Seq != 0 || head.Digest != "" {
		t.Fatalf("fresh head = %+v, want the empty chain", head)
	}

	content := []byte("run_meta\n")
	digest := r.uploadBlob(t, claimed.LeaseToken, claimed.LeaseID, content)
	commit := quarantine.CommitRequest{
		ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 1,
		Entries: []quarantine.Entry{{Path: "run_meta.json", Digest: digest,
			Size: int64(len(content)), MTime: r.clock.now().UnixNano()}},
	}
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/manifest", commit)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("first commit: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	// The same seq with a different body is out of order, not a silent merge.
	commit.Entries[0].Path = "other.json"
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/manifest", commit)
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("replayed seq with a new body: got %d, want 409", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeManifestOutOfOrder {
		t.Fatalf("replayed seq code = %q, want manifest_out_of_order", code)
	}

	// A commit naming a digest nobody uploaded changes nothing.
	resp = r.doLease(t, claimed.LeaseToken, http.MethodGet, base+"/manifest", nil)
	decodeInto(t, resp, &head)
	missing := quarantine.CommitRequest{
		ManifestVersion: 1, LeaseEpoch: claimed.LeaseEpoch, Seq: 2, Parent: head.Digest,
		Entries: []quarantine.Entry{{Path: "absent.pt",
			Digest: "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
			Size:   4, MTime: r.clock.now().UnixNano()}},
	}
	resp = r.doLease(t, claimed.LeaseToken, http.MethodPost, base+"/manifest", missing)
	if resp.StatusCode != http.StatusConflict {
		t.Fatalf("commit naming an unheld digest: got %d, want 409", resp.StatusCode)
	}
	if code := errorCode(t, resp); code != nashnet.CodeBlobsMissing {
		t.Fatalf("unheld digest code = %q, want blobs_missing", code)
	}
}

// TestSnapshotServesTheBundleWithRange covers the code-delivery route: the
// claim's descriptor is what is served, with an ETag and byte ranges so an
// interrupted fetch resumes.
func TestSnapshotServesTheBundleWithRange(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 1)
	r.queueJob(t, JobSpec{Name: "snapshot-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	path := "/nashnet/leases/" + claimed.LeaseID + "/snapshot"
	if claimed.Snapshot.URL != path {
		t.Fatalf("claim snapshot url = %q, want %q", claimed.Snapshot.URL, path)
	}

	resp := r.doLease(t, claimed.LeaseToken, http.MethodGet, path, nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("snapshot: got %d, want 200", resp.StatusCode)
	}
	whole, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	if int64(len(whole)) != claimed.Snapshot.Size {
		t.Fatalf("served %d bytes, the claim reported %d", len(whole), claimed.Snapshot.Size)
	}
	sum := sha256.Sum256(whole)
	if hex.EncodeToString(sum[:]) != claimed.Snapshot.SHA256 {
		t.Fatal("the served bundle does not hash to the digest the claim reported")
	}
	if resp.Header.Get("ETag") == "" {
		t.Fatal("the snapshot must carry an ETag so a fetch can resume")
	}

	ranged := r.raw(t, http.MethodGet, path, nil, map[string]string{
		nashnet.HeaderLeaseToken: claimed.LeaseToken,
		"Range":                  "bytes=5-9",
	})
	if ranged.StatusCode != http.StatusPartialContent {
		t.Fatalf("ranged snapshot: got %d, want 206", ranged.StatusCode)
	}
	window, _ := io.ReadAll(ranged.Body)
	ranged.Body.Close()
	if string(window) != string(whole[5:10]) {
		t.Fatalf("ranged read = %q, want %q", window, whole[5:10])
	}
}

// TestRevokeSupersedesEveryLease is D60: the tombstone the operator writes is
// consulted on the next call, the epoch bump supersedes every lease the node
// held, and the record survives for audit.
func TestRevokeSupersedesEveryLease(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "revoked-job"})
	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}

	resp := r.do(http.MethodPost, "/nashnet/nodes/"+r.nodeA.id+"/revoke", nil)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("revoke: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()

	if _, err := os.Stat(filepath.Join(r.grantDir, r.nodeA.id+authtoken.TombstoneFileSuffix)); err != nil {
		t.Fatalf("revoke wrote no tombstone: %v", err)
	}
	// The node's own credential is refused on its next call, with no id
	// enumeration in the status.
	next := r.doNode(t, r.nodeA, http.MethodPost, "/nashnet/claim", nashnet.ClaimRequest{SlotsFree: 1})
	if next.StatusCode != http.StatusUnauthorized {
		t.Fatalf("revoked node claim: got %d, want 401", next.StatusCode)
	}
	next.Body.Close()
	// Its lease token dies with it.
	lease := r.doLease(t, claimed.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+claimed.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: claimed.LeaseEpoch, Phase: nashnet.PhaseRunning})
	if lease.StatusCode != http.StatusUnauthorized && lease.StatusCode != http.StatusConflict {
		t.Fatalf("revoked lease progress: got %d, want 401 or 409", lease.StatusCode)
	}
	lease.Body.Close()
	if rec, ok := r.pool.nodes.Get(r.nodeA.id); !ok || !rec.Revoked {
		t.Fatal("the record must be kept, marked revoked, for audit")
	}
}

// TestSweeperAppliesTheExpiryVerdict is D7: a lease past its deadline that never
// projected a pid returns its job to the ready set at the same queue position,
// and one whose pid was projected is finalized rather than requeued.
func TestSweeperAppliesTheExpiryVerdict(t *testing.T) {
	r := newPoolRig(t, poolRigConfig{})
	sweeper := r.pool.Sweeper()
	r.register(t, r.nodeA, 2)
	r.queueJob(t, JobSpec{Name: "expiring-job"})

	_, claimed := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("expected a claim")
	}
	r.clock.advance(time.Duration(nashnet.DefaultLeaseTTLSeconds+1) * time.Second)
	out := sweeper.Tick()
	if len(out) != 1 || out[0].Verdict != nashnet.VerdictRequeue {
		t.Fatalf("sweep outcomes = %+v, want one requeue", out)
	}
	view, _ := r.disp.resolveView("expiring-job")
	if isTerminal(view.State) {
		t.Fatalf("a never-launched expiry failed the job: %q", view.State)
	}
	again, second := r.claim(t, r.nodeA, nashnet.ClaimRequest{})
	if second == nil {
		t.Fatalf("the requeued job was not re-placed (claim %d)", again.StatusCode)
	}

	// Once a pid is projected the job has run, so the same expiry finalizes.
	resp := r.doLease(t, second.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+second.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: second.LeaseEpoch, Phase: nashnet.PhaseRunning, PID: 9911})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("progress: got %d, want 200", resp.StatusCode)
	}
	resp.Body.Close()
	r.clock.advance(time.Duration(nashnet.DefaultLeaseTTLSeconds+1) * time.Second)
	out = sweeper.Tick()
	if len(out) != 1 || out[0].Verdict != nashnet.VerdictFinalize {
		t.Fatalf("sweep outcomes = %+v, want one finalize", out)
	}
	view, _ = r.disp.resolveView("expiring-job")
	if !isTerminal(view.State) {
		t.Fatalf("a launched job's expiry left it at %q, want a terminal", view.State)
	}
}

// patchChunk uploads one chunk and fails the test on anything but 200.
func (r *poolRig) patchChunk(t *testing.T, c *nashnet.ClaimResponse, digest string, chunk []byte, start, total int) {
	t.Helper()
	resp := r.patchChunkRaw(t, c, digest, chunk, start, total)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("chunk at %d: got %d, want 200 (%s)", start, resp.StatusCode, body)
	}
}

// patchChunkRaw uploads one chunk and hands back the response for a test that
// expects a refusal.
func (r *poolRig) patchChunkRaw(t *testing.T, c *nashnet.ClaimResponse, digest string, chunk []byte, start, total int) *http.Response {
	t.Helper()
	return r.raw(t, http.MethodPatch,
		"/nashnet/leases/"+c.LeaseID+"/blobs/"+digest, chunk, map[string]string{
			nashnet.HeaderLeaseToken: c.LeaseToken,
			"Content-Range": "bytes " + strconv.Itoa(start) + "-" +
				strconv.Itoa(start+len(chunk)-1) + "/" + strconv.Itoa(total),
		})
}
