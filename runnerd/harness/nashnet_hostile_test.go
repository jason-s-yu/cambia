package harness

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/quarantine"
)

// The hostile-node suite of D42: one node implementation whose only purpose is
// to attempt every item in the "cannot" list of design section 5. Each attempt
// names the rejection code it expects and the paths that stayed untouched,
// which is what keeps the threat model executable as routes are added rather
// than a paragraph that ages.
//
// The hostile node is the rig's raw HTTP client rather than nodeagent.Client:
// the production client will not build most of these requests (it refuses an
// absolute download path, it mints only node-audience tokens, it never names
// another node in a body), and a threat model that could only send
// well-formed requests would test nothing. Its credentials are honest: it holds
// node-a's real enrollment grant and a lease it really claimed, so every
// refusal below is the coordinator declining an authenticated caller rather
// than an unauthenticated one bouncing off the door.

// hostileNode is one enrolled node acting in bad faith, plus the coordinator
// state a test compares against before and after every attempt.
type hostileNode struct {
	*nodeRig
	// lease is a lease this node really holds, obtained by an honest claim.
	lease *nashnet.ClaimResponse
	// victim is the second enrolled node, whose record and leases every
	// impersonation attempt must leave exactly as it found them.
	victim fixtureNode
}

// newHostileNode stands up the coordinator, enrolls both nodes, and hands the
// hostile one a real lease on its own job.
func newHostileNode(t *testing.T, cfg nodeRigConfig) *hostileNode {
	t.Helper()
	r := newNodeRig(t, cfg)
	r.register(t, r.node, 2)
	r.queueFixtureJob(t, "hostile-job", 2, "quick")
	_, claimed := r.claim(t, r.node, nashnet.ClaimRequest{})
	if claimed == nil {
		t.Fatal("the hostile node was handed no lease: it needs an honest one to abuse")
	}
	return &hostileNode{nodeRig: r, lease: claimed, victim: r.nodeB}
}

// fileFacts is what a snapshot records per file: enough to catch a write, a
// truncation, or a replacement, and nothing that changes on its own.
type fileFacts struct {
	Size   int64
	Mode   os.FileMode
	Digest string
}

// fsSnapshot is the state of every coordinator-owned tree a hostile attempt
// must not change: the runs dir, the quarantine tree, the enrollment grants,
// and the git mirror.
type fsSnapshot map[string]fileFacts

// snapshotTrees walks every root and records one entry per regular file, keyed
// by root name plus its relative path.
func snapshotTrees(t *testing.T, roots map[string]string) fsSnapshot {
	t.Helper()
	out := fsSnapshot{}
	for name, root := range roots {
		err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				if os.IsNotExist(err) {
					return nil
				}
				return err
			}
			rel, rerr := filepath.Rel(root, path)
			if rerr != nil {
				return rerr
			}
			key := name + "/" + filepath.ToSlash(rel)
			info, ierr := d.Info()
			if ierr != nil {
				if os.IsNotExist(ierr) {
					return nil
				}
				return ierr
			}
			if d.IsDir() {
				out[key+"/"] = fileFacts{Mode: info.Mode()}
				return nil
			}
			if !info.Mode().IsRegular() {
				// A symlink, socket, fifo, or device node is itself a finding:
				// record its type so its appearance shows up as a change.
				out[key] = fileFacts{Mode: info.Mode(), Digest: "irregular"}
				return nil
			}
			body, rerr := os.ReadFile(path)
			if rerr != nil {
				if os.IsNotExist(rerr) {
					return nil
				}
				return rerr
			}
			sum := sha256.Sum256(body)
			out[key] = fileFacts{Size: info.Size(), Mode: info.Mode(), Digest: hex.EncodeToString(sum[:])}
			return nil
		})
		if err != nil {
			t.Fatalf("snapshot %s: %v", root, err)
		}
	}
	return out
}

// roots are the coordinator-owned trees every hostile attempt is measured
// against.
func (h *hostileNode) roots() map[string]string {
	return map[string]string{
		"runs":       h.runsDir,
		"quarantine": h.quarDir,
		"grants":     h.grantDir,
		"mirror":     h.repo.coordDir,
	}
}

// snapshot records the coordinator's trees as they stand.
func (h *hostileNode) snapshot(t *testing.T) fsSnapshot {
	t.Helper()
	return snapshotTrees(t, h.roots())
}

// assertUntouched is the second half of every hostile assertion: the attempt
// was refused with a named code, and no path under any coordinator-owned tree
// changed. It reports the first difference by name rather than a bare count, so
// a regression says which file the node reached.
func (h *hostileNode) assertUntouched(t *testing.T, what string, before fsSnapshot) {
	t.Helper()
	after := h.snapshot(t)
	var changes []string
	for key, was := range before {
		now, ok := after[key]
		if !ok {
			changes = append(changes, "deleted "+key)
			continue
		}
		if now != was {
			changes = append(changes, fmt.Sprintf("changed %s (%d bytes %s -> %d bytes %s)",
				key, was.Size, short(was.Digest), now.Size, short(now.Digest)))
		}
	}
	for key := range after {
		if _, ok := before[key]; !ok {
			changes = append(changes, "created "+key)
		}
	}
	if len(changes) > 0 {
		sort.Strings(changes)
		t.Fatalf("%s touched the coordinator: %s", what, strings.Join(changes, "; "))
	}
}

func short(digest string) string {
	if len(digest) > 8 {
		return digest[:8]
	}
	return digest
}

// refusal issues one request and asserts the status and error code it comes
// back with, returning nothing: an attempt that is refused has no useful body
// beyond its code.
func (h *hostileNode) refusal(t *testing.T, what string, resp *http.Response, wantStatus int, wantCode string) {
	t.Helper()
	if resp.StatusCode != wantStatus {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("%s: got %d, want %d (%s)", what, resp.StatusCode, wantStatus, body)
	}
	if wantCode == "" {
		resp.Body.Close()
		return
	}
	if code := errorCode(t, resp); code != wantCode {
		t.Fatalf("%s: code = %q, want %q", what, code, wantCode)
	}
}

// TestHostileNodeCannotReachTheOperatorAPI covers the first line of the cannot
// list: a node may not submit, cancel, purge, or resume any job, nor enumerate
// jobs, runs, nodes, or the queue (D26). Every one is the same refusal, named
// rather than a bare 401, and none of them changes anything.
func TestHostileNodeCannotReachTheOperatorAPI(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	attempts := []struct {
		what   string
		method string
		path   string
		body   any
	}{
		{"submit a job", http.MethodPost, "/harness/jobs", baseSpec("hostile-submit", "fake-quick")},
		{"cancel a job", http.MethodDelete, "/harness/jobs/hostile-job", nil},
		{"purge a job", http.MethodDelete, "/harness/jobs/hostile-job?purge=true", nil},
		{"resume a job", http.MethodPost, "/harness/jobs/hostile-job/resume", nil},
		{"enumerate jobs", http.MethodGet, "/harness/jobs", nil},
		{"enumerate one job", http.MethodGet, "/harness/jobs/hostile-job", nil},
		{"enumerate a run's artifacts", http.MethodGet, "/harness/jobs/hostile-job/artifacts", nil},
		{"enumerate nodes", http.MethodGet, "/nashnet/nodes", nil},
		{"read another node's record", http.MethodGet, "/nashnet/nodes/" + h.victim.id, nil},
		{"drain another node", http.MethodPost, "/nashnet/nodes/" + h.victim.id + "/drain", nil},
		{"revoke another node", http.MethodPost, "/nashnet/nodes/" + h.victim.id + "/revoke", nil},
	}
	for _, a := range attempts {
		before := h.snapshot(t)
		resp := h.doNode(t, h.node, a.method, a.path, a.body)
		h.refusal(t, a.what, resp, http.StatusForbidden, nashnet.CodeWrongAudience)
		h.assertUntouched(t, a.what, before)
	}
}

// TestHostileNodeCannotReadAForeignLog is the foreign log read: the log tail is
// an operator route, and a node holding a lease on one job has no route to any
// job's log at all (D26, D54).
func TestHostileNodeCannotReadAForeignLog(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	before := h.snapshot(t)
	resp := h.doNode(t, h.node, http.MethodGet, "/ws/harness/jobs/hostile-job/logs", nil)
	h.refusal(t, "read a job log", resp, http.StatusForbidden, nashnet.CodeWrongAudience)
	h.assertUntouched(t, "read a job log", before)
}

// TestHostileNodeCredentialsAreNotInterchangeable is the credential matrix: the
// node token, the operator token, and the lease token each name their own side,
// and presenting one on another's route says which rather than answering a bare
// 401 (D26, D44).
func TestHostileNodeCredentialsAreNotInterchangeable(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	cases := []struct {
		what    string
		method  string
		path    string
		headers map[string]string
	}{
		{
			what: "a node token on an operator route", method: http.MethodGet, path: "/harness/jobs",
			headers: map[string]string{"Authorization": "Bearer " + h.nodeToken(t, h.node)},
		},
		{
			what: "an operator token on a node route", method: http.MethodPost, path: "/nashnet/claim",
			headers: map[string]string{"Authorization": "Bearer " + h.token},
		},
		{
			what: "a lease token on a node route", method: http.MethodPost, path: "/nashnet/nodes/register",
			headers: map[string]string{nashnet.HeaderLeaseToken: h.lease.LeaseToken},
		},
		{
			what:   "a node token on a lease route",
			method: http.MethodPost, path: "/nashnet/leases/" + h.lease.LeaseID + "/progress",
			headers: map[string]string{"Authorization": "Bearer " + h.nodeToken(t, h.node)},
		},
		{
			what:   "an operator token on a lease route",
			method: http.MethodPost, path: "/nashnet/leases/" + h.lease.LeaseID + "/progress",
			headers: map[string]string{"Authorization": "Bearer " + h.token},
		},
	}
	for _, c := range cases {
		before := h.snapshot(t)
		resp := h.raw(t, c.method, c.path, nashnet.ProgressRequest{}, c.headers)
		h.refusal(t, c.what, resp, http.StatusForbidden, nashnet.CodeWrongAudience)
		h.assertUntouched(t, c.what, before)
	}
}

// TestHostileNodeCannotActOnALeaseItDoesNotHold covers acting on another
// lease, a bumped epoch, and a token after revocation: every mutating lease
// call is fenced on (lease_id, lease_epoch, node_epoch) plus a constant-time
// token compare, and the tombstone is consulted on every verification (D4, D44,
// D60).
func TestHostileNodeCannotActOnALeaseItDoesNotHold(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})

	// The victim's lease, obtained honestly by the victim, is what the hostile
	// node then tries to act on with its own token.
	h.register(t, h.victim, 2)
	h.queueFixtureJob(t, "victim-job", 1, "quick")
	_, victimLease := h.claim(t, h.victim, nashnet.ClaimRequest{})
	if victimLease == nil {
		t.Fatal("the victim was handed no lease")
	}

	before := h.snapshot(t)
	resp := h.doLease(t, h.lease.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+victimLease.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: victimLease.LeaseEpoch, Phase: nashnet.PhaseRunning})
	h.refusal(t, "progress on another node's lease", resp,
		http.StatusConflict, nashnet.CodeLeaseSuperseded)
	h.assertUntouched(t, "progress on another node's lease", before)

	before = h.snapshot(t)
	resp = h.doLease(t, h.lease.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: h.lease.LeaseEpoch + 7, Phase: nashnet.PhaseRunning})
	h.refusal(t, "progress at a bumped lease epoch", resp,
		http.StatusConflict, nashnet.CodeLeaseSuperseded)
	h.assertUntouched(t, "progress at a bumped lease epoch", before)

	// A revoked node keeps nothing: the tombstone is read on every call, so the
	// lease token it still holds authorizes nothing afterwards (D36, D60).
	op := h.do(http.MethodPost, "/nashnet/nodes/"+h.node.id+"/revoke", nil)
	if op.StatusCode != http.StatusOK && op.StatusCode != http.StatusAccepted {
		t.Fatalf("operator revoke: got %d", op.StatusCode)
	}
	op.Body.Close()

	before = h.snapshot(t)
	resp = h.doLease(t, h.lease.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/progress",
		nashnet.ProgressRequest{LeaseEpoch: h.lease.LeaseEpoch, Phase: nashnet.PhaseRunning})
	if resp.StatusCode != http.StatusUnauthorized && resp.StatusCode != http.StatusConflict {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("progress after revocation: got %d, want 401 or 409 (%s)", resp.StatusCode, body)
	}
	resp.Body.Close()
	h.assertUntouched(t, "progress after revocation", before)

	before = h.snapshot(t)
	resp = h.doNode(t, h.node, http.MethodPost, "/nashnet/claim", nashnet.ClaimRequest{SlotsFree: 1})
	if resp.StatusCode != http.StatusUnauthorized && resp.StatusCode != http.StatusForbidden {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("claim after revocation: got %d, want 401 or 403 (%s)", resp.StatusCode, body)
	}
	resp.Body.Close()
	h.assertUntouched(t, "claim after revocation", before)
}

// TestHostileNodeCannotGuessAnotherLeasesData covers reading another job's or
// node's data by guessing a digest or a seed path: the probe answers over this
// lease's provable set alone, and an out-of-set read is indistinguishable from
// an unknown one, byte for byte (D50, D53).
func TestHostileNodeCannotGuessAnotherLeasesData(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})

	// The victim uploads a blob under its own lease. Its digest is the secret
	// the hostile node then guesses.
	h.register(t, h.victim, 2)
	h.queueFixtureJob(t, "victim-job", 1, "quick")
	_, victimLease := h.claim(t, h.victim, nashnet.ClaimRequest{})
	if victimLease == nil {
		t.Fatal("the victim was handed no lease")
	}
	secret := []byte("the victim's checkpoint bytes\n")
	digest := h.uploadBlob(t, victimLease.LeaseToken, victimLease.LeaseID, secret)

	before := h.snapshot(t)
	resp := h.doLease(t, h.lease.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/blobs/probe", ProbeBody{Digests: []string{digest}})
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("probe: got %d, want 200", resp.StatusCode)
	}
	var probe struct {
		Have []string `json:"have"`
		Want []string `json:"want"`
	}
	decodeInto(t, resp, &probe)
	if len(probe.Have) != 0 {
		t.Fatalf("the probe proved another lease's digest: have = %v", probe.Have)
	}
	if len(probe.Want) != 1 || probe.Want[0] != digest {
		t.Fatalf("want = %v, want exactly the unproved digest", probe.Want)
	}
	h.assertUntouched(t, "probing another lease's digest", before)

	// An unknown digest of the same shape answers identically, which is what
	// makes the answer carry no information (D50).
	unknown := sha256Of([]byte("nothing anyone ever uploaded"))
	resp = h.doLease(t, h.lease.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/blobs/probe", ProbeBody{Digests: []string{unknown}})
	var control struct {
		Have []string `json:"have"`
		Want []string `json:"want"`
	}
	decodeInto(t, resp, &control)
	if len(control.Have) != len(probe.Have) || len(control.Want) != len(probe.Want) {
		t.Fatalf("an out-of-set digest answers %+v and an unknown one %+v: the two are distinguishable",
			probe, control)
	}

	// The HEAD offset of a digest held under another lease is zero: this lease
	// has no part and no blob by that name.
	before = h.snapshot(t)
	head := h.raw(t, http.MethodHead, "/nashnet/leases/"+h.lease.LeaseID+"/blobs/"+digest, nil,
		map[string]string{nashnet.HeaderLeaseToken: h.lease.LeaseToken})
	if head.StatusCode != http.StatusOK {
		t.Fatalf("HEAD on another lease's digest: got %d", head.StatusCode)
	}
	if got := head.Header.Get(HeaderOffset); got != "0" {
		t.Fatalf("HEAD offset = %q for a digest this lease never uploaded, want 0", got)
	}
	head.Body.Close()
	h.assertUntouched(t, "HEAD on another lease's digest", before)

	// A commit naming a digest this lease never uploaded is refused with the
	// list of what it lacks, and nothing is promoted.
	before = h.snapshot(t)
	resp = h.doLease(t, h.lease.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/manifest", quarantine.CommitRequest{
			ManifestVersion: quarantine.ManifestVersion,
			LeaseEpoch:      h.lease.LeaseEpoch,
			Seq:             1,
			Entries: []quarantine.Entry{{
				Path: "stolen.pt", Digest: digest, Size: int64(len(secret)),
			}},
		})
	h.refusal(t, "committing a digest it never uploaded", resp,
		http.StatusConflict, nashnet.CodeBlobsMissing)
	h.assertUntouched(t, "committing a digest it never uploaded", before)
}

// ProbeBody is the blob-probe request shape. The node client spells the same
// body; this suite carries its own so a hostile request is never limited to
// what the production client will build.
type ProbeBody struct {
	Digests []string `json:"digests"`
}

// TestHostileNodeCannotReadASeedOutsideItsGrant covers the seed half of D53: a
// seed path the claim never granted is 404, and it is the same 404 an entry
// that does not exist gets, so the answer carries nothing.
func TestHostileNodeCannotReadASeedOutsideItsGrant(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	before := h.snapshot(t)

	paths := []string{
		"/nashnet/leases/" + h.lease.LeaseID + "/seeds/victim-job/snapshots/prtcfr_checkpoint.pt",
		"/nashnet/leases/" + h.lease.LeaseID + "/seeds/hostile-job/env.json",
		"/nashnet/leases/" + h.lease.LeaseID + "/seeds/no-such-run/no-such-file",
	}
	var bodies []string
	for _, p := range paths {
		resp := h.raw(t, http.MethodGet, p, nil,
			map[string]string{nashnet.HeaderLeaseToken: h.lease.LeaseToken})
		if resp.StatusCode != http.StatusNotFound {
			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			t.Fatalf("seed read %s: got %d, want 404 (%s)", p, resp.StatusCode, body)
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		bodies = append(bodies, string(body))
	}
	for i := 1; i < len(bodies); i++ {
		if bodies[i] != bodies[0] {
			t.Fatalf("the 404 bodies differ, so an out-of-set read is distinguishable:\n%q\n%q",
				bodies[0], bodies[i])
		}
	}
	h.assertUntouched(t, "seed reads outside the grant set", before)
}

// TestHostileNodeCannotEscapeItsRunDir covers the path shapes: every escape,
// every reserved name, and every unrepresentable path is refused per entry with
// its own reason, the rest of the manifest still commits, and nothing appears
// outside the run dir (D49, D52).
func TestHostileNodeCannotEscapeItsRunDir(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	body := []byte("payload the node would like somewhere it should not be\n")
	digest := h.uploadBlob(t, h.lease.LeaseToken, h.lease.LeaseID, body)

	cases := []struct {
		path   string
		reason string
	}{
		{"../escaped.pt", quarantine.ReasonPathTraversal},
		{"a/../../escaped.pt", quarantine.ReasonPathTraversal},
		{"/etc/passwd", quarantine.ReasonPathAbsolute},
		{"", quarantine.ReasonPathEmpty},
		{`windows\path.pt`, quarantine.ReasonPathBackslash},
		{"nul\x00byte.pt", quarantine.ReasonPathNUL},
		{"process.json", quarantine.ReasonPathReserved},
		{"jobspec.json", quarantine.ReasonPathReserved},
		{"lease.json", quarantine.ReasonPathReserved},
		{"env.json", quarantine.ReasonPathReserved},
		{".nashnet/current.json", quarantine.ReasonPathReserved},
		{"logs/training.log", quarantine.ReasonPathReserved},
		{strings.Repeat("deep/", 40) + "file.pt", quarantine.ReasonPathTooDeep},
		{strings.Repeat("n", 300) + ".pt", quarantine.ReasonSegmentTooLong},
	}

	before := h.snapshot(t)
	seq := int64(0)
	parent := ""
	for _, c := range cases {
		seq++
		entries := []quarantine.Entry{
			{Path: c.path, Digest: digest, Size: int64(len(body))},
			{Path: "kept/" + strconv.FormatInt(seq, 10) + ".pt", Digest: digest, Size: int64(len(body))},
		}
		resp := h.doLease(t, h.lease.LeaseToken, http.MethodPost,
			"/nashnet/leases/"+h.lease.LeaseID+"/manifest", quarantine.CommitRequest{
				ManifestVersion: quarantine.ManifestVersion,
				LeaseEpoch:      h.lease.LeaseEpoch,
				Seq:             seq,
				Parent:          parent,
				Entries:         entries,
			})
		if resp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			t.Fatalf("commit naming %q: got %d, want 200 with a per-entry rejection (%s)",
				c.path, resp.StatusCode, raw)
		}
		var out quarantine.CommitResponse
		decodeInto(t, resp, &out)
		parent = out.Digest

		found := ""
		for _, rej := range out.Rejected {
			if rej.Path == c.path {
				found = rej.Reason
			}
		}
		if found != c.reason {
			t.Fatalf("commit naming %q rejected it as %q, want %q (rejections: %+v)",
				c.path, found, c.reason, out.Rejected)
		}
		// The good entry beside it still promoted: one refused path does not
		// cost the commit.
		if len(out.Promoted) != 1 {
			t.Fatalf("commit naming %q promoted %v, want only the well-formed entry",
				c.path, out.Promoted)
		}
	}

	// Nothing the loop named exists anywhere the coordinator owns, and nothing
	// outside runs/<job>/kept/ was created at all.
	after := h.snapshot(t)
	for key := range after {
		if _, ok := before[key]; ok {
			continue
		}
		if strings.HasPrefix(key, "runs/hostile-job/kept/") ||
			strings.HasPrefix(key, "runs/hostile-job/.nashnet/") ||
			strings.HasPrefix(key, "quarantine/") {
			continue
		}
		t.Fatalf("a rejected path created %s", key)
	}
	for _, c := range cases {
		// A reserved path names a file the coordinator itself authors, so the
		// assertion there is that it did not change, which the snapshot
		// comparison above already made. What must not exist at all is
		// anything the escape shapes named.
		if c.path == "" || c.reason == quarantine.ReasonPathReserved || strings.ContainsRune(c.path, 0) {
			continue
		}
		abs := filepath.Join(h.runsDir, "hostile-job", filepath.FromSlash(c.path))
		if _, err := os.Lstat(abs); err == nil {
			t.Fatalf("the rejected path %q exists at %s", c.path, abs)
		}
	}
	if _, err := os.Lstat(filepath.Join(filepath.Dir(h.runsDir), "escaped.pt")); err == nil {
		t.Fatal("a traversal entry landed beside the runs dir")
	}
}

// TestHostileNodeCannotOfferAnIrregularFile covers the create-a-symlink line:
// the manifest names a path and a digest, and the coordinator materializes a
// regular file from a verified blob, so there is no manifest shape that asks
// for a symlink, a device node, or a directory. What a hostile node can do is
// name a path whose parent is a file, and that is refused per entry (D49).
func TestHostileNodeCannotOfferAnIrregularFile(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	body := []byte("a regular file's bytes\n")
	digest := h.uploadBlob(t, h.lease.LeaseToken, h.lease.LeaseID, body)

	// The file the second commit tries to descend into is promoted first, so
	// the validator sees a parent that already exists and is not a directory.
	first := h.doLease(t, h.lease.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/manifest", quarantine.CommitRequest{
			ManifestVersion: quarantine.ManifestVersion,
			LeaseEpoch:      h.lease.LeaseEpoch,
			Seq:             1,
			Entries:         []quarantine.Entry{{Path: "file.pt", Digest: digest, Size: int64(len(body))}},
		})
	if first.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(first.Body)
		first.Body.Close()
		t.Fatalf("first commit: got %d, want 200 (%s)", first.StatusCode, raw)
	}
	var head quarantine.CommitResponse
	decodeInto(t, first, &head)

	resp := h.doLease(t, h.lease.LeaseToken, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/manifest", quarantine.CommitRequest{
			ManifestVersion: quarantine.ManifestVersion,
			LeaseEpoch:      h.lease.LeaseEpoch,
			Seq:             2,
			Parent:          head.Digest,
			Entries: []quarantine.Entry{
				{Path: "file.pt/child.pt", Digest: digest, Size: int64(len(body))},
			},
		})
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("commit naming a path under a file: got %d, want 200 (%s)", resp.StatusCode, raw)
	}
	var out quarantine.CommitResponse
	decodeInto(t, resp, &out)
	if len(out.Rejected) != 1 || out.Rejected[0].Path != "file.pt/child.pt" {
		t.Fatalf("rejections = %+v, want the entry whose parent is a file", out.Rejected)
	}
	if out.Rejected[0].Reason != quarantine.ReasonPathParentNotDir {
		t.Fatalf("reason = %q, want %q", out.Rejected[0].Reason, quarantine.ReasonPathParentNotDir)
	}

	// Everything the commit did materialize is a regular file: nothing in the
	// run dir is a symlink, a device node, or a socket.
	err := filepath.WalkDir(filepath.Join(h.runsDir, "hostile-job"), func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		info, ierr := d.Info()
		if ierr != nil {
			return ierr
		}
		if !info.Mode().IsRegular() {
			t.Errorf("%s is not a regular file: %s", path, info.Mode())
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}

// TestHostileNodeCannotPromoteAForgedJournal covers the run_db line of D55: the
// journal is the one node-authored format the coordinator parses, and a copy
// carrying a second runs row, a name that is not this job's, or a value outside
// the enum is rejected rather than promoted. The corpus is the shared fixture
// set, so this suite and the validator's own cannot drift apart.
func TestHostileNodeCannotPromoteAForgedJournal(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	cases := []string{"second_row.sqlite", "wrong_name.sqlite", "out_of_enum.sqlite", "corrupt.sqlite"}

	seq := int64(0)
	parent := ""
	for _, fixture := range cases {
		body, err := os.ReadFile(filepath.Join("testdata", "rundb", fixture))
		if err != nil {
			t.Fatal(err)
		}
		digest := h.uploadBlob(t, h.lease.LeaseToken, h.lease.LeaseID, body)
		seq++
		before := h.snapshot(t)
		resp := h.doLease(t, h.lease.LeaseToken, http.MethodPost,
			"/nashnet/leases/"+h.lease.LeaseID+"/manifest", quarantine.CommitRequest{
				ManifestVersion: quarantine.ManifestVersion,
				LeaseEpoch:      h.lease.LeaseEpoch,
				Seq:             seq,
				Parent:          parent,
				Entries: []quarantine.Entry{{
					Path: quarantine.RunDBPath, Digest: digest, Size: int64(len(body)),
				}},
			})
		if resp.StatusCode != http.StatusOK {
			raw, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			t.Fatalf("commit of %s: got %d, want 200 with a rejection (%s)", fixture, resp.StatusCode, raw)
		}
		var out quarantine.CommitResponse
		decodeInto(t, resp, &out)
		parent = out.Digest
		if len(out.Rejected) != 1 || out.Rejected[0].Reason != quarantine.ReasonRunDBInvalid {
			t.Fatalf("commit of %s rejected %+v, want one rundb_invalid", fixture, out.Rejected)
		}
		if len(out.Promoted) != 0 {
			t.Fatalf("commit of %s promoted %v", fixture, out.Promoted)
		}
		if _, err := os.Stat(filepath.Join(h.runsDir, "hostile-job", quarantine.RunDBPath)); err == nil {
			t.Fatalf("the rejected journal %s reached the run dir", fixture)
		}
		_ = before
	}
}

// TestHostileNodeCannotSteerABundleBuild covers the mirror line: a claim's
// have_commits reaches a bundle build, so a basis entry that is not a 40-hex
// commit present in the mirror is refused inside BundleCreate before any
// subprocess runs, and the claim falls back to the shared full-tree bundle
// rather than failing (D48). The witnesses are the cache and the mirror: no
// basis-suffixed artifact is ever built, the mirror is byte-identical, and
// nothing a metacharacter would have created exists.
func TestHostileNodeCannotSteerABundleBuild(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	sha := h.queueFixtureJob(t, "basis-job", 1, "quick")
	marker := filepath.Join(t.TempDir(), "reached-a-shell")

	hostile := []string{
		"$(touch " + marker + ")",
		"`id`",
		"--upload-pack=/bin/sh",
		"../../../../etc/passwd",
		strings.Repeat("z", 40),
	}
	before := h.snapshot(t)
	for _, basis := range hostile {
		resp, claimed := h.claim(t, h.node, nashnet.ClaimRequest{HaveCommits: []string{basis}})
		resp.Body.Close()
		if h.repo.hasCachedBundle(t, sha, true) {
			t.Fatalf("a thin bundle was built for the hostile basis %q", basis)
		}
		if claimed == nil {
			continue
		}
		// Return the lease so the next attempt reaches the same job again.
		h.nack(t, claimed, nashnet.NackSnapshotFailed)
		h.clock.advance(2 * time.Second)
	}

	// The mirror is exactly as it was: no ref written, no object added.
	after := snapshotTrees(t, map[string]string{"mirror": h.repo.coordDir})
	for key, was := range before {
		if !strings.HasPrefix(key, "mirror/") {
			continue
		}
		if now, ok := after[key]; !ok || now != was {
			t.Fatalf("the mirror changed at %s while a hostile basis was refused", key)
		}
	}
	if _, err := os.Stat(marker); err == nil {
		t.Fatalf("a basis metacharacter reached a shell and created %s", marker)
	}
}

// TestHostileNodeCannotExceedItsQuotas covers the quota line of D56: the
// per-request chunk cap, the per-tick lease budget, and the per-call log cap
// each refuse over-budget bytes by name, and none of them lands a byte.
func TestHostileNodeCannotExceedItsQuotas(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{pool: poolRigConfig{ceilings: Ceilings{
		ChunkBytes:        1 << 10,
		LeaseBytesPerTick: 2 << 10,
		LogBytesPerCall:   1 << 10,
	}}})

	// A single chunk over the per-request cap.
	oversized := make([]byte, (1<<10)+1)
	for i := range oversized {
		oversized[i] = 'a'
	}
	before := h.snapshot(t)
	resp := h.patchChunkRaw(t, h.lease, sha256Of(oversized), oversized, 0, len(oversized))
	h.refusal(t, "a chunk over the per-request cap", resp,
		http.StatusRequestEntityTooLarge, nashnet.CodeOverCap)
	h.assertUntouched(t, "a chunk over the per-request cap", before)

	// Within the chunk cap but past the lease's budget for this tick.
	chunk := make([]byte, 1<<10)
	for i := range chunk {
		chunk[i] = 'b'
	}
	whole := append(append([]byte(nil), chunk...), append(append([]byte(nil), chunk...), chunk...)...)
	digest := sha256Of(whole)
	h.patchChunk(t, h.lease, digest, chunk, 0, len(whole))
	h.patchChunk(t, h.lease, digest, chunk, 1<<10, len(whole))
	resp = h.patchChunkRaw(t, h.lease, digest, chunk, 2<<10, len(whole))
	h.refusal(t, "a chunk past the lease's tick budget", resp,
		http.StatusRequestEntityTooLarge, nashnet.CodeOverCap)

	// A log append over the per-call cap.
	logBody := make([]byte, (1<<10)+64)
	for i := range logBody {
		logBody[i] = 'c'
	}
	before = h.snapshot(t)
	resp = h.raw(t, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/logs?offset=0", logBody,
		map[string]string{nashnet.HeaderLeaseToken: h.lease.LeaseToken})
	h.refusal(t, "a log append over the per-call cap", resp,
		http.StatusRequestEntityTooLarge, nashnet.CodeOverCap)
	h.assertUntouched(t, "a log append over the per-call cap", before)
}

// TestHostileNodeCannotImpersonateAnotherNode is the first two of the six
// review-derived cases: a register and a claim naming another enrolled node's
// node_id in the body are 403 wrong_subject, and the victim's record, its
// node_epoch, and its leases are exactly as they were.
func TestHostileNodeCannotImpersonateAnotherNode(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	victimEpoch := h.register(t, h.victim, 2)
	h.queueFixtureJob(t, "victim-job", 1, "quick")
	_, victimLease := h.claim(t, h.victim, nashnet.ClaimRequest{})
	if victimLease == nil {
		t.Fatal("the victim was handed no lease")
	}

	for _, a := range []struct {
		what string
		path string
		body any
	}{
		{"register naming the victim", "/nashnet/nodes/register",
			nashnet.RegisterRequest{NodeID: h.victim.id, Slots: 2, Capabilities: declaration(2),
				GateReport: admitReport(2)}},
		{"claim naming the victim", "/nashnet/claim",
			nashnet.ClaimRequest{NodeID: h.victim.id, SlotsFree: 2, Capabilities: declaration(2),
				GateReport: admitReport(2)}},
	} {
		before := h.snapshot(t)
		resp := h.doNode(t, h.node, http.MethodPost, a.path, a.body)
		h.refusal(t, a.what, resp, http.StatusForbidden, nashnet.CodeWrongSubject)
		h.assertUntouched(t, a.what, before)

		rec, ok := h.pool.nodes.Get(h.victim.id)
		if !ok {
			t.Fatalf("%s removed the victim's record", a.what)
		}
		if rec.NodeEpoch != victimEpoch {
			t.Fatalf("%s moved the victim's node_epoch to %d, want %d",
				a.what, rec.NodeEpoch, victimEpoch)
		}
		live := h.pool.leases.LiveForNode(h.victim.id)
		if len(live) != 1 || live[0].LeaseID != victimLease.LeaseID {
			t.Fatalf("%s changed the victim's leases: %+v", a.what, live)
		}
	}
}

// TestHostileNodeCannotBorrowTheCoordinatorsPGID is the third review-derived
// case: a progress post carrying a pid the coordinator would recognize as its
// own process group, followed by a cancel of that job, sends no signal. The
// projection pins PGID to zero precisely so the unsupervised stop branch cannot
// be steered into signalling an arbitrary group on the coordinator host.
func TestHostileNodeCannotBorrowTheCoordinatorsPGID(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})

	// A real child in its own process group stands in for whatever the node
	// would like signalled. If the cancel signalled the reported pid's group,
	// this process would die.
	bystanderPID := startBystander(t)

	h.progress(t, h.lease, nashnet.ProgressRequest{
		Phase: nashnet.PhaseRunning,
		PID:   bystanderPID,
	})
	st := readProcessState(t, h.runsDir, "hostile-job")
	if st.PGID != 0 {
		t.Fatalf("the projection recorded PGID %d from a node-supplied pid, want 0", st.PGID)
	}
	if st.PID != bystanderPID {
		t.Fatalf("projection PID = %d, want the reported %d", st.PID, bystanderPID)
	}

	h.cancelJob(t, "hostile-job", true)

	// The bystander is untouched: the cancel routed through the lease, and the
	// unsupervised signal branch was never reachable with a zero pgid.
	time.Sleep(200 * time.Millisecond)
	if !processAlive(bystanderPID) {
		t.Fatalf("the cancel signalled pid %d, which the node named in a progress post", bystanderPID)
	}
	after := readProcessState(t, h.runsDir, "hostile-job")
	if after.PGID != 0 {
		t.Fatalf("PGID = %d after the cancel, want 0", after.PGID)
	}
}

// TestHostileNodeCannotNameADigestWithASeparator is the fourth review-derived
// case: a blob digest carrying ".." or a path separator is 422 invalid_digest,
// and nothing exists outside the lease's own quarantine tree afterwards.
func TestHostileNodeCannotNameADigestWithASeparator(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	body := []byte("bytes aimed at a path\n")

	digests := []string{
		"..",
		"../../etc/passwd",
		strings.Repeat("a", 32) + "/" + strings.Repeat("b", 31),
		"..%2f..%2fetc",
		strings.Repeat("a", 63),
		strings.Repeat("G", 64),
	}
	for _, d := range digests {
		before := h.snapshot(t)
		resp := h.raw(t, http.MethodPatch,
			"/nashnet/leases/"+h.lease.LeaseID+"/blobs/"+d, body,
			map[string]string{
				nashnet.HeaderLeaseToken: h.lease.LeaseToken,
				"Content-Range":          fmt.Sprintf("bytes 0-%d/%d", len(body)-1, len(body)),
			})
		if resp.StatusCode == http.StatusNotFound || resp.StatusCode == http.StatusMovedPermanently {
			// A separator makes the request address a route that does not
			// exist, which is a refusal before any handler sees it.
			resp.Body.Close()
			h.assertUntouched(t, "a blob digest of "+d, before)
			continue
		}
		h.refusal(t, "a blob digest of "+d, resp,
			http.StatusUnprocessableEntity, nashnet.CodeInvalidDigest)
		h.assertUntouched(t, "a blob digest of "+d, before)
	}
}

// TestHostileNodeCannotHoldALeaseForever is the fifth review-derived case: a
// node that renews its lease forever without ever launching is revoked at the
// coordinator's lease lifetime cap, and the job it was holding returns to the
// queue rather than being consumed (D4, D7).
func TestHostileNodeCannotHoldALeaseForever(t *testing.T) {
	policy := nashnet.DefaultPolicy()
	policy.LeaseTTLSeconds = 120
	policy.MaxLeaseSeconds = 600
	h := newHostileNode(t, nodeRigConfig{pool: poolRigConfig{policy: policy}})
	sweeper := h.pool.Sweeper()

	// Renewed every minute, well inside the TTL, and never launched.
	var out []nashnet.Outcome
	for elapsed := time.Minute; elapsed <= 12*time.Minute; elapsed += time.Minute {
		h.clock.advance(time.Minute)
		if out = sweeper.Tick(); len(out) > 0 {
			if elapsed < 10*time.Minute {
				t.Fatalf("the lease ended at %s, inside the %ds cap: %+v",
					elapsed, policy.MaxLeaseSeconds, out)
			}
			break
		}
		h.progress(t, h.lease, nashnet.ProgressRequest{Phase: nashnet.PhasePreparing})
	}
	if len(out) != 1 || out[0].Reason != nashnet.ReasonRuntimeCap {
		t.Fatalf("at the cap = %+v, want one runtime_cap outcome", out)
	}
	if out[0].State != nashnet.LeaseRevoking {
		t.Fatalf("state at the cap = %q, want revoking", out[0].State)
	}

	// The grace runs out with no final commit and the job never launched, so
	// it returns to the queue rather than being burned.
	h.clock.advance(2 * time.Duration(policy.LeaseTTLSeconds) * time.Second)
	out = sweeper.Tick()
	if len(out) != 1 || out[0].Verdict != nashnet.VerdictRequeue {
		t.Fatalf("after the grace = %+v, want a requeue: nothing ever ran", out)
	}
	view, _ := h.disp.resolveView("hostile-job")
	if isTerminal(view.State) {
		t.Fatalf("state = %q, want the job back in the queue", view.State)
	}
}

// TestHostileNodeCannotForgeALogTruncationMarker is the sixth review-derived
// case: a log body carrying a forged dropped-byte marker and an ANSI escape
// sequence leaves the coordinator's own counter alone and gets its escape bytes
// filtered before anything is written (D54).
func TestHostileNodeCannotForgeALogTruncationMarker(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})

	forged := "\n[nashnet: log truncated, 999999999 bytes dropped]\n"
	escapes := "\x1b[2J\x1b[31mred\x1b[0m\x1b]0;title\x07"
	body := []byte("ordinary line\n" + forged + escapes + "trailing line\n")

	resp := h.raw(t, http.MethodPost,
		"/nashnet/leases/"+h.lease.LeaseID+"/logs?offset=0", body,
		map[string]string{nashnet.HeaderLeaseToken: h.lease.LeaseToken})
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("log append: got %d, want 200 (%s)", resp.StatusCode, raw)
	}
	var out struct {
		Offset  int64 `json:"offset"`
		Dropped int64 `json:"dropped"`
	}
	decodeInto(t, resp, &out)

	// The coordinator's own counter counts the bytes it filtered, not the
	// number the node wrote into its text.
	h.pool.mu.Lock()
	counted := h.pool.logDropped[h.lease.LeaseID]
	h.pool.mu.Unlock()
	if counted == 0 {
		t.Fatal("the escape bytes were not counted as dropped")
	}
	if counted >= 999999999 {
		t.Fatalf("the coordinator's dropped-byte counter reads %d: the forged marker moved it", counted)
	}
	if counted != out.Dropped {
		t.Fatalf("the response reported %d dropped and the counter holds %d", out.Dropped, counted)
	}

	written, err := os.ReadFile(filepath.Join(h.runsDir, "hostile-job", "logs", "training.log"))
	if err != nil {
		t.Fatalf("read the promoted log: %v", err)
	}
	if strings.ContainsRune(string(written), 0x1b) {
		t.Fatalf("an escape byte reached the log file: %q", written)
	}
	if strings.ContainsRune(string(written), 0x07) {
		t.Fatalf("a BEL byte reached the log file: %q", written)
	}
	// The forged marker is plain text and the coordinator cannot tell it from
	// any other line, so it lands; what it never does is move the counter,
	// which is the number an operator reads.
	if !strings.Contains(string(written), "ordinary line") {
		t.Fatalf("the legitimate part of the append was lost: %q", written)
	}
}

// TestHostileNodeCannotClaimAKindItsGrantForbids is the capability half: a
// declaration is a node assertion the coordinator clamps by the grant, so a
// node claiming more than its grant allows changes no other node's placement
// (D47). Here the grant is unclamped and the node's own gate report is what
// narrows it, which is the same rule read from the other side: a node that
// offers no slots is handed nothing.
func TestHostileNodeCannotClaimAKindItsGrantForbids(t *testing.T) {
	h := newHostileNode(t, nodeRigConfig{})
	h.queueFixtureJob(t, "gated-job", 1, "quick")

	before := h.snapshot(t)
	resp := h.doNode(t, h.node, http.MethodPost, "/nashnet/claim", nashnet.ClaimRequest{
		SlotsFree:    99,
		Capabilities: declaration(99),
		GateReport:   denyReport(h.clock.now().Add(time.Hour)),
	})
	if resp.StatusCode != http.StatusNoContent {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		t.Fatalf("a claim with a denying gate report: got %d, want 204 (%s)", resp.StatusCode, body)
	}
	if hold := holdReason(resp); hold != nashnet.HoldNodeGated {
		t.Fatalf("hold = %q, want node_gated", hold)
	}
	h.assertUntouched(t, "a claim with an inflated declaration and a denying gate", before)

	view, _ := h.disp.resolveView("gated-job")
	if view.State != StateQueued {
		t.Fatalf("gated-job = %q, want it still queued for a node that will take it", view.State)
	}
}

// startBystander forks a sleeping process into its own process group and
// returns its pid. It stands in for whatever a node would like the coordinator
// to signal on its behalf, and its own group means a signal aimed at it could
// not reach the test process by accident.
func startBystander(t *testing.T) int {
	t.Helper()
	cmd := exec.Command("sleep", "60")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})
	return cmd.Process.Pid
}
