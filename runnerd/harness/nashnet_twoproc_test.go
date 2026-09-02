package harness

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jason-s-yu/cambia/runnerd/ingest"
	"github.com/jason-s-yu/cambia/runnerd/ingestapi"
	"github.com/jason-s-yu/cambia/runnerd/nashnet"
	"github.com/jason-s-yu/cambia/runnerd/nashnet/gates"
	"github.com/jason-s-yu/cambia/runnerd/nodeagent"
	"github.com/jason-s-yu/cambia/runnerd/procmgr"
)

// This file is the two-process integration rig of D42 and the first acceptance
// of W4-T16: one Go test binary standing up a coordinator on loopback TLS and a
// node agent against it, with the production nodeagent.Client as the node's
// transport. Nothing here dials off the loopback interface, nothing here needs
// an accelerator, and the only external binary it runs is git, against
// repositories this file creates in temp directories.
//
// Three transport configurations exist in the codebase and they are not
// interchangeable. The pinned HTTPS client of D27 is the one a real node runs
// and the one this suite drives. The in-process loopback of D65 (cambia-1721)
// is the embedded node's transport and is covered once, by
// TestSchedulingOverBothTransports. The rig's own raw http.Client, inherited
// from poolRig, stands in for a hostile node: it is the only way to send a
// request the production client would refuse to build.
//
// The node agent runs in this process rather than in a forked one, so the two
// "processes" of the ticket name are two runnerd roles rather than two pids.
// What that costs is named at each scenario that would otherwise signal the
// node: a SIGSTOP or SIGKILL aimed at the node would take the coordinator with
// it, so those scenarios model the node's silence and its abrupt death instead,
// and the signal-level cases belong to the live battery of D43.

// nodeRigConfig configures a coordinator plus one node agent.
type nodeRigConfig struct {
	pool poolRigConfig
	// gates is the node's own gate config (D46). The zero value configures no
	// gate, which admits everything.
	gates gates.Config
	// slots is the node's declared concurrency.
	slots int
	// claimOnce stops the agent after one claim attempt and one events
	// response, which is how a scenario drives exactly one cycle without
	// racing a background loop.
	claimOnce bool
	// claimWaitSeconds is the node's requested long-poll hold. Zero makes both
	// long polls answer at once, which is what a one-shot cycle wants; a
	// scenario that keeps the agent running sets it so neither loop spins.
	claimWaitSeconds int
	// pollInterval is how often the agent re-reads a launched job's
	// process.json.
	pollInterval time.Duration
	// prepareErr, when set, is what the node's staging step returns.
	prepareErr error
	// noBundleFetch skips the real git import, for a scenario whose subject is
	// not snapshot delivery. The bundle is still served and downloaded.
	noBundleFetch bool
	// verbose sends the agent's own log to stderr. Off by default: a passing
	// scenario's log is noise, and a failing one is diagnosed by re-running
	// with it on.
	verbose bool
}

// nodeRig is a coordinator and one node agent wired to each other.
type nodeRig struct {
	*poolRig
	agent    *nodeagent.Agent
	nodeCfg  nodeagent.Config
	client   *nodeagent.Client
	env      *nodeEnv
	prober   *nodeProber
	nodePM   *procmgr.ProcessManager
	requests *requestLog
	repo     *gitFixture
	// embeddedEnv is the staging boundary of the coordinator's own node, set
	// when a scenario builds one.
	embeddedEnv *nodeEnv
	worktree    string
	// node is the fixture node this agent acts as: node-a of the pool rig, so
	// the coordinator admits it through the same operator-signed grant every
	// other suite uses.
	node fixtureNode
}

// newNodeRig builds the coordinator, the git fixture behind its snapshot
// route, and one node agent holding node-a's key.
func newNodeRig(t *testing.T, cfg nodeRigConfig) *nodeRig {
	t.Helper()
	repo := newGitFixture(t)

	if cfg.pool.policy.LeaseTTLSeconds == 0 {
		cfg.pool.policy = fastPoolPolicy()
	}
	cfg.pool.bundles = repo.coordinator
	requests := &requestLog{}
	cfg.pool.rig.wrap = requests.wrap

	pr := newPoolRig(t, cfg.pool)
	assertLoopbackOnly(t, pr.baseURL)

	base := t.TempDir()
	nodeRunsDir := filepath.Join(base, "runs")
	if err := os.MkdirAll(nodeRunsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	worktree := t.TempDir()
	writeIntegrationJob(t, worktree)

	slots := cfg.slots
	if slots == 0 {
		slots = 2
	}
	poll := cfg.pollInterval
	if poll == 0 {
		poll = 10 * time.Millisecond
	}

	nodeCfg := nodeagent.Config{
		Coordinator: nodeagent.Coordinator{
			URL:        pr.baseURL,
			CertSHA256: certFingerprint(t, pr),
		},
		KeyPath:          filepath.Join(base, "node.key"),
		BaseDir:          base,
		RunsDir:          nodeRunsDir,
		Gates:            cfg.gates,
		AgentVersion:     "1.1.0",
		PlatformTag:      "linux-x86_64",
		Slots:            slots,
		Kinds:            []string{KindTrain, KindEvaluate, KindMeasure, KindHeadToHead, KindBench},
		ClaimWaitSeconds: cfg.claimWaitSeconds,
		PythonBin:        "/bin/sh",
	}

	signer, err := nodeagent.NewSigner(pr.nodeA.priv, pr.clock.now)
	if err != nil {
		t.Fatalf("signer: %v", err)
	}
	if signer.NodeID() != pr.nodeA.id {
		t.Fatalf("signer id %s, rig node-a id %s", signer.NodeID(), pr.nodeA.id)
	}
	client, err := nodeagent.NewClient(nodeCfg, signer)
	if err != nil {
		t.Fatalf("node client: %v", err)
	}

	env := &nodeEnv{
		worktree:   worktree,
		runsDir:    nodeRunsDir,
		mirror:     repo.node,
		prepareErr: cfg.prepareErr,
		skipFetch:  cfg.noBundleFetch,
	}
	pm := procmgr.NewProcessManager(nodeRunsDir, "", "cambia", NewRunResolver(nodeRunsDir), nil)
	prober := newNodeProber()

	agent, err := nodeagent.New(nodeagent.Options{
		Config:            nodeCfg,
		Signer:            signer,
		Client:            client,
		Env:               env,
		Launcher:          nodeagent.NewLauncher(pm),
		Prober:            prober,
		Logger:            agentLogger(cfg.verbose),
		Now:               pr.clock.now,
		PollInterval:      poll,
		CanBuildLibcambia: true,
		ClaimOnce:         cfg.claimOnce,
	})
	if err != nil {
		t.Fatalf("node agent: %v", err)
	}
	t.Cleanup(pm.KillAll)

	return &nodeRig{
		poolRig: pr, agent: agent, nodeCfg: nodeCfg, client: client,
		env: env, prober: prober, nodePM: pm, requests: requests,
		repo: repo, worktree: worktree, node: pr.nodeA,
	}
}

// agentLogger is the node agent's log sink.
func agentLogger(verbose bool) *log.Logger {
	if verbose {
		return log.New(os.Stderr, "nashnet-node: ", log.Lmicroseconds)
	}
	return log.New(io.Discard, "", 0)
}

// fastPoolPolicy is the pool policy the two-process suite runs under: seconds
// where production uses minutes, so a real agent posts several progress ticks
// and several incremental commits inside one test.
func fastPoolPolicy() nashnet.Policy {
	p := nashnet.DefaultPolicy()
	p.LeaseTTLSeconds = 30
	p.ProgressIntervalSeconds = 1
	p.ChunkBytes = 1 << 12
	return p
}

// runCycle drives the agent through register, one events poll, one claim, and
// the whole lease that claim hands out, and returns when the result is posted.
// It is the one-shot driver of D42's scheduling scenarios: ClaimOnce stops both
// loops after one pass, so the cycle is deterministic rather than raced against
// a background claim.
func (r *nodeRig) runCycle(t *testing.T, timeout time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	if err := r.agent.Run(ctx); err != nil {
		t.Fatalf("agent run: %v", err)
	}
	if ctx.Err() != nil {
		t.Fatalf("agent run did not finish inside %s", timeout)
	}
}

// startAgent runs the agent's loops in the background and returns a stop
// function. It is what a scenario uses when the coordinator must act on a live
// node: a revocation delivered on the held events poll, a cancel, a drain.
func (r *nodeRig) startAgent(t *testing.T) func() {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = r.agent.Run(ctx)
	}()
	stopped := false
	return func() {
		if stopped {
			return
		}
		stopped = true
		cancel()
		select {
		case <-done:
		case <-time.After(30 * time.Second):
			t.Error("the node agent did not stop within 30s of its context being canceled")
		}
	}
}

// nodeRunDir is where the node's own copy of a job's artifacts lives.
func (r *nodeRig) nodeRunDir(job string) string {
	return filepath.Join(r.nodeCfg.RunsDir, job)
}

// coordRunDir is the promoted run dir on the coordinator.
func (r *nodeRig) coordRunDir(job string) string {
	return filepath.Join(r.runsDir, job)
}

// blobPath is where a lease's verified blob sits in the quarantine tree
// (D49): quarantine/<node>/<job>/<lease>/blobs/<digest>.
func (r *nodeRig) blobPath(job, lease, digest string) string {
	return filepath.Join(r.quarDir, r.node.id, job, lease, "blobs", digest)
}

// partPath is where a lease's in-flight upload sits, whose own size is the
// resume offset (D50).
func (r *nodeRig) partPath(job, lease, digest string) string {
	return filepath.Join(r.quarDir, r.node.id, job, lease, "parts", digest+".part")
}

// integrationSpec is the job shape every scheduling scenario queues: a measure job
// running the fixture shell script, which is the sleep-based job of D42. The
// script writes its artifacts into the run dir it is handed and exits.
func integrationSpec(name string, args ...string) JobSpec {
	return JobSpec{
		Kind:   KindMeasure,
		Name:   name,
		Script: integrationScript,
		Args:   args,
	}
}

// integrationScript is the worktree-relative path of the fixture job.
const integrationScript = "cfr/scripts/integration-job.sh"

// writeIntegrationJob writes the fixture job into a worktree. It takes the run
// dir as its first argument and a mode as its second: "quick" writes one round
// of artifacts and exits, "steps" writes a new file per second for the given
// number of seconds (which is what drives several incremental commits), and
// "sleep" blocks until it is signaled.
func writeIntegrationJob(t *testing.T, worktree string) {
	t.Helper()
	dir := filepath.Join(worktree, "cfr", "scripts")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := `run="$1"
mode="${2:-quick}"
n="${3:-3}"
mkdir -p "$run/snapshots"
printf '{"iter":0}\n' > "$run/metrics.jsonl"
case "$mode" in
  quick)
    printf 'weights-0' > "$run/snapshots/prtcfr_checkpoint.pt"
    printf '{"iteration": 1}\n' > "$run/resume_state.json"
    echo "job finished"
    ;;
  steps)
    i=1
    while [ "$i" -le "$n" ]; do
      printf 'weights-%s' "$i" > "$run/snapshots/prtcfr_checkpoint.pt"
      printf '{"iter":%s}\n' "$i" >> "$run/metrics.jsonl"
      echo "step $i"
      sleep 1
      i=$((i+1))
    done
    echo "job finished"
    ;;
  sleep)
    echo "sleeping"
    exec sleep 120
    ;;
esac
`
	path := filepath.Join(dir, "integration-job.sh")
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
}

// certFingerprint is the sha256 of the coordinator's leaf certificate in DER
// form, the value a node pins (D27). Reading it off the live listener is what
// makes the pin real rather than a constant both sides agree on.
func certFingerprint(t *testing.T, r *poolRig) string {
	t.Helper()
	cert := r.ts.Certificate()
	if cert == nil {
		t.Fatal("the test listener served no certificate")
	}
	sum := sha256.Sum256(cert.Raw)
	return hex.EncodeToString(sum[:])
}

// assertLoopbackOnly is the no-network half of AC1. The production
// nodeagent.Client builds its own http.Transport, so a test cannot inject a
// dialer into it; what it can assert is that the only address the node is ever
// given is a loopback IP literal, which needs neither DNS nor a route off the
// host. A suite that accidentally pointed a node at a name would fail here
// rather than in CI with the network namespace unavailable.
func assertLoopbackOnly(t *testing.T, rawURL string) {
	t.Helper()
	u, err := url.Parse(rawURL)
	if err != nil {
		t.Fatalf("coordinator url %q: %v", rawURL, err)
	}
	host, _, err := net.SplitHostPort(u.Host)
	if err != nil {
		host = u.Host
	}
	ip := net.ParseIP(host)
	if ip == nil {
		t.Fatalf("coordinator host %q is a name, not an IP literal: the suite would need DNS", host)
	}
	if !ip.IsLoopback() {
		t.Fatalf("coordinator host %s is not a loopback address", ip)
	}
}

// requestLog records every request the listener serves, which is the suite's
// observation point for what the node put on the wire: the Range header of a
// resumed download, the round trip a cancel arrives on, the offset a resumed
// upload restarted at.
type requestLog struct {
	mu     sync.Mutex
	rows   []requestRow
	before func(*http.Request)
}

type requestRow struct {
	Method       string
	Path         string
	Range        string
	ContentRange string
	Status       int
	// Hold is the claim hold the coordinator answered a 204 with, read off the
	// response header where the pool puts it (a 204 carries no body).
	Hold string
	At   time.Time
}

// rangeStart parses the first byte offset of a Range header, or -1.
func (r requestRow) rangeStart() int64 {
	if !strings.HasPrefix(r.Range, "bytes=") {
		return -1
	}
	spec := strings.TrimPrefix(r.Range, "bytes=")
	dash := strings.Index(spec, "-")
	if dash <= 0 {
		return -1
	}
	n, err := strconv.ParseInt(spec[:dash], 10, 64)
	if err != nil {
		return -1
	}
	return n
}

// chunkStart parses the first byte offset of a Content-Range header, or -1.
func (r requestRow) chunkStart() int64 {
	var start, end, total int64
	if _, err := fmt.Sscanf(r.ContentRange, "bytes %d-%d/%d", &start, &end, &total); err != nil {
		return -1
	}
	return start
}

func (l *requestLog) wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		l.mu.Lock()
		before := l.before
		l.mu.Unlock()
		if before != nil {
			before(r)
		}
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		row := requestRow{
			Method: r.Method, Path: r.URL.Path,
			Range: r.Header.Get("Range"), ContentRange: r.Header.Get("Content-Range"),
			At: time.Now(),
		}
		next.ServeHTTP(rec, r)
		row.Status = rec.status
		row.Hold = w.Header().Get(nashnet.HeaderClaimHold)
		l.mu.Lock()
		l.rows = append(l.rows, row)
		l.mu.Unlock()
	})
}

// interceptOnce installs a hook that runs before the named request reaches the
// coordinator, exactly once. It is how a scenario provokes a condition the
// coordinator answers legitimately but a test cannot otherwise time: a blob
// lost between its upload and the commit that names it, a coordinator restart
// arriving mid-upload.
func (l *requestLog) interceptOnce(method, pathSubstr string, hook func()) {
	var once sync.Once
	l.mu.Lock()
	l.before = func(r *http.Request) {
		if r.Method != method || !strings.Contains(r.URL.Path, pathSubstr) {
			return
		}
		once.Do(hook)
	}
	l.mu.Unlock()
}

// clearIntercept drops any installed hook.
func (l *requestLog) clearIntercept() {
	l.mu.Lock()
	l.before = nil
	l.mu.Unlock()
}

// statusRecorder captures the status code a handler wrote. It forwards Flush
// so the events and claim long polls still stream, and Hijack so the WS routes
// still upgrade.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written bool
}

func (s *statusRecorder) WriteHeader(code int) {
	if !s.written {
		s.status = code
		s.written = true
	}
	s.ResponseWriter.WriteHeader(code)
}

func (s *statusRecorder) Write(b []byte) (int, error) {
	s.written = true
	return s.ResponseWriter.Write(b)
}

func (s *statusRecorder) Flush() {
	if f, ok := s.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// snapshot returns a copy of the recorded rows.
func (l *requestLog) snapshot() []requestRow {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]requestRow(nil), l.rows...)
}

// forPath returns every recorded row whose path contains substr.
func (l *requestLog) forPath(substr string) []requestRow {
	var out []requestRow
	for _, row := range l.snapshot() {
		if strings.Contains(row.Path, substr) {
			out = append(out, row)
		}
	}
	return out
}

// reset drops the recorded rows, so a scenario asserts on the requests of one
// phase rather than on everything since registration.
func (l *requestLog) reset() {
	l.mu.Lock()
	l.rows = nil
	l.mu.Unlock()
}

// nodeEnv is the node's ingest boundary: a real mirror on the git path and a
// fake one everywhere else (D42). BundleFetch imports the coordinator-served
// bundle into a real bare repository with real git, so the snapshot scenarios
// prove the artifact rather than a byte count; Prepare stages nothing, because
// a uv venv and a libcambia build are neither offline nor free.
type nodeEnv struct {
	worktree   string
	runsDir    string
	mirror     *ingest.Manager
	prepareErr error
	skipFetch  bool

	mu        sync.Mutex
	fetched   []string
	prepared  int
	cleaned   int
	lastError error
	// onPrepare runs inside Prepare after the run dir exists, so a scenario
	// can seed a run dir or observe the moment staging completes.
	onPrepare func(runDir string) error
}

func (e *nodeEnv) BundleFetch(ctx context.Context, jobID, bundlePath string) error {
	e.mu.Lock()
	e.fetched = append(e.fetched, jobID)
	skip := e.skipFetch
	e.mu.Unlock()
	if skip {
		return nil
	}
	return e.mirror.BundleFetch(ctx, jobID, bundlePath)
}

func (e *nodeEnv) Prepare(_ context.Context, jobID, _, _, _, _, _ string, _ map[string]string) (*ingestapi.Prepared, error) {
	e.mu.Lock()
	e.prepared++
	err := e.prepareErr
	hook := e.onPrepare
	e.mu.Unlock()
	if err != nil {
		return nil, err
	}
	runDir := filepath.Join(e.runsDir, jobID)
	if err := os.MkdirAll(filepath.Join(runDir, "logs"), 0o755); err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(runDir, "env.json"),
		[]byte(`{"origin_host":"node-a"}`+"\n"), 0o644); err != nil {
		return nil, err
	}
	if hook != nil {
		if err := hook(runDir); err != nil {
			return nil, err
		}
	}
	return &ingestapi.Prepared{
		WorktreeDir: e.worktree,
		RunDir:      runDir,
		VenvPython:  "/bin/sh",
	}, nil
}

func (e *nodeEnv) Cleanup(string, bool) error {
	e.mu.Lock()
	e.cleaned++
	e.mu.Unlock()
	return nil
}

// fetchCount reports how many snapshot imports ran.
func (e *nodeEnv) fetchCount() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.fetched)
}

// setPrepareErr installs a staging failure for the next claim.
func (e *nodeEnv) setPrepareErr(err error) {
	e.mu.Lock()
	e.prepareErr = err
	e.mu.Unlock()
}

// nodeProber is the node's host measurement: fixed, accelerator-free numbers,
// so the gate evaluator and the declaration are the same on every machine that
// runs this suite and no GPU is touched.
type nodeProber struct {
	mu  sync.Mutex
	obs nodeagent.Observation
}

func newNodeProber() *nodeProber {
	ram := 64.0
	disk := 500.0
	load := 0.5
	return &nodeProber{obs: nodeagent.Observation{
		Cores:       8,
		RAMFreeGB:   &ram,
		RAMTotalGB:  &ram,
		DiskFreeGB:  &disk,
		DiskTotalGB: &disk,
		Load1:       &load,
		Devices:     []gates.DeviceSnapshot{{ID: "cpu", Kind: "cpu"}},
	}}
}

func (p *nodeProber) Observe(string) nodeagent.Observation {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.obs
}

func (p *nodeProber) setFreeRAM(v float64) {
	p.mu.Lock()
	p.obs.RAMFreeGB = &v
	p.mu.Unlock()
}

// gitFixture is the real repository behind the snapshot route: a source repo
// with three commits, a bare coordinator mirror the job refs are pushed into,
// and a node-side mirror the bundles are fetched into. Both mirrors are real
// bare repositories under temp dirs; git runs locally and reaches no network.
type gitFixture struct {
	src         string
	coordBase   string
	coordDir    string
	nodeDir     string
	coordinator *ingest.Manager
	node        *ingest.Manager
	commits     []string
}

func newGitFixture(t *testing.T) *gitFixture {
	t.Helper()
	requireGit(t)
	src := t.TempDir()
	runGitCmd(t, src, "init", "-q", "-b", "main")
	runGitCmd(t, src, "config", "user.email", "rig@coordinator.test")
	runGitCmd(t, src, "config", "user.name", "rig")
	runGitCmd(t, src, "config", "commit.gpgsign", "false")

	var commits []string
	for i := 0; i < 3; i++ {
		dir := filepath.Join(src, "cfr")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		name := filepath.Join(dir, fmt.Sprintf("commit-%d.txt", i))
		if err := os.WriteFile(name, []byte(fmt.Sprintf("commit %d payload\n", i)), 0o644); err != nil {
			t.Fatal(err)
		}
		runGitCmd(t, src, "add", "-A")
		runGitCmd(t, src, "commit", "-q", "-m", fmt.Sprintf("commit %d", i))
		commits = append(commits, strings.TrimSpace(runGitCmd(t, src, "rev-parse", "HEAD")))
	}

	coordBase := t.TempDir()
	nodeBase := t.TempDir()
	coordMirror := filepath.Join(coordBase, "mirror.git")
	nodeMirror := filepath.Join(nodeBase, "mirror.git")
	runGitCmd(t, "", "init", "--bare", "-q", coordMirror)

	f := &gitFixture{
		src: src, coordBase: coordBase, coordDir: coordMirror, nodeDir: nodeMirror, commits: commits,
		coordinator: ingest.New(ingest.Config{BaseDir: coordBase, RunsDir: t.TempDir()}),
		node:        ingest.New(ingest.Config{BaseDir: nodeBase, RunsDir: t.TempDir()}),
	}
	return f
}

// push publishes one commit under a job's ref in the coordinator mirror, which
// is the only way a job ref is ever authored: the client's submit-time push
// (mirror.go:30). commitIdx selects which of the fixture's three commits.
func (f *gitFixture) push(t *testing.T, jobID string, commitIdx int) string {
	t.Helper()
	sha := f.commits[commitIdx]
	runGitCmd(t, f.src, "push", "-q", f.coordDir, sha+":refs/harness/"+jobID)
	return sha
}

// nodeHas reports whether the node's mirror holds the commit, which is the
// assertion a snapshot delivery scenario ends on: the bytes became git objects,
// not just a file on disk.
func (f *gitFixture) nodeHas(t *testing.T, sha string) bool {
	t.Helper()
	cmd := exec.Command("git", "-C", f.nodeDir, "cat-file", "-e", sha+"^{commit}")
	return cmd.Run() == nil
}

// nodeRef reads the commit a job's ref resolves to in the node's mirror.
func (f *gitFixture) nodeRef(t *testing.T, jobID string) string {
	t.Helper()
	out, err := exec.Command("git", "-C", f.nodeDir, "rev-parse", "--verify", "--quiet",
		"refs/harness/"+jobID+"^{commit}").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// seedNodeMirror imports one commit into the node's mirror ahead of time,
// which is what makes a later claim ask for a thin bundle.
func (f *gitFixture) seedNodeMirror(t *testing.T, jobID string, commitIdx int) string {
	t.Helper()
	sha := f.commits[commitIdx]
	runGitCmd(t, "", "init", "--bare", "-q", f.nodeDir)
	runGitCmd(t, f.src, "push", "-q", f.nodeDir, sha+":refs/harness/"+jobID)
	return sha
}

// runGitCmd runs git in dir (or with no working directory when dir is empty)
// and fails the test on a non-zero exit. Every invocation is local: no remote,
// no submodule, no hook.
func runGitCmd(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	if dir != "" {
		cmd.Dir = dir
	}
	cmd.Env = append(os.Environ(),
		"GIT_CONFIG_NOSYSTEM=1",
		"GIT_TERMINAL_PROMPT=0",
		"HOME="+t.TempDir(),
		"GIT_AUTHOR_NAME=rig", "GIT_AUTHOR_EMAIL=rig@coordinator.test",
		"GIT_COMMITTER_NAME=rig", "GIT_COMMITTER_EMAIL=rig@coordinator.test",
	)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

// requireGit skips a scenario on a machine with no git. Nothing else in this
// suite needs a binary off the standard library.
func requireGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not on PATH: the snapshot scenarios need a real bundle")
	}
}

// waitUntil polls cond until it holds or the deadline passes, and fails with
// the given description otherwise. The suite has both an injected clock (the
// coordinator's grace and expiry math) and a real one (a forked job's exit), so
// a wall-clock wait is the honest tool for the second kind.
func waitUntil(t *testing.T, what string, timeout time.Duration, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out after %s waiting for %s", timeout, what)
}

// leaseIDFor returns the live lease this node holds on a job, waiting for the
// claim to land.
func (r *nodeRig) leaseIDFor(t *testing.T, job string) string {
	t.Helper()
	var id string
	waitUntil(t, "a lease on "+job, 20*time.Second, func() bool {
		for _, l := range r.pool.leases.LiveForNode(r.node.id) {
			if l.JobID == job {
				id = l.LeaseID
				return true
			}
		}
		return false
	})
	return id
}

// promotedBody reads a promoted file out of the coordinator's run dir.
func (r *nodeRig) promotedBody(t *testing.T, job, rel string) []byte {
	t.Helper()
	body, err := os.ReadFile(filepath.Join(r.coordRunDir(job), filepath.FromSlash(rel)))
	if err != nil {
		t.Fatalf("read promoted %s of %s: %v", rel, job, err)
	}
	return body
}

// commitCount reports how many manifests a lease committed, which is the
// incremental-commit assertion: the manifests directory holds one file per
// accepted commit (D49).
func (r *nodeRig) commitCount(t *testing.T, job, lease string) int {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(r.quarDir, r.node.id, job, lease, "manifests"))
	if err != nil {
		if os.IsNotExist(err) {
			return 0
		}
		t.Fatalf("read manifests of %s: %v", lease, err)
	}
	return len(entries)
}

// hasCachedBundle reports whether the coordinator built a bundle for the
// commit. The cache key distinguishes the two shapes (D48): a full-tree bundle
// is named for the commit alone, a thin one carries the basis digest as a
// suffix, so the filename is the witness that a claim's have_commits reached
// the build.
func (f *gitFixture) hasCachedBundle(t *testing.T, sha string, thin bool) bool {
	t.Helper()
	entries, err := os.ReadDir(filepath.Join(f.coordBase, "snapshots"))
	if err != nil {
		t.Fatalf("read the snapshot cache: %v", err)
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".bundle") {
			continue
		}
		stem := strings.TrimSuffix(name, ".bundle")
		if thin && strings.HasPrefix(stem, sha+"-") {
			return true
		}
		if !thin && stem == sha {
			return true
		}
	}
	return false
}

// lastLeaseFor names the most recent lease the node held on a job, read off
// the quarantine tree rather than the live set, so it resolves after the lease
// retired.
func (r *nodeRig) lastLeaseFor(t *testing.T, job string) string {
	t.Helper()
	dir := filepath.Join(r.quarDir, r.node.id, job)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("no quarantine tree for %s: %v", job, err)
	}
	best := ""
	for _, e := range entries {
		if e.IsDir() && e.Name() > best {
			best = e.Name()
		}
	}
	if best == "" {
		t.Fatalf("the quarantine tree for %s holds no lease", job)
	}
	return best
}

// queueFixtureJob pushes the named commit under the job's ref in the
// coordinator mirror and queues a measure job pinned to it, which is the shape
// every scheduling scenario starts from: a real ref, a real commit, and a job
// the placement scan can hand out.
func (r *nodeRig) queueFixtureJob(t *testing.T, name string, commitIdx int, mode string, extra ...string) string {
	t.Helper()
	return r.queueFixtureJobAt(t, name, commitIdx, r.nodeRunDir(name), mode, extra...)
}

// queueFixtureJobAt is queueFixtureJob with an explicit run dir, which the
// embedded leg needs: an in-place run writes into the coordinator's own runs
// dir rather than a node-local one (D40).
func (r *nodeRig) queueFixtureJobAt(t *testing.T, name string, commitIdx int, runDir, mode string, extra ...string) string {
	t.Helper()
	sha := r.repo.push(t, name, commitIdx)
	args := append([]string{runDir, mode}, extra...)
	spec := integrationSpec(name, args...)
	spec.Commit = sha
	r.queueJob(t, spec)
	return sha
}
