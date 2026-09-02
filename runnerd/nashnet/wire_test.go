package nashnet

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"
)

// roundTrip encodes in, decodes it back into a fresh value of the same type,
// and asserts the two are identical. Every section 1 message goes through it,
// so a tag typo or a field the decoder cannot see fails here rather than
// against a live node.
func roundTrip[T any](t *testing.T, in T) T {
	t.Helper()
	data, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal %T: %v", in, err)
	}
	var out T
	if err := json.Unmarshal(data, &out); err != nil {
		t.Fatalf("unmarshal %T: %v (%s)", in, err, data)
	}
	if !reflect.DeepEqual(in, out) {
		t.Fatalf("%T round trip changed the value\n in: %+v\nout: %+v\nwire: %s", in, in, out, data)
	}
	return out
}

// fieldNames returns the top-level JSON object keys of v.
func fieldNames(t *testing.T, v any) map[string]bool {
	t.Helper()
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(data, &m); err != nil {
		t.Fatalf("unmarshal to map: %v", err)
	}
	out := make(map[string]bool, len(m))
	for k := range m {
		out[k] = true
	}
	return out
}

func TestCodecRoundTripRegister(t *testing.T) {
	roundTrip(t, RegisterRequest{
		NodeID:       "node-a",
		AgentVersion: "1.1.0",
		PlatformTag:  "linux-x86_64",
		Slots:        2,
		Kinds:        []string{"train", "evaluate", "measure"},
		Capabilities: json.RawMessage(`{"schema":1,"slots":2}`),
		GateReport:   json.RawMessage(`{"admit":true,"slots_offered":2}`),
		HaveCommits:  []string{"0123456789abcdef0123456789abcdef01234567"},
		LiveLeases: []LiveLease{
			{LeaseID: "01ARYZ6S41ABCDEFGHJKMNPQRS", JobID: "v0.4-prtcfr-r13", TokenHash: "deadbeef"},
		},
	})
	roundTrip(t, RegisterResponse{
		NodeID:        "node-a",
		NodeEpoch:     3,
		Policy:        DefaultPolicy(),
		ReboundLeases: []string{"01ARYZ6S41ABCDEFGHJKMNPQRS"},
		RevokedLeases: []string{"01ARYZ6S41ZZZZZZZZZZZZZZZZ"},
		ServerTime:    "2026-09-01T12:00:00Z",
	})
}

func TestCodecRoundTripHeartbeatAndEvents(t *testing.T) {
	roundTrip(t, HeartbeatRequest{
		AgentVersion: "1.1.0",
		SlotsFree:    1,
		Capabilities: json.RawMessage(`{"schema":1}`),
		GateReport:   json.RawMessage(`{"admit":false}`),
		HaveCommits:  []string{"0123456789abcdef0123456789abcdef01234567"},
	})
	roundTrip(t, HeartbeatResponse{NodeEpoch: 3, Hold: HoldReasonBreaker, ServerTime: "2026-09-01T12:00:00Z"})
	roundTrip(t, EventsResponse{
		Events: []Event{
			{Type: EventRevoke, LeaseID: "01ARYZ6S41ABCDEFGHJKMNPQRS", Force: true},
			{Type: EventDrain, Hold: HoldReasonDrain},
		},
		NodeEpoch:  3,
		ServerTime: "2026-09-01T12:00:00Z",
	})
}

func TestCodecRoundTripClaim(t *testing.T) {
	roundTrip(t, ClaimRequest{
		NodeID:       "node-a",
		NodeEpoch:    3,
		AgentVersion: "1.1.0",
		SlotsFree:    2,
		BusyJobIDs:   []string{"v0.4-prtcfr-r12"},
		Capabilities: json.RawMessage(`{"schema":1}`),
		GateReport:   json.RawMessage(`{"admit":true}`),
		Kinds:        []string{"train"},
		HaveCommits:  []string{"0123456789abcdef0123456789abcdef01234567"},
		WaitSeconds:  25,
	})
	resp := ClaimResponse{
		JobID:         "v0.4-prtcfr-r13",
		LeaseID:       "01ARYZ6S41ABCDEFGHJKMNPQRS",
		LeaseEpoch:    7,
		LeaseDeadline: "2026-09-01T12:02:00Z",
		LeaseToken:    "dG9rZW4",
		Spec:          json.RawMessage(`{"kind":"train","name":"v0.4-prtcfr-r13"}`),
		Attempt:       1,
		Snapshot: SnapshotRef{
			URL:    "/nashnet/leases/01ARYZ6S41ABCDEFGHJKMNPQRS/snapshot",
			Commit: "0123456789abcdef0123456789abcdef01234567",
			SHA256: "abc123",
			Size:   41288312,
		},
		Seeds: []Seed{{
			SeedID:  "resume",
			Kind:    "run_dir",
			Entries: []SeedEntry{{Path: "snapshots/prtcfr_checkpoint.pt", Size: 3512320, SHA256: "def456"}},
		}},
		Policy: DefaultPolicy(),
	}
	roundTrip(t, resp)
	roundTrip(t, ClaimHold{RetryAfterSeconds: 5, Hold: HoldNoMatch})

	got := fieldNames(t, resp)
	for _, want := range []string{
		"job_id", "lease_id", "lease_epoch", "lease_deadline", "lease_token",
		"spec", "attempt", "snapshot", "seeds", "policy",
	} {
		if !got[want] {
			t.Errorf("claim response is missing the %q field of D2", want)
		}
	}
}

func TestCodecRoundTripProgressAndResult(t *testing.T) {
	req := ProgressRequest{
		LeaseEpoch:     7,
		Phase:          PhaseRunning,
		PID:            4242,
		StartedAt:      "2026-09-01T12:00:00Z",
		ManifestSeq:    12,
		ManifestDigest: "abc123",
		LogOffset:      8192,
		BytesUploaded:  1 << 20,
		RunDBRows:      RunDBRows{Checkpoints: 3, Evals: 1},
		GateReport:     json.RawMessage(`{"admit":true}`),
	}
	roundTrip(t, req)
	roundTrip(t, ProgressResponse{
		Revoke: true, Force: false, Hold: HoldReasonDrain,
		LeaseDeadline: "2026-09-01T12:02:00Z", RetryAfterSeconds: 30,
	})

	got := fieldNames(t, req)
	for _, want := range []string{
		"lease_epoch", "phase", "pid", "started_at", "manifest_seq",
		"manifest_digest", "log_offset", "bytes_uploaded", "rundb_rows", "gate_report",
	} {
		if !got[want] {
			t.Errorf("progress request is missing the %q field of D5", want)
		}
	}

	exit := 0
	roundTrip(t, ResultRequest{
		LeaseEpoch: 7, State: ResultStopped, ExitCode: &exit,
		FinishedAt: "2026-09-01T13:00:00Z", FinalManifestDigest: "abc123", Attempt: 1,
	})
	roundTrip(t, ResultResponse{
		JobID: "v0.4-prtcfr-r13", LeaseID: "01ARYZ6S41ABCDEFGHJKMNPQRS",
		LeaseEpoch: 7, State: ResultStopped, ExitCode: &exit, RecordedAt: "2026-09-01T13:00:01Z",
	})
}

func TestCodecRoundTripNackDrainAndError(t *testing.T) {
	roundTrip(t, NackRequest{
		LeaseEpoch: 7, Reason: NackGateBreach, CooldownSeconds: 300, Detail: "outside 22:00-07:00",
	})
	roundTrip(t, DrainRequest{Drain: true, ClearBreaker: true})
	roundTrip(t, ErrorBody{
		Error: CodeOffsetMismatch, Detail: "seek to the true offset", Offset: 4096,
	})
	roundTrip(t, ErrorBody{Error: CodeRateLimited, RetryAfterSeconds: 30})
}

func TestCodecRoundTripLeaseRecord(t *testing.T) {
	l := Lease{
		JobID: "v0.4-prtcfr-r13", NodeID: "node-a", NodeEpoch: 3,
		LeaseID: "01ARYZ6S41ABCDEFGHJKMNPQRS", LeaseEpoch: 7, State: LeaseActive,
		TokenHash: HashLeaseToken("token"),
		GrantedAt: mustTime(t, "2026-09-01T12:00:00Z"),
		Deadline:  mustTime(t, "2026-09-01T12:02:00Z"),
		Attempt:   1,
		GrantSet: GrantSet{
			Snapshot: "abc123",
			Seeds:    map[string][]GrantEntry{"resume": {{Path: "snapshots/ck.pt", SHA256: "def456"}}},
		},
		Phase: PhaseRunning, PIDProjected: true, MaxRuntime: 48 * time.Hour,
	}
	f := l.toFile()
	got, err := f.toLease()
	if err != nil {
		t.Fatalf("toLease: %v", err)
	}
	if !reflect.DeepEqual(l, *got) {
		t.Fatalf("lease record round trip changed the value\n in: %+v\nout: %+v", l, *got)
	}

	names := fieldNames(t, f)
	for _, want := range []string{
		"job_id", "node_id", "node_epoch", "lease_id", "lease_epoch", "state",
		"token_hash", "granted_at", "deadline", "stop_requested_at", "attempt",
		"grant_set", "manifest_seq", "manifest_digest",
	} {
		if !names[want] {
			t.Errorf("lease record is missing the %q field of D4", want)
		}
	}
}

func TestDefaultPolicyMatchesTheBriefsNumbers(t *testing.T) {
	p := DefaultPolicy()
	cases := []struct {
		name string
		got  int64
		want int64
	}{
		{"lease_ttl_seconds", int64(p.LeaseTTLSeconds), 120},
		{"max_lease_seconds", int64(p.MaxLeaseSeconds), 259200},
		{"progress_interval_seconds", int64(p.ProgressIntervalSeconds), 30},
		{"chunk_bytes", p.ChunkBytes, 8388608},
		{"max_file_bytes", p.MaxFileBytes, 8589934592},
		{"max_lease_bytes", p.MaxLeaseBytes, 68719476736},
		{"log_bytes_per_call", p.LogBytesPerCall, 2097152},
	}
	for _, c := range cases {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}
}

func TestValidateNodePhaseRefusesStopping(t *testing.T) {
	for _, p := range []string{PhaseClaimed, PhaseFetching, PhasePreparing, PhaseRunning, PhaseUploading, PhaseCommitting} {
		if err := ValidateNodePhase(p); err != nil {
			t.Errorf("phase %q should be reportable by a node: %v", p, err)
		}
	}
	for _, p := range []string{PhaseStopping, "", "finished", "STOPPING"} {
		if err := ValidateNodePhase(p); err == nil {
			t.Errorf("phase %q is outside the node vocabulary and must be refused", p)
		}
	}
}
