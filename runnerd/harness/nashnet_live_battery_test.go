package harness

import (
	"os"
	"testing"
)

// The live battery of D43 and the fidelity check of W4-T16 AC5. Neither runs
// here. Both need two enrolled nodes with different declarations on real hosts,
// which means a deploy, and every runner deploy has been gated on an explicit
// user go; the offline half of this ticket (AC1 to AC3) is what runs in CI.
//
// This file exists so the battery is a checklist in the repository rather than
// a paragraph in a design document, and so a run of the suite says out loud
// that the live half has not run. Each scenario below is one line of D43. When
// the go arrives, the operator runs them against the deployed coordinator and
// records the per-scenario verdicts as a hub note, exactly as the M5 battery
// gated v1.0.
//
// The gate is the environment variable named below rather than a build tag, so
// the checklist is visible in a normal test run and the battery is opt-in by an
// operator who has the two nodes rather than by a compile-time decision.

// liveBatteryEnv opts a run into the live battery. Nothing in this file
// executes without it, and nothing in it can execute without two enrolled
// remote nodes.
const liveBatteryEnv = "CAMBIA_NASHNET_LIVE_BATTERY"

// liveScenario is one line of the D43 battery: what the operator does and what
// verdict the hub note records.
type liveScenario struct {
	name string
	// does is what the operator performs on the deployed pool.
	does string
	// verdict is what makes the scenario green.
	verdict string
}

// d43Battery is the whole battery, in the order D43 states it. The two nodes
// are declared only by their declarations: one CPU-only, one with an
// accelerator, with different gate configs including a time window that closes
// mid-run.
func d43Battery() []liveScenario {
	return []liveScenario{
		{
			name:    "golden path per declared capability",
			does:    "submit one job per kind each node declares and let it run to a clean terminal",
			verdict: "each job runs on a node whose declaration matches it and promotes its artifacts",
		},
		{
			name:    "three-node DAG with a deliberate parent failure per policy",
			does:    "submit a fan-in whose parent fails, once per on_failure policy (skip, run, fail)",
			verdict: "the dependent is skipped, run, or failed exactly as its policy names",
		},
		{
			name:    "kill -9 of a node agent mid-upload",
			does:    "SIGKILL the agent while a blob is in flight, then let it restart",
			verdict: "the restarted agent reattaches, resumes at the coordinator's part offset, and the job finishes",
		},
		{
			name:    "node host restart mid-job",
			does:    "reboot the node host while it supervises a job",
			verdict: "the lease expires or is re-bound per D7, and the job reaches a terminal without operator action",
		},
		{
			name:    "coordinator systemctl restart with two live leases and one in-flight upload",
			does:    "restart runnerd on the coordinator while two nodes hold leases and one is uploading",
			verdict: "both leases survive, the upload resumes from the part size, and neither node re-claims",
		},
		{
			name:    "lease expiry requeue of a pre-launch job",
			does:    "hold a node before launch until its lease expires",
			verdict: "the job returns to the queue at its original attempt and another node takes it",
		},
		{
			name:    "revocation of a node holding a live lease",
			does:    "revoke a node from the operator API while it runs a job",
			verdict: "the node stops its process group within one events round trip and keeps no right afterwards",
		},
		{
			name:    "an exclusive measure job on a busy node",
			does:    "submit an exclusive measure job to a node already running work",
			verdict: "the job waits for the node to drain to one slot and no second job launches beside it",
		},
		{
			name:    "purge with pending dependents",
			does:    "purge a job that other queued jobs name in their after list",
			verdict: "every lease tree for the job is removed and the dependents route through on_failure",
		},
		{
			name:    "resume of a PRT-CFR train job pinned to its reservoir node",
			does:    "resume a train job whose reservoir lives on one node, with that node available",
			verdict: "the job is placed back on the node that holds the reservoir and runs there (D12)",
		},
		{
			name:    "the same resume with the reservoir node drained",
			does:    "drain the node holding the reservoir, then resume the same job",
			verdict: "the job holds as unplaceable with reason reservoir_unavailable rather than being placed elsewhere",
		},
		{
			name:    "resume of a job with no reservoir onto the other node",
			does:    "resume a job whose inputs are all promoted artifacts, with its prior node drained",
			verdict: "the job is re-placed onto the other node and runs from served seeds",
		},
		{
			name:    "bytes per progress tick on a real PRT-CFR train job",
			does:    "measure the bytes the coordinator's listener moves per progress tick on a real train run",
			verdict: "the measurement is recorded in the hub note before the train scenario is declared green",
		},
	}
}

// TestD43LiveBatteryIsDeclaredNotRun is the standing record of AC4: the live
// battery is a named checklist here and a hub note there, and it does not run
// offline. Without the opt-in it skips, naming every scenario, so a CI run says
// which half of W4-T16 it covered.
func TestD43LiveBatteryIsDeclaredNotRun(t *testing.T) {
	battery := d43Battery()
	if len(battery) != 13 {
		t.Fatalf("the battery lists %d scenarios, want the 13 of D43", len(battery))
	}
	for _, s := range battery {
		if s.name == "" || s.does == "" || s.verdict == "" {
			t.Fatalf("scenario %q is not fully declared: %+v", s.name, s)
		}
	}
	if os.Getenv(liveBatteryEnv) == "" {
		for _, s := range battery {
			t.Logf("live scenario (not run): %s -> %s", s.name, s.verdict)
		}
		t.Skipf("the D43 live battery needs two enrolled remote nodes and a deploy; "+
			"set %s once they exist and an operator has given the go", liveBatteryEnv)
	}
	t.Fatalf("%s is set, but the live battery is executed by an operator against the "+
		"deployed pool and recorded as a hub note, not by this test binary", liveBatteryEnv)
}

// TestFidelityCheckIsDeclaredNotRun is the standing record of AC5. The check
// compares the client's cambia_runs.db against both nodes' journals and the
// coordinator's runs dir against its own manifests, and it is scoped to the two
// enrolled remote nodes: on the embedded node the run writes its own dir in
// place (D40), so the "coordinator is the only writer" property does not hold
// there and is not asserted.
//
// It cannot run offline for two reasons that are not about effort. The client
// side reads a cambia_runs.db the reconciler built from a real pull over ssh,
// which is the client's own trust root and has no node-side equivalent; and the
// node side reads each node's own run_db.sqlite off that host.
func TestFidelityCheckIsDeclaredNotRun(t *testing.T) {
	checks := []string{
		"every run row in the client's cambia_runs.db has a matching row in the journal of the node that executed it",
		"every runs.executed_on the client replayed names a node the coordinator actually placed the job on",
		"the coordinator's runs/ holds no file it did not author or promote through a manifest",
		"every promoted path in a lease's receipt exists in the run dir with the digest the manifest named",
	}
	for _, c := range checks {
		t.Logf("fidelity check (not run): %s", c)
	}
	t.Skipf("the fidelity check reads two remote nodes' journals and a pulled client "+
		"cambia_runs.db; it runs with the D43 battery under %s", liveBatteryEnv)
}
