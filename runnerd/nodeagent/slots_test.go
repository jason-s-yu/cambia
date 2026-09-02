package nodeagent

import (
	"testing"
	"time"
)

// TestSlotsAdmission is the relocated white-box unit that pinned the
// dispatcher's canLaunchLocked before the accounting moved here with the
// launch path it guards (D1). Its assertions are rewritten for the per-node
// exclusivity of D13: the rules are the same three, but they now describe one
// node rather than the whole daemon, which is what makes a second node in the
// pool free to run while this one is held.
func TestSlotsAdmission(t *testing.T) {
	s := NewSlots(3)

	if !s.Admit(true) {
		t.Fatal("an exclusive job should launch into an idle node")
	}
	s.Claim(false)
	if s.Admit(true) {
		t.Fatal("an exclusive job must not launch onto an occupied node")
	}
	if !s.Admit(false) {
		t.Fatal("a normal job should launch with a free slot (1<3)")
	}

	s.Release(false)
	s.Claim(true)
	if s.Admit(false) {
		t.Fatal("a normal job must not launch while an exclusive job holds the node")
	}
	if s.Admit(true) {
		t.Fatal("a second exclusive job must not launch while one holds the node")
	}
	// While the node is held it offers no capacity at all, so a claim never
	// advertises a slot the launch would refuse.
	if free := s.Free(); free != 0 {
		t.Fatalf("Free() = %d while an exclusive job holds the node, want 0", free)
	}

	s.Release(true)
	if got := s.ExclusiveHolds(); got != 0 {
		t.Fatalf("ExclusiveHolds() = %d after the exclusive job released, want 0", got)
	}
	for i := 0; i < 3; i++ {
		s.Claim(false)
	}
	if s.Admit(false) {
		t.Fatal("a normal job must not launch when the node is full (3>=3)")
	}
	if got := s.Active(); got != 3 {
		t.Fatalf("Active() = %d, want 3", got)
	}
	if free := s.Free(); free != 0 {
		t.Fatalf("Free() = %d on a full node, want 0", free)
	}
}

// TestSlotsUnlimitedCeiling pins the maxJobs<=0 spelling the dispatcher and a
// node config share: no ceiling means a normal job always admits, while the
// exclusive rules still hold.
func TestSlotsUnlimitedCeiling(t *testing.T) {
	s := NewSlots(0)
	for i := 0; i < 5; i++ {
		if !s.Admit(false) {
			t.Fatalf("normal job %d refused under an unlimited ceiling", i)
		}
		s.Claim(false)
	}
	if s.Admit(true) {
		t.Fatal("an exclusive job must not launch onto an occupied node, ceiling or not")
	}
}

// TestSlotsHoldsUntilLastExclusiveReleases guards the defensive multi-hold
// case the count exists for: two exclusive holds (a reattach that adopted more
// than one) keep the node held until the LAST one exits, where a flag would
// have opened it after the first.
func TestSlotsHoldsUntilLastExclusiveReleases(t *testing.T) {
	s := NewSlots(2)
	s.Claim(true)
	s.Claim(true)
	s.Release(true)
	if s.Admit(false) {
		t.Fatal("the node opened while a second exclusive hold still stood")
	}
	s.Release(true)
	if !s.Admit(false) {
		t.Fatal("the node stayed held after the last exclusive job released")
	}
}

// stubLauncher answers AwaitReattachedExit's polls from a scripted sequence of
// statuses, so the watch loop is driven with no process and no sleep.
type stubLauncher struct {
	statuses []ProcessStatus
	calls    int
}

func (s *stubLauncher) Ensure(name, algorithm string) error { return nil }
func (s *stubLauncher) Start(name string, l Launch) (int, error) {
	return 0, nil
}
func (s *stubLauncher) Stop(name string, force bool) error { return nil }
func (s *stubLauncher) Status(name string) ProcessStatus {
	i := s.calls
	s.calls++
	if i >= len(s.statuses) {
		i = len(s.statuses) - 1
	}
	return s.statuses[i]
}

// TestAwaitReattachedExit pins the relocated watch loop: it returns on a
// terminal EFFECTIVE status (a reattached row stays `running` on disk until
// its pid dies, so keying on the raw status would never return) and on a run
// dir that vanished under it.
func TestAwaitReattachedExit(t *testing.T) {
	live := ProcessStatus{Found: true, Status: "running", Effective: "running"}

	exited := &stubLauncher{statuses: []ProcessStatus{
		live, live,
		{Found: true, Status: "running", Effective: "crashed"},
	}}
	AwaitReattachedExit(exited, "adopted", time.Millisecond, nil)
	if exited.calls < 3 {
		t.Fatalf("watch returned after %d polls, want it to wait for the effective terminal", exited.calls)
	}

	purged := &stubLauncher{statuses: []ProcessStatus{live, {}}}
	AwaitReattachedExit(purged, "adopted", time.Millisecond, nil)
	if purged.calls != 2 {
		t.Fatalf("watch polled %d times past a purged run dir, want 2", purged.calls)
	}

	stop := make(chan struct{})
	close(stop)
	held := &stubLauncher{statuses: []ProcessStatus{live}}
	AwaitReattachedExit(held, "adopted", time.Hour, stop)
	if held.calls != 1 {
		t.Fatalf("watch polled %d times after its stop channel closed, want 1", held.calls)
	}
}
