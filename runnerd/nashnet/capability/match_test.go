package capability

import (
	"testing"
)

// gpuNode ("node-a") is a mixed cpu+cuda node with every device fact
// declared. cpuOnlyNode ("node-b") is a plain cpu-only node, used for the
// device-unavailable and undeclared-field cases.
func gpuNode() Candidate {
	return Candidate{
		NodeID: "node-a",
		Declaration: Declaration{
			Schema: 1,
			Slots:  2,
			Kinds:  []string{"train", "evaluate", "measure"},
			Devices: []Device{
				{ID: "cpu", Kind: "cpu", Cores: i(32), RAMTotalGB: f(64)},
				{ID: "cuda:0", Kind: "cuda", VRAMTotalGB: f(24)},
			},
			DiskTotalGB:       f(900),
			CanBuildLibcambia: true,
		},
	}
}

func cpuOnlyNode() Candidate {
	return Candidate{
		NodeID: "node-b",
		Declaration: Declaration{
			Schema: 1,
			Slots:  4,
			Kinds:  []string{"evaluate", "measure"},
			Devices: []Device{
				{ID: "cpu", Kind: "cpu", Cores: i(8)}, // no RAMTotalGB declared
			},
			// no DiskTotalGB declared
			CanBuildLibcambia: false,
		},
	}
}

// TestMatchTable exercises every Match constraint, including
// capability_undeclared, over the two named fixture nodes.
func TestMatchTable(t *testing.T) {
	cases := []struct {
		name       string
		kind       string
		requires   Requires
		candidate  Candidate
		wantOK     bool
		wantReason Reason
	}{
		{
			name:      "kind supported and cpu satisfied",
			kind:      "train",
			requires:  Requires{}.Normalize("cpu"),
			candidate: gpuNode(),
			wantOK:    true,
		},
		{
			name:       "kind not in clamped kinds list",
			kind:       "bench",
			requires:   Requires{}.Normalize("cpu"),
			candidate:  gpuNode(),
			wantOK:     false,
			wantReason: ReasonKindUnsupported,
		},
		{
			name:       "device unavailable on cpu-only node",
			kind:       "evaluate",
			requires:   Requires{Device: "cuda"}.Normalize("cuda"),
			candidate:  cpuOnlyNode(),
			wantOK:     false,
			wantReason: ReasonDeviceUnavailable,
		},
		{
			name:      "min_vram_gb satisfied",
			kind:      "train",
			requires:  Requires{Device: "cuda", MinVRAMGB: f(16)}.Normalize("cuda"),
			candidate: gpuNode(),
			wantOK:    true,
		},
		{
			name:       "min_vram_gb below floor",
			kind:       "train",
			requires:   Requires{Device: "cuda", MinVRAMGB: f(48)}.Normalize("cuda"),
			candidate:  gpuNode(),
			wantOK:     false,
			wantReason: ReasonMinVRAMGB,
		},
		{
			name:      "min_cores satisfied",
			kind:      "evaluate",
			requires:  Requires{MinCores: 8}.Normalize("cpu"),
			candidate: gpuNode(),
			wantOK:    true,
		},
		{
			name:       "min_cores below floor",
			kind:       "evaluate",
			requires:   Requires{MinCores: 16}.Normalize("cpu"),
			candidate:  cpuOnlyNode(),
			wantOK:     false,
			wantReason: ReasonMinCores,
		},
		{
			name:      "min_ram_gb satisfied",
			kind:      "train",
			requires:  Requires{MinRAMGB: f(32)}.Normalize("cpu"),
			candidate: gpuNode(),
			wantOK:    true,
		},
		{
			name:       "min_ram_gb undeclared on node-b",
			kind:       "evaluate",
			requires:   Requires{MinRAMGB: f(8)}.Normalize("cpu"),
			candidate:  cpuOnlyNode(),
			wantOK:     false,
			wantReason: ReasonCapabilityUndeclared,
		},
		{
			name:      "min_disk_gb satisfied",
			kind:      "train",
			requires:  Requires{MinDiskGB: f(100)}.Normalize("cpu"),
			candidate: gpuNode(),
			wantOK:    true,
		},
		{
			name:       "min_disk_gb undeclared on node-b",
			kind:       "evaluate",
			requires:   Requires{MinDiskGB: f(20)}.Normalize("cpu"),
			candidate:  cpuOnlyNode(),
			wantOK:     false,
			wantReason: ReasonCapabilityUndeclared,
		},
		{
			name:      "min_vram_gb undeclared on a device that never reports vram",
			kind:      "measure",
			requires:  Requires{Device: "xpu", MinVRAMGB: f(4)}.Normalize("xpu"),
			candidate: Candidate{NodeID: "node-b", Declaration: Declaration{Kinds: []string{"measure"}, Devices: []Device{{ID: "xpu:0", Kind: "xpu"}}}},
			wantOK:    false,
			// The xpu device exists but declares no vram_total_gb (AC6): the
			// undeclared-field reason wins over device_unavailable.
			wantReason: ReasonCapabilityUndeclared,
		},
		{
			name:      "needs_libcambia default true fails on a no-cc node",
			kind:      "evaluate",
			requires:  Requires{}.Normalize("cpu"),
			candidate: cpuOnlyNode(),
			wantOK:    false,
			// cpuOnlyNode declares CanBuildLibcambia:false (AC7's fixture).
			wantReason: ReasonNeedsLibcambia,
		},
		{
			name: "needs_libcambia false skips the check",
			kind: "evaluate",
			requires: func() Requires {
				no := false
				return Requires{NeedsLibcambia: &no}.Normalize("cpu")
			}(),
			candidate: cpuOnlyNode(),
			wantOK:    true,
		},
		{
			name:      "node pin matches",
			kind:      "train",
			requires:  Requires{Node: "node-a"}.Normalize("cpu"),
			candidate: gpuNode(),
			wantOK:    true,
		},
		{
			name:       "node pin mismatches",
			kind:       "train",
			requires:   Requires{Node: "node-a"}.Normalize("cpu"),
			candidate:  cpuOnlyNode(),
			wantOK:     false,
			wantReason: ReasonNodeMismatch,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			ok, reasons := Match(c.kind, c.requires, c.candidate)
			if ok != c.wantOK {
				t.Fatalf("Match ok = %v, want %v (reasons %v)", ok, c.wantOK, reasons)
			}
			if !c.wantOK {
				found := false
				for _, r := range reasons {
					if r == c.wantReason {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("reasons %v do not include %v", reasons, c.wantReason)
				}
			}
		})
	}
}

// TestMatchLabelsAnyAgainstGrantLabels covers labels_any matched against the
// grant's labels with the node's self-declared labels ignored (D47), the one
// case TestClampInflationIsInert does not isolate on its own.
func TestMatchLabelsAnyAgainstGrantLabels(t *testing.T) {
	cand := gpuNode()
	cand.GrantLabels = []string{"interactive"}
	cand.Declaration.Labels = nil // Clamp always drops this; Match must never read it anyway

	ok, reasons := Match("train", Requires{LabelsAny: []string{"interactive"}}.Normalize("cpu"), cand)
	if !ok {
		t.Fatalf("expected a match against grant labels, got reasons %v", reasons)
	}

	ok, reasons = Match("train", Requires{LabelsAny: []string{"trusted"}}.Normalize("cpu"), cand)
	if ok {
		t.Fatalf("expected no match: grant does not carry the trusted label")
	}
	found := false
	for _, r := range reasons {
		if r == ReasonLabelsAny {
			found = true
		}
	}
	if !found {
		t.Fatalf("reasons %v missing ReasonLabelsAny", reasons)
	}

	// An empty labels_any is unconstrained.
	ok, _ = Match("train", Requires{}.Normalize("cpu"), Candidate{NodeID: "node-b", Declaration: cand.Declaration})
	if !ok {
		t.Fatalf("empty labels_any should match with no grant labels at all")
	}
}
