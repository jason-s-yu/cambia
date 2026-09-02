package capability

import (
	"reflect"
	"testing"
)

// TestClampD47Table exercises one row per entry of the D47 clamp table: an
// inflated declared value is clamped down to the grant's cap, and an unset
// cap leaves the declared value unclamped (the enrolling operator's explicit
// choice, D47).
func TestClampD47Table(t *testing.T) {
	cases := []struct {
		name  string
		decl  Declaration
		grant Grant
		check func(t *testing.T, out Declaration)
	}{
		{
			name:  "slots clamped to max_slots",
			decl:  Declaration{Slots: 1000},
			grant: Grant{MaxSlots: i(4)},
			check: func(t *testing.T, out Declaration) {
				if out.Slots != 4 {
					t.Fatalf("Slots = %d, want 4", out.Slots)
				}
			},
		},
		{
			name:  "slots unclamped when cap unset",
			decl:  Declaration{Slots: 1000},
			grant: Grant{},
			check: func(t *testing.T, out Declaration) {
				if out.Slots != 1000 {
					t.Fatalf("Slots = %d, want 1000 (no cap)", out.Slots)
				}
			},
		},
		{
			name:  "kinds intersected with caps.kinds",
			decl:  Declaration{Kinds: []string{"train", "evaluate", "bench"}},
			grant: Grant{Kinds: []string{"train"}},
			check: func(t *testing.T, out Declaration) {
				if !reflect.DeepEqual(out.Kinds, []string{"train"}) {
					t.Fatalf("Kinds = %v, want [train]", out.Kinds)
				}
			},
		},
		{
			name:  "kinds unclamped when cap unset",
			decl:  Declaration{Kinds: []string{"train", "evaluate"}},
			grant: Grant{},
			check: func(t *testing.T, out Declaration) {
				if !reflect.DeepEqual(out.Kinds, []string{"train", "evaluate"}) {
					t.Fatalf("Kinds = %v, want unchanged", out.Kinds)
				}
			},
		},
		{
			name: "devices filtered to caps.device_kinds",
			decl: Declaration{Devices: []Device{
				{ID: "cpu", Kind: "cpu"},
				{ID: "cuda:0", Kind: "cuda"},
				{ID: "xpu:0", Kind: "xpu"},
			}},
			grant: Grant{DeviceKinds: []string{"cpu", "cuda"}},
			check: func(t *testing.T, out Declaration) {
				if len(out.Devices) != 2 {
					t.Fatalf("len(Devices) = %d, want 2: %+v", len(out.Devices), out.Devices)
				}
				for _, d := range out.Devices {
					if d.Kind == "xpu" {
						t.Fatalf("xpu device survived the clamp: %+v", out.Devices)
					}
				}
			},
		},
		{
			name: "max_vram_gb clamped per device kind",
			decl: Declaration{Devices: []Device{
				{ID: "cuda:0", Kind: "cuda", VRAMTotalGB: f(80)},
				{ID: "xpu:0", Kind: "xpu", VRAMTotalGB: f(80)},
			}},
			grant: Grant{MaxVRAMGB: map[string]float64{"cuda": 24}},
			check: func(t *testing.T, out Declaration) {
				cuda, _ := deviceOfKind(out.Devices, "cuda")
				if cuda.VRAMTotalGB == nil || *cuda.VRAMTotalGB != 24 {
					t.Fatalf("cuda VRAMTotalGB = %v, want 24", cuda.VRAMTotalGB)
				}
				xpu, _ := deviceOfKind(out.Devices, "xpu")
				if xpu.VRAMTotalGB == nil || *xpu.VRAMTotalGB != 80 {
					t.Fatalf("xpu VRAMTotalGB = %v, want 80 (no cap for xpu)", xpu.VRAMTotalGB)
				}
			},
		},
		{
			name:  "max_cores clamped",
			decl:  Declaration{Devices: []Device{{ID: "cpu", Kind: "cpu", Cores: i(128)}}},
			grant: Grant{MaxCores: i(16)},
			check: func(t *testing.T, out Declaration) {
				cpu, _ := deviceOfKind(out.Devices, "cpu")
				if cpu.Cores == nil || *cpu.Cores != 16 {
					t.Fatalf("Cores = %v, want 16", cpu.Cores)
				}
			},
		},
		{
			name:  "max_ram_gb clamped",
			decl:  Declaration{Devices: []Device{{ID: "cpu", Kind: "cpu", RAMTotalGB: f(512)}}},
			grant: Grant{MaxRAMGB: f(64)},
			check: func(t *testing.T, out Declaration) {
				cpu, _ := deviceOfKind(out.Devices, "cpu")
				if cpu.RAMTotalGB == nil || *cpu.RAMTotalGB != 64 {
					t.Fatalf("RAMTotalGB = %v, want 64", cpu.RAMTotalGB)
				}
			},
		},
		{
			name:  "max_disk_gb clamped",
			decl:  Declaration{DiskTotalGB: f(9000)},
			grant: Grant{MaxDiskGB: f(900)},
			check: func(t *testing.T, out Declaration) {
				if out.DiskTotalGB == nil || *out.DiskTotalGB != 900 {
					t.Fatalf("DiskTotalGB = %v, want 900", out.DiskTotalGB)
				}
			},
		},
		{
			name:  "labels ignored entirely",
			decl:  Declaration{Labels: []string{"trusted", "interactive"}},
			grant: Grant{Labels: []string{"interactive"}},
			check: func(t *testing.T, out Declaration) {
				if out.Labels != nil {
					t.Fatalf("Labels = %v, want nil (D47: ignored entirely)", out.Labels)
				}
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := Clamp(c.decl, c.grant)
			c.check(t, out)
		})
	}
}

// TestClampInflationIsInert is the D47 exactness property, using the fixture
// node names the design brief pins: node-a inflates every field far past its
// grant, and after Clamp it attracts nothing beyond its enrollment.
func TestClampInflationIsInert(t *testing.T) {
	nodeA := Declaration{
		Schema: 1,
		Slots:  1000,
		Kinds:  []string{"train", "evaluate", "measure"},
		Devices: []Device{
			{ID: "cpu", Kind: "cpu", Cores: i(999), RAMTotalGB: f(4000)},
			{ID: "cuda:0", Kind: "cuda", VRAMTotalGB: f(400)},
		},
		DiskTotalGB: f(90000),
		Labels:      []string{"trusted"},
	}
	grant := Grant{
		MaxSlots:    i(2),
		Kinds:       []string{"train"},
		DeviceKinds: []string{"cpu"},
		MaxCores:    i(8),
		MaxRAMGB:    f(16),
		MaxDiskGB:   f(100),
		Labels:      []string{"untrusted-pool"},
	}

	out := Clamp(nodeA, grant)
	if out.Slots != 2 {
		t.Errorf("Slots = %d, want 2", out.Slots)
	}
	if !reflect.DeepEqual(out.Kinds, []string{"train"}) {
		t.Errorf("Kinds = %v, want [train]", out.Kinds)
	}
	if len(out.Devices) != 1 || out.Devices[0].Kind != "cpu" {
		t.Errorf("Devices = %+v, want only the cpu entry (cuda filtered by device_kinds)", out.Devices)
	}
	cpu := out.Devices[0]
	if cpu.Cores == nil || *cpu.Cores != 8 {
		t.Errorf("Cores = %v, want 8", cpu.Cores)
	}
	if cpu.RAMTotalGB == nil || *cpu.RAMTotalGB != 16 {
		t.Errorf("RAMTotalGB = %v, want 16", cpu.RAMTotalGB)
	}
	if out.DiskTotalGB == nil || *out.DiskTotalGB != 100 {
		t.Errorf("DiskTotalGB = %v, want 100", out.DiskTotalGB)
	}
	if out.Labels != nil {
		t.Errorf("Labels = %v, want nil", out.Labels)
	}

	// node-a's declared "trusted" label never survives the clamp for
	// matching: Match reads candidate.GrantLabels (the grant's own set), and
	// a labels_any:[trusted] job does not place on node-a even though node-a
	// self-declared it.
	cand := Candidate{NodeID: "node-a", Declaration: out, GrantLabels: grant.Labels}
	req := Requires{LabelsAny: []string{"trusted"}}.Normalize("cpu")
	if ok, reasons := Match("train", req, cand); ok {
		t.Errorf("Match should fail on the self-declared label, got ok with reasons %v", reasons)
	}
}
