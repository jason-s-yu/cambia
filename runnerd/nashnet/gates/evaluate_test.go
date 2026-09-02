package gates

import (
	"testing"
	"time"
)

func fp(v float64) *float64 { return &v }
func ip(v int) *int         { return &v }
func bp(v bool) *bool       { return &v }

var fixedNow = time.Date(2026, 6, 9, 12, 0, 0, 0, time.UTC) // a Tuesday, inside no window by default

// nodeADevices and nodeBDevices are the fixture device inventories the
// design brief pins the fixture node names against (D47's "Test fixtures
// name nodes node-a and node-b").
func nodeADevices() []DeviceRef {
	return []DeviceRef{{ID: "cpu", Kind: "cpu"}, {ID: "cuda:0", Kind: "cuda"}, {ID: "cuda:1", Kind: "cuda"}}
}

func nodeBDevices() []DeviceRef {
	return []DeviceRef{{ID: "cpu", Kind: "cpu"}, {ID: "xpu:0", Kind: "xpu"}}
}

// TestEvaluateGateTable exercises one row per gate category named in AC(3):
// floors, ceilings, co-tenant, drain, power, and concurrency, each admitting
// when satisfied and refusing (with the configured on_breach echoed) when
// breached.
func TestEvaluateGateTable(t *testing.T) {
	cases := []struct {
		name       string
		cfg        Config
		snap       Snapshot
		wantAdmit  bool
		wantGate   string
		wantOK     bool
		wantBreach OnBreach
	}{
		{
			name:      "floors.free_ram_gb satisfied",
			cfg:       Config{Floors: &FloorsGate{FreeRAMGB: fp(8), OnBreach: OnBreachFinish}},
			snap:      Snapshot{RAMFreeGB: fp(21.4)},
			wantAdmit: true, wantGate: "floors.free_ram_gb", wantOK: true, wantBreach: OnBreachFinish,
		},
		{
			name:      "floors.free_ram_gb breached",
			cfg:       Config{Floors: &FloorsGate{FreeRAMGB: fp(8), OnBreach: OnBreachFinish}},
			snap:      Snapshot{RAMFreeGB: fp(2.0)},
			wantAdmit: false, wantGate: "floors.free_ram_gb", wantOK: false, wantBreach: OnBreachFinish,
		},
		{
			name:      "floors.free_vram_gb per-device breached",
			cfg:       Config{Floors: &FloorsGate{FreeVRAMGB: map[string]float64{"cuda:0": 12}, OnBreach: OnBreachFinish}},
			snap:      Snapshot{Devices: []DeviceSnapshot{{ID: "cuda:0", Kind: "cuda", VRAMFreeGB: fp(6.1)}}},
			wantAdmit: false, wantGate: "floors.free_vram_gb.cuda:0", wantOK: false, wantBreach: OnBreachFinish,
		},
		{
			name:      "ceilings.load1_per_core satisfied",
			cfg:       Config{Ceilings: &CeilingsGate{Load1PerCore: fp(0.6), OnBreach: OnBreachFinish}},
			snap:      Snapshot{Load1: fp(4.0), Cores: 32},
			wantAdmit: true, wantGate: "ceilings.load1_per_core", wantOK: true, wantBreach: OnBreachFinish,
		},
		{
			name:      "ceilings.load1_per_core breached",
			cfg:       Config{Ceilings: &CeilingsGate{Load1PerCore: fp(0.6), OnBreach: OnBreachFinish}},
			snap:      Snapshot{Load1: fp(30.0), Cores: 32},
			wantAdmit: false, wantGate: "ceilings.load1_per_core", wantOK: false, wantBreach: OnBreachFinish,
		},
		{
			name:      "ceilings.gpu_temp_c breached on one device",
			cfg:       Config{Ceilings: &CeilingsGate{GPUTempC: fp(80), OnBreach: OnBreachFinish}},
			snap:      Snapshot{Devices: []DeviceSnapshot{{ID: "cuda:0", Kind: "cuda", TempC: fp(88)}}},
			wantAdmit: false, wantGate: "ceilings.gpu_temp_c.cuda:0", wantOK: false, wantBreach: OnBreachFinish,
		},
		{
			name:      "cotenants.deny_when_process_matches breached",
			cfg:       Config{Cotenants: &CotenantsGate{DenyWhenProcessMatches: []string{"^vllm", "^ollama"}, OnBreach: OnBreachStop}},
			snap:      Snapshot{ProcessNames: []string{"bash", "vllm-server"}},
			wantAdmit: false, wantGate: "cotenants.deny_when_process_matches", wantOK: false, wantBreach: OnBreachStop,
		},
		{
			name:      "cotenants.deny_when_process_matches clean",
			cfg:       Config{Cotenants: &CotenantsGate{DenyWhenProcessMatches: []string{"^vllm", "^ollama"}, OnBreach: OnBreachStop}},
			snap:      Snapshot{ProcessNames: []string{"bash", "python"}},
			wantAdmit: true, wantGate: "cotenants.deny_when_process_matches", wantOK: true, wantBreach: OnBreachStop,
		},
		{
			name:      "cotenants.deny_when_vram_held_by_others_gb breached",
			cfg:       Config{Cotenants: &CotenantsGate{DenyWhenVRAMHeldByOthersGB: map[string]float64{"cuda:0": 40}, OnBreach: OnBreachStop}},
			snap:      Snapshot{Devices: []DeviceSnapshot{{ID: "cuda:0", Kind: "cuda", VRAMHeldByOthersGB: fp(50)}}},
			wantAdmit: false, wantGate: "cotenants.deny_when_vram_held_by_others_gb.cuda:0", wantOK: false, wantBreach: OnBreachStop,
		},
		{
			name:      "drain file present",
			cfg:       Config{Drain: &DrainGate{File: "/tmp/DRAIN", OnBreach: OnBreachDrain}},
			snap:      Snapshot{DrainFilePresent: true},
			wantAdmit: false, wantGate: "drain", wantOK: false, wantBreach: OnBreachDrain,
		},
		{
			name:      "drain file absent",
			cfg:       Config{Drain: &DrainGate{File: "/tmp/DRAIN", OnBreach: OnBreachDrain}},
			snap:      Snapshot{DrainFilePresent: false},
			wantAdmit: true, wantGate: "drain", wantOK: true, wantBreach: OnBreachDrain,
		},
		{
			name:      "power.require_ac breached (on battery)",
			cfg:       Config{Power: &PowerGate{RequireAC: true, OnBreach: OnBreachDrain}},
			snap:      Snapshot{ACPower: bp(false)},
			wantAdmit: false, wantGate: "power.require_ac", wantOK: false, wantBreach: OnBreachDrain,
		},
		{
			name:      "power.min_battery_percent breached",
			cfg:       Config{Power: &PowerGate{MinBatteryPercent: fp(40), OnBreach: OnBreachDrain}},
			snap:      Snapshot{BatteryPercent: fp(15)},
			wantAdmit: false, wantGate: "power.min_battery_percent", wantOK: false, wantBreach: OnBreachDrain,
		},
		{
			name:      "concurrency.max_slots at cap",
			cfg:       Config{Concurrency: &ConcurrencyGate{MaxSlots: ip(2)}},
			snap:      Snapshot{RunningSlots: 2},
			wantAdmit: false, wantGate: "concurrency.max_slots", wantOK: false,
		},
		{
			name:      "concurrency.max_slots under cap",
			cfg:       Config{Concurrency: &ConcurrencyGate{MaxSlots: ip(2)}},
			snap:      Snapshot{RunningSlots: 1},
			wantAdmit: true, wantGate: "concurrency.max_slots", wantOK: true,
		},
		{
			name:      "concurrency.max_accelerator_jobs at cap",
			cfg:       Config{Concurrency: &ConcurrencyGate{MaxAcceleratorJobs: ip(1)}},
			snap:      Snapshot{RunningAcceleratorJobs: 1},
			wantAdmit: false, wantGate: "concurrency.max_accelerator_jobs", wantOK: false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			report := Evaluate(c.cfg, nodeADevices(), c.snap, fixedNow)
			if report.Admit != c.wantAdmit {
				t.Fatalf("Admit = %v, want %v (checks: %+v)", report.Admit, c.wantAdmit, report.Checks)
			}
			var found *Check
			for i := range report.Checks {
				if report.Checks[i].Gate == c.wantGate {
					found = &report.Checks[i]
					break
				}
			}
			if found == nil {
				t.Fatalf("no check named %q in %+v", c.wantGate, report.Checks)
			}
			if found.OK != c.wantOK {
				t.Errorf("check %q OK = %v, want %v", c.wantGate, found.OK, c.wantOK)
			}
			if c.wantBreach != "" && found.OnBreach != c.wantBreach {
				t.Errorf("check %q OnBreach = %q, want %q", c.wantGate, found.OnBreach, c.wantBreach)
			}
		})
	}
}

// TestEvaluateUnmeasuredFactFailsClosed covers a floor whose fact the node
// never measured: it must fail, not silently pass.
func TestEvaluateUnmeasuredFactFailsClosed(t *testing.T) {
	cfg := Config{Floors: &FloorsGate{FreeRAMGB: fp(8), OnBreach: OnBreachFinish}}
	report := Evaluate(cfg, nodeADevices(), Snapshot{}, fixedNow)
	if report.Admit {
		t.Fatal("expected admit=false when the RAM fact was never measured")
	}
}

// TestEvaluateNoConfigAdmitsWithOneSlot covers an unconfigured node (every
// gate nil): nothing blocks it, and with no concurrency gate it offers a
// single slot.
func TestEvaluateNoConfigAdmitsWithOneSlot(t *testing.T) {
	report := Evaluate(Config{}, nodeADevices(), Snapshot{}, fixedNow)
	if !report.Admit {
		t.Fatalf("expected admit=true with no gates configured, got checks %+v", report.Checks)
	}
	if report.SlotsOffered != 1 {
		t.Errorf("SlotsOffered = %d, want 1", report.SlotsOffered)
	}
}

// TestEvaluateDevicesAllowedResolvedIDSet covers AC(3)'s devices_allowed
// cases: a kind entry, an id entry, and a mixed list, with the report
// echoing the resolved id set.
func TestEvaluateDevicesAllowedResolvedIDSet(t *testing.T) {
	cases := []struct {
		name     string
		patterns []string
		want     []string
	}{
		{"kind entry matches every device of that kind", []string{"cuda"}, []string{"cuda:0", "cuda:1"}},
		{"id entry matches exactly one device", []string{"cuda:0"}, []string{"cuda:0"}},
		{"mixed list unions kind and id entries", []string{"cpu", "cuda:1"}, []string{"cpu", "cuda:1"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			cfg := Config{DevicesAllowed: c.patterns}
			report := Evaluate(cfg, nodeADevices(), Snapshot{}, fixedNow)
			if len(report.DevicesAllowed) != len(c.want) {
				t.Fatalf("DevicesAllowed = %v, want %v", report.DevicesAllowed, c.want)
			}
			for i, id := range c.want {
				if report.DevicesAllowed[i] != id {
					t.Errorf("DevicesAllowed[%d] = %q, want %q (full: %v)", i, report.DevicesAllowed[i], id, report.DevicesAllowed)
				}
			}
		})
	}
}

// TestEvaluateNodeBFixture exercises the second pinned fixture name against
// its own device inventory (xpu-only accelerator), independent of node-a.
func TestEvaluateNodeBFixture(t *testing.T) {
	cfg := Config{Floors: &FloorsGate{FreeVRAMGB: map[string]float64{"xpu:0": 4}, OnBreach: OnBreachFinish}}
	snap := Snapshot{Devices: []DeviceSnapshot{{ID: "xpu:0", Kind: "xpu", VRAMFreeGB: fp(6)}}}
	report := Evaluate(cfg, nodeBDevices(), snap, fixedNow)
	if !report.Admit {
		t.Fatalf("expected node-b to admit, got checks %+v", report.Checks)
	}
}
