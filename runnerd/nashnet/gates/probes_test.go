package gates

import "testing"

// TestParseGPURowsFullFieldList covers the extended field list this ticket
// adds (memory.free,memory.total,utilization.gpu,temperature.gpu,name),
// distinct from procmgr.DefaultGPUQuery/GPUVRAMCheck's narrower
// memory.free,utilization.gpu,name: every row yields VRAMTotalGB and TempC,
// which GPUVRAMCheck's parser never reads at all.
func TestParseGPURowsFullFieldList(t *testing.T) {
	csv := "6144, 24576, 4, 41, NVIDIA RTX 4090\n" +
		"20480, 24576, 87, 79, NVIDIA RTX 4090\n"
	devs := ParseGPURows(csv)
	if len(devs) != 2 {
		t.Fatalf("len(devs) = %d, want 2", len(devs))
	}

	d0 := devs[0]
	if d0.ID != "cuda:0" || d0.Kind != "cuda" {
		t.Errorf("device 0 id/kind = %s/%s, want cuda:0/cuda", d0.ID, d0.Kind)
	}
	if d0.VRAMFreeGB == nil || *d0.VRAMFreeGB != 6.0 {
		t.Errorf("device 0 VRAMFreeGB = %v, want 6.0", d0.VRAMFreeGB)
	}
	if d0.VRAMTotalGB == nil || *d0.VRAMTotalGB != 24.0 {
		t.Errorf("device 0 VRAMTotalGB = %v, want 24.0", d0.VRAMTotalGB)
	}
	if d0.BusyPct == nil || *d0.BusyPct != 4 {
		t.Errorf("device 0 BusyPct = %v, want 4", d0.BusyPct)
	}
	if d0.TempC == nil || *d0.TempC != 41 {
		t.Errorf("device 0 TempC = %v, want 41", d0.TempC)
	}
	if d0.Name != "NVIDIA RTX 4090" {
		t.Errorf("device 0 Name = %q, want NVIDIA RTX 4090", d0.Name)
	}

	d1 := devs[1]
	if d1.ID != "cuda:1" {
		t.Errorf("device 1 id = %s, want cuda:1 (row order)", d1.ID)
	}
	if d1.TempC == nil || *d1.TempC != 79 {
		t.Errorf("device 1 TempC = %v, want 79", d1.TempC)
	}
}

// TestParseGPURowsMalformedFieldOmitted covers a truncated or unparseable
// row: the fact that field represents is nil, never a fabricated zero (D9).
func TestParseGPURowsMalformedFieldOmitted(t *testing.T) {
	csv := "6144, not-a-number, 4\n" // no temperature.gpu or name field at all
	devs := ParseGPURows(csv)
	if len(devs) != 1 {
		t.Fatalf("len(devs) = %d, want 1", len(devs))
	}
	d := devs[0]
	if d.VRAMFreeGB == nil || *d.VRAMFreeGB != 6.0 {
		t.Errorf("VRAMFreeGB = %v, want 6.0", d.VRAMFreeGB)
	}
	if d.VRAMTotalGB != nil {
		t.Errorf("VRAMTotalGB = %v, want nil (unparseable field)", *d.VRAMTotalGB)
	}
	if d.TempC != nil {
		t.Errorf("TempC = %v, want nil (field absent from the row)", *d.TempC)
	}
	if d.Name != "" {
		t.Errorf("Name = %q, want empty (field absent from the row)", d.Name)
	}
}

// TestParseXPURowKnownLimits covers AC(6): the XPU probe's known limits mean
// a real XPU node's declaration never carries vram_total_gb -- only free
// memory for device 0 is ever parsed.
func TestParseXPURowKnownLimits(t *testing.T) {
	dev, ok := ParseXPURow(`{"device": 0, "memory_free_mib": 8192}`)
	if !ok {
		t.Fatal("expected a parse")
	}
	if dev.ID != "xpu:0" || dev.Kind != "xpu" {
		t.Errorf("id/kind = %s/%s, want xpu:0/xpu", dev.ID, dev.Kind)
	}
	if dev.VRAMFreeGB == nil || *dev.VRAMFreeGB != 8.0 {
		t.Errorf("VRAMFreeGB = %v, want 8.0", dev.VRAMFreeGB)
	}
	if dev.VRAMTotalGB != nil {
		t.Errorf("VRAMTotalGB = %v, want nil: the probe never measures total VRAM (AC6)", *dev.VRAMTotalGB)
	}
	if dev.BusyPct != nil || dev.TempC != nil {
		t.Error("BusyPct/TempC should be nil: xpu-smi output at this field set carries neither")
	}
}

func TestParseXPURowNoMatch(t *testing.T) {
	if _, ok := ParseXPURow("garbage output with no free-memory field"); ok {
		t.Fatal("expected no parse")
	}
}
