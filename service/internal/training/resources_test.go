package training

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// fakeGPUs builds a gpuStatsFunc seam returning fixed output, so tests never
// touch a real GPU or launch GPU work.
func fakeGPUs(gpus []GPUStat, avail bool) gpuStatsFunc {
	return func(ctx context.Context) ([]GPUStat, bool) {
		return gpus, avail
	}
}

func TestResourceSnapshotListsGPUs(t *testing.T) {
	m := NewResourceMonitor(t.TempDir(), time.Second)
	defer m.Close()
	m.gpuStats = fakeGPUs([]GPUStat{
		{Index: 0, Name: "RTX PRO 6000", MemTotalMB: 97887, MemUsedMB: 95784, MemFreeMB: 856, UtilPct: 78, TempC: 58},
		{Index: 1, Name: "RTX PRO 6000", MemTotalMB: 97887, MemUsedMB: 1024, MemFreeMB: 96863, UtilPct: 3, TempC: 40},
	}, true)

	snap := m.Snapshot()

	if !snap.GPUAvailable {
		t.Fatal("expected gpu_available true")
	}
	if len(snap.GPUs) != 2 {
		t.Fatalf("expected 2 GPUs, got %d", len(snap.GPUs))
	}
	// Co-tenant VRAM pressure is visible in device-total used memory.
	if snap.GPUs[0].MemUsedMB != 95784 {
		t.Errorf("expected GPU0 mem_used 95784, got %v", snap.GPUs[0].MemUsedMB)
	}
	if snap.GPUs[0].MemTotalMB != 97887 {
		t.Errorf("expected GPU0 mem_total 97887, got %v", snap.GPUs[0].MemTotalMB)
	}
	// Host fields still populated from /proc + statfs.
	if snap.MemTotalMB <= 0 {
		t.Errorf("expected mem_total_mb > 0, got %v", snap.MemTotalMB)
	}
	if snap.DiskTotalGB <= 0 {
		t.Errorf("expected disk_total_gb > 0, got %v", snap.DiskTotalGB)
	}
	if snap.Timestamp == "" {
		t.Error("expected a timestamp")
	}
}

func TestResourceSnapshotCPUHost(t *testing.T) {
	m := NewResourceMonitor(t.TempDir(), time.Second)
	defer m.Close()
	// Simulate a CPU host: nvidia-smi absent -> (nil, false).
	m.gpuStats = fakeGPUs(nil, false)

	snap := m.Snapshot()

	if snap.GPUAvailable {
		t.Error("expected gpu_available false on a CPU host")
	}
	if len(snap.GPUs) != 0 {
		t.Errorf("expected no GPUs, got %d", len(snap.GPUs))
	}
	if snap.GPUs == nil {
		t.Error("expected an empty (non-nil) gpus slice for JSON []")
	}
	// CPU/mem/disk still populated without a GPU.
	if snap.MemTotalMB <= 0 {
		t.Errorf("expected mem_total_mb > 0, got %v", snap.MemTotalMB)
	}
	if snap.DiskTotalGB <= 0 {
		t.Errorf("expected disk_total_gb > 0, got %v", snap.DiskTotalGB)
	}
}

func TestResourceWSBackfillAndStream(t *testing.T) {
	m := NewResourceMonitor(t.TempDir(), 40*time.Millisecond)
	defer m.Close()
	m.gpuStats = fakeGPUs([]GPUStat{
		{Index: 0, Name: "fake", MemTotalMB: 1000, MemUsedMB: 500, MemFreeMB: 500, UtilPct: 10, TempC: 45},
	}, true)

	if m.sampling() {
		t.Fatal("sampler should not run with zero clients")
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/ws/training/resources", m.HandleWS)
	ts := httptest.NewServer(mux)
	defer ts.Close()

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/training/resources"
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	c, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatal(err)
	}

	// First frame is the backfill envelope.
	_, data, err := c.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var msg WSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatal(err)
	}
	if msg.Type != "resource_backfill" {
		t.Fatalf("expected resource_backfill, got %s", msg.Type)
	}
	if _, ok := msg.Data.(map[string]interface{})["samples"]; !ok {
		t.Error("backfill missing samples key")
	}

	// Next frame is a live sample from the sampler.
	_, data, err = c.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &msg); err != nil {
		t.Fatal(err)
	}
	if msg.Type != "resource_sample" {
		t.Fatalf("expected resource_sample, got %s", msg.Type)
	}
	sampleData, ok := msg.Data.(map[string]interface{})
	if !ok {
		t.Fatal("expected sample data to be an object")
	}
	if avail, _ := sampleData["gpu_available"].(bool); !avail {
		t.Error("expected gpu_available true in the sample")
	}
	if gpus, ok := sampleData["gpus"].([]interface{}); !ok || len(gpus) != 1 {
		t.Errorf("expected 1 GPU in the sample, got %v", sampleData["gpus"])
	}

	if !m.sampling() {
		t.Error("sampler should be running with a connected client")
	}

	// Disconnect and assert the sampler stops (client count returns to 0).
	c.Close(websocket.StatusNormalClosure, "done")

	deadline := time.Now().Add(3 * time.Second)
	for m.sampling() && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if m.sampling() {
		t.Error("sampler should stop after the last client disconnects")
	}
	if m.clientCount() != 0 {
		t.Errorf("expected 0 clients after disconnect, got %d", m.clientCount())
	}

	// No further ticks: the ring is frozen once the sampler stops.
	m.mu.Lock()
	frozen := len(m.ring)
	m.mu.Unlock()
	time.Sleep(120 * time.Millisecond)
	m.mu.Lock()
	after := len(m.ring)
	m.mu.Unlock()
	if after != frozen {
		t.Errorf("ring grew after the sampler stopped: %d -> %d", frozen, after)
	}
}

func TestGPUReuseOnTransientFailure(t *testing.T) {
	m := NewResourceMonitor(t.TempDir(), time.Second)
	defer m.Close()

	// First broadcast establishes a GPU.
	m.broadcast(ResourceSnapshot{GPUAvailable: true, GPUs: []GPUStat{{Index: 0, Name: "g", MemUsedMB: 500}}})
	// A transient query failure returns no GPUs.
	m.broadcast(ResourceSnapshot{GPUAvailable: false, GPUs: []GPUStat{}})

	m.mu.Lock()
	last := m.ring[len(m.ring)-1]
	m.mu.Unlock()
	if !last.GPUAvailable {
		t.Error("a transient failure should reuse the last known GPUs, not report no-GPU")
	}
	if len(last.GPUs) != 1 || last.GPUs[0].MemUsedMB != 500 {
		t.Errorf("expected reused GPU with mem_used 500, got %+v", last.GPUs)
	}
}

func TestGPUCPUHostStaysUnavailable(t *testing.T) {
	m := NewResourceMonitor(t.TempDir(), time.Second)
	defer m.Close()

	// A CPU host never establishes GPUs, so it stays gpu_available false.
	m.broadcast(ResourceSnapshot{GPUAvailable: false, GPUs: []GPUStat{}})
	m.broadcast(ResourceSnapshot{GPUAvailable: false, GPUs: []GPUStat{}})

	m.mu.Lock()
	last := m.ring[len(m.ring)-1]
	m.mu.Unlock()
	if last.GPUAvailable {
		t.Error("a CPU host should stay gpu_available false")
	}
	if len(last.GPUs) != 0 {
		t.Errorf("expected no GPUs on a CPU host, got %d", len(last.GPUs))
	}
}

func TestResourceRingBounded(t *testing.T) {
	m := NewResourceMonitor(t.TempDir(), time.Second)
	defer m.Close()
	for i := 0; i < resourceRingSize+25; i++ {
		m.broadcast(ResourceSnapshot{GPUAvailable: false, GPUs: []GPUStat{}})
	}
	m.mu.Lock()
	n := len(m.ring)
	m.mu.Unlock()
	if n != resourceRingSize {
		t.Errorf("expected ring capped at %d, got %d", resourceRingSize, n)
	}
}

func TestHandleSnapshotJSON(t *testing.T) {
	m := NewResourceMonitor(t.TempDir(), time.Second)
	defer m.Close()
	m.gpuStats = fakeGPUs(nil, false)

	req := httptest.NewRequest(http.MethodGet, "/training/system/resources", nil)
	w := httptest.NewRecorder()
	m.HandleSnapshot(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var snap ResourceSnapshot
	if err := json.Unmarshal(w.Body.Bytes(), &snap); err != nil {
		t.Fatalf("response is not a ResourceSnapshot: %v", err)
	}
	if snap.MemTotalMB <= 0 {
		t.Errorf("expected mem_total_mb > 0, got %v", snap.MemTotalMB)
	}

	// Non-GET is rejected.
	w2 := httptest.NewRecorder()
	m.HandleSnapshot(w2, httptest.NewRequest(http.MethodPost, "/training/system/resources", nil))
	if w2.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405 on POST, got %d", w2.Code)
	}
}

func TestParseGPUCSV(t *testing.T) {
	// The real nvidia-smi format on this host (comma-space separated), plus a
	// malformed line and an [N/A] temperature.
	out := "0, NVIDIA RTX PRO 6000, 97887, 95784, 856, 78, 58\n" +
		"garbage line\n" +
		"1, NVIDIA RTX PRO 6000, 97887, 1024, 96863, 3, [N/A]\n"
	gpus := parseGPUCSV(out)
	if len(gpus) != 2 {
		t.Fatalf("expected 2 parsed GPUs (malformed line skipped), got %d", len(gpus))
	}
	if gpus[0].Index != 0 || gpus[0].Name != "NVIDIA RTX PRO 6000" {
		t.Errorf("bad GPU0 header: %+v", gpus[0])
	}
	if gpus[0].MemUsedMB != 95784 || gpus[0].MemFreeMB != 856 || gpus[0].UtilPct != 78 || gpus[0].TempC != 58 {
		t.Errorf("bad GPU0 values: %+v", gpus[0])
	}
	// [N/A] temperature falls back to 0 without dropping the GPU.
	if gpus[1].TempC != 0 {
		t.Errorf("expected [N/A] temp -> 0, got %v", gpus[1].TempC)
	}
}

func TestReadHostParsers(t *testing.T) {
	// These read real host /proc and statfs; they should return sane values on
	// any Linux CI host without launching GPU work.
	prev, err := readProcStat()
	if err != nil {
		t.Fatalf("readProcStat: %v", err)
	}
	if prev.agg.total == 0 {
		t.Error("expected non-zero aggregate CPU jiffies")
	}
	time.Sleep(60 * time.Millisecond)
	cur, err := readProcStat()
	if err != nil {
		t.Fatalf("readProcStat: %v", err)
	}
	pct, perCore := cpuUtil(prev, cur)
	if pct < 0 || pct > 100 {
		t.Errorf("cpu util out of range: %v", pct)
	}
	for i, c := range perCore {
		if c < 0 || c > 100 {
			t.Errorf("core %d util out of range: %v", i, c)
		}
	}

	totalMB, availMB := readMemInfo()
	if totalMB <= 0 || availMB <= 0 || availMB > totalMB {
		t.Errorf("bad meminfo: total=%v avail=%v", totalMB, availMB)
	}

	total, used, free := diskStat(t.TempDir())
	if total <= 0 || free < 0 || used < 0 {
		t.Errorf("bad diskStat: total=%v used=%v free=%v", total, used, free)
	}
}
