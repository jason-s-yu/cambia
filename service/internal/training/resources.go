package training

import (
	"context"
	"net/http"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/coder/websocket"
)

// Resource monitor tuning constants.
const (
	// resourceRingSize is the bounded live window of samples kept in memory to
	// backfill a newly-connected client. Not a historical store.
	resourceRingSize = 60
	// gpuQueryTimeout bounds a single nvidia-smi exec so a stalled GPU (under
	// co-tenant load) cannot block the broadcast hub (contract R1).
	gpuQueryTimeout = 3 * time.Second
	// cpuDeltaWindow is the spacing between the two /proc/stat reads a one-off
	// Snapshot uses to compute a CPU-utilization delta without the sampler.
	cpuDeltaWindow = 100 * time.Millisecond
	// wsClientBuffer bounds a slow client's queue; the hub drops samples for a
	// client whose buffer is full rather than stalling every other viewer.
	wsClientBuffer = 8
	// defaultResourceInterval is the sampler tick when NewResourceMonitor is
	// given a non-positive interval.
	defaultResourceInterval = 2 * time.Second
)

// GPUProc is one process holding memory on a GPU, when the platform exposes it.
// On WSL2 the process list is commonly empty or degraded (contract R2); the UI
// falls back to device-total pressure.
type GPUProc struct {
	PID   int     `json:"pid"`
	Name  string  `json:"name"`
	MemMB float64 `json:"mem_mb"`
}

// GPUStat is one GPU's memory, utilization, and temperature. MemUsedMB reflects
// device-total used memory, so a co-tenant holding the card is visible even when
// this project holds none (GPU-contention policy).
type GPUStat struct {
	Index      int       `json:"index"`
	Name       string    `json:"name"`
	MemTotalMB float64   `json:"mem_total_mb"`
	MemUsedMB  float64   `json:"mem_used_mb"`
	MemFreeMB  float64   `json:"mem_free_mb"`
	UtilPct    float64   `json:"util_pct"`
	TempC      float64   `json:"temp_c"`
	Processes  []GPUProc `json:"processes,omitempty"`
}

// ResourceSnapshot is one host sample: CPU, memory, disk, load average, and the
// per-GPU list. GPUAvailable is false on a CPU host (no nvidia-smi), in which
// case GPUs is empty but the CPU/mem/disk fields are still populated.
type ResourceSnapshot struct {
	Timestamp    string     `json:"timestamp"`
	CPUPct       float64    `json:"cpu_pct"`
	PerCorePct   []float64  `json:"per_core_pct,omitempty"`
	LoadAvg      [3]float64 `json:"load_avg"`
	MemTotalMB   float64    `json:"mem_total_mb"`
	MemUsedMB    float64    `json:"mem_used_mb"`
	MemAvailMB   float64    `json:"mem_avail_mb"`
	DiskTotalGB  float64    `json:"disk_total_gb"`
	DiskUsedGB   float64    `json:"disk_used_gb"`
	DiskFreeGB   float64    `json:"disk_free_gb"`
	GPUs         []GPUStat  `json:"gpus"`
	GPUAvailable bool       `json:"gpu_available"`
}

// gpuStatsFunc is the seam for the per-GPU query. The default shells nvidia-smi;
// tests inject a fake to avoid touching a real GPU. It returns (nil, false) when
// the binary is absent (CPU host) or a query fails, and (gpus, true) otherwise.
type gpuStatsFunc func(ctx context.Context) ([]GPUStat, bool)

// resClient is one connected WebSocket viewer. The hub pushes samples to ch; a
// per-connection writer in HandleWS drains it, so one slow client never stalls
// the sampler.
type resClient struct {
	ch chan ResourceSnapshot
}

// ResourceMonitor samples host CPU/RAM/disk/load and per-GPU stats on an
// interval and fans each sample out to all connected WS clients. It is the
// inverse of the per-connection log tailer: one sampler, many viewers. The
// sampler starts on the first client and stops after the last disconnects, so an
// unwatched dashboard runs no nvidia-smi loop.
type ResourceMonitor struct {
	runsDir  string
	interval time.Duration
	gpuStats gpuStatsFunc

	baseCtx    context.Context
	baseCancel context.CancelFunc

	mu      sync.Mutex
	clients map[*resClient]struct{}
	ring    []ResourceSnapshot
	running bool
	stopCh  chan struct{}
	closed  bool
	// lastGPUs / lastGPUAvail let a transient gpu-query failure reuse the last
	// known GPU list rather than flapping the UI to "no GPU" (contract R1).
	lastGPUs     []GPUStat
	lastGPUAvail bool
}

// NewResourceMonitor builds a monitor sampling runsDir's filesystem for disk and
// the host /proc for CPU/RAM/load. interval <= 0 falls back to 2s.
func NewResourceMonitor(runsDir string, interval time.Duration) *ResourceMonitor {
	if interval <= 0 {
		interval = defaultResourceInterval
	}
	ctx, cancel := context.WithCancel(context.Background())
	return &ResourceMonitor{
		runsDir:    runsDir,
		interval:   interval,
		gpuStats:   defaultGPUStats,
		baseCtx:    ctx,
		baseCancel: cancel,
		clients:    make(map[*resClient]struct{}),
	}
}

// Snapshot returns one host sample. When the sampler is running it returns the
// most recent broadcast sample (avoiding a redundant nvidia-smi shell-out during
// active streaming); otherwise it takes a fresh one-off sample, spacing two
// /proc/stat reads to compute a CPU delta.
func (m *ResourceMonitor) Snapshot() ResourceSnapshot {
	m.mu.Lock()
	if m.running && len(m.ring) > 0 {
		s := m.ring[len(m.ring)-1]
		m.mu.Unlock()
		return s
	}
	m.mu.Unlock()

	prev, _ := readProcStat()
	time.Sleep(cpuDeltaWindow)
	snap, _ := m.sample(prev)
	return snap
}

// HandleSnapshot serves a one-off GET snapshot as JSON.
func (m *ResourceMonitor) HandleSnapshot(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, m.Snapshot())
}

// HandleWS upgrades to WebSocket, sends a resource_backfill frame with the
// current ring, then streams resource_sample frames. Registering the client
// starts the sampler if it was idle; disconnecting stops it when it was the last
// client.
func (m *ResourceMonitor) HandleWS(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		OriginPatterns: []string{"*"},
	})
	if err != nil {
		return
	}
	defer c.CloseNow()

	ctx, cancel := context.WithCancel(r.Context())
	defer cancel()

	cl := &resClient{ch: make(chan ResourceSnapshot, wsClientBuffer)}
	backfill := m.register(cl)
	defer m.unregister(cl)

	if err := writeWSMessage(ctx, c, WSMessage{
		Type: "resource_backfill",
		Data: map[string]interface{}{"samples": backfill},
	}); err != nil {
		return
	}

	// Drain client reads so control frames (close/ping) are handled and a
	// disconnect cancels the stream.
	go func() {
		for {
			if _, _, err := c.Read(ctx); err != nil {
				cancel()
				return
			}
		}
	}()

	for {
		select {
		case <-ctx.Done():
			return
		case snap := <-cl.ch:
			if err := writeWSMessage(ctx, c, WSMessage{
				Type: "resource_sample",
				Data: snap,
			}); err != nil {
				return
			}
		}
	}
}

// Close stops the sampler and cancels any in-flight GPU query.
func (m *ResourceMonitor) Close() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return
	}
	m.closed = true
	if m.running {
		close(m.stopCh)
		m.running = false
	}
	if m.baseCancel != nil {
		m.baseCancel()
	}
}

// register adds a client, starting the sampler when the client count goes 0->1,
// and returns a copy of the ring for backfill.
func (m *ResourceMonitor) register(cl *resClient) []ResourceSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return []ResourceSnapshot{}
	}
	m.clients[cl] = struct{}{}
	if !m.running {
		m.running = true
		m.stopCh = make(chan struct{})
		go m.runSampler(m.stopCh)
	}
	out := make([]ResourceSnapshot, len(m.ring))
	copy(out, m.ring)
	return out
}

// unregister removes a client, stopping the sampler when the last one leaves. It
// is idempotent: a client already gone is a no-op.
func (m *ResourceMonitor) unregister(cl *resClient) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.clients[cl]; !ok {
		return
	}
	delete(m.clients, cl)
	if len(m.clients) == 0 && m.running {
		close(m.stopCh)
		m.running = false
	}
}

// runSampler ticks every interval, samples the host, and broadcasts. It holds
// the previous /proc/stat reading so each sample is a true interval CPU delta.
func (m *ResourceMonitor) runSampler(stop chan struct{}) {
	prev, _ := readProcStat()
	ticker := time.NewTicker(m.interval)
	defer ticker.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ticker.C:
			snap, cur := m.sample(prev)
			prev = cur
			m.broadcast(snap)
		}
	}
}

// sample assembles one ResourceSnapshot from prev->current /proc/stat plus mem,
// load, disk, and the GPU seam. It returns the current /proc/stat reading for
// the next delta.
func (m *ResourceMonitor) sample(prev procStat) (ResourceSnapshot, procStat) {
	cur, err := readProcStat()
	var cpuPct float64
	var perCore []float64
	if err == nil {
		cpuPct, perCore = cpuUtil(prev, cur)
	}

	memTotal, memAvail := readMemInfo()
	load := readLoadAvg()
	diskTotal, diskUsed, diskFree := diskStat(m.runsDir)

	ctx := m.baseCtx
	if ctx == nil {
		ctx = context.Background()
	}
	gpus, avail := m.gpuStats(ctx)
	if gpus == nil {
		gpus = []GPUStat{}
	}

	return ResourceSnapshot{
		Timestamp:    time.Now().UTC().Format(time.RFC3339),
		CPUPct:       cpuPct,
		PerCorePct:   perCore,
		LoadAvg:      load,
		MemTotalMB:   memTotal,
		MemUsedMB:    memTotal - memAvail,
		MemAvailMB:   memAvail,
		DiskTotalGB:  diskTotal,
		DiskUsedGB:   diskUsed,
		DiskFreeGB:   diskFree,
		GPUs:         gpus,
		GPUAvailable: avail,
	}, cur
}

// broadcast applies the gpu reuse-on-failure heuristic, appends to the ring, and
// pushes the sample to every client (dropping for a client whose buffer is full).
func (m *ResourceMonitor) broadcast(snap ResourceSnapshot) {
	m.mu.Lock()
	// A transient gpu-query failure (timeout under load) returns no GPUs; if we
	// have previously seen GPUs, reuse them rather than reporting "no GPU". A
	// genuine CPU host never established GPUs, so it keeps gpu_available false.
	if !snap.GPUAvailable && len(snap.GPUs) == 0 && m.lastGPUAvail {
		snap.GPUs = m.lastGPUs
		snap.GPUAvailable = true
	} else {
		m.lastGPUs = snap.GPUs
		m.lastGPUAvail = snap.GPUAvailable
	}

	m.ring = append(m.ring, snap)
	if len(m.ring) > resourceRingSize {
		m.ring = m.ring[len(m.ring)-resourceRingSize:]
	}

	chans := make([]chan ResourceSnapshot, 0, len(m.clients))
	for cl := range m.clients {
		chans = append(chans, cl.ch)
	}
	m.mu.Unlock()

	for _, ch := range chans {
		select {
		case ch <- snap:
		default: // slow client: drop this sample rather than stall the hub
		}
	}
}

// clientCount reports the number of connected WS clients.
func (m *ResourceMonitor) clientCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.clients)
}

// sampling reports whether the sampler goroutine is running.
func (m *ResourceMonitor) sampling() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.running
}

// procStat holds aggregate and per-core CPU jiffy totals from /proc/stat.
type procStat struct {
	agg   cpuTimes
	cores []cpuTimes
}

// cpuTimes is one CPU's total and idle jiffies.
type cpuTimes struct {
	total uint64
	idle  uint64
}

// readProcStat parses the aggregate and per-core CPU lines of /proc/stat.
func readProcStat() (procStat, error) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return procStat{}, err
	}
	var ps procStat
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, "cpu") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 5 {
			continue
		}
		ct, ok := parseCPULine(fields)
		if !ok {
			continue
		}
		if fields[0] == "cpu" {
			ps.agg = ct
		} else {
			ps.cores = append(ps.cores, ct)
		}
	}
	return ps, nil
}

// parseCPULine sums a "cpu"/"cpuN" line's jiffies into total, treating the idle
// and iowait columns (indices 4 and 5) as idle time.
func parseCPULine(fields []string) (cpuTimes, bool) {
	var total, idle uint64
	for i := 1; i < len(fields); i++ {
		v, err := strconv.ParseUint(fields[i], 10, 64)
		if err != nil {
			return cpuTimes{}, false
		}
		total += v
		if i == 4 || i == 5 { // idle + iowait
			idle += v
		}
	}
	return cpuTimes{total: total, idle: idle}, true
}

// cpuUtil returns the aggregate utilization percent and per-core percents from a
// prev->cur delta.
func cpuUtil(prev, cur procStat) (float64, []float64) {
	agg := utilDelta(prev.agg, cur.agg)
	n := len(cur.cores)
	if len(prev.cores) < n {
		n = len(prev.cores)
	}
	var perCore []float64
	if n > 0 {
		perCore = make([]float64, n)
		for i := 0; i < n; i++ {
			perCore[i] = utilDelta(prev.cores[i], cur.cores[i])
		}
	}
	return agg, perCore
}

// utilDelta computes busy/total * 100 over a jiffy delta. A zero or negative
// total delta (no elapsed time, or a counter reset) yields 0.
func utilDelta(prev, cur cpuTimes) float64 {
	dTotal := int64(cur.total) - int64(prev.total)
	dIdle := int64(cur.idle) - int64(prev.idle)
	if dTotal <= 0 {
		return 0
	}
	busy := dTotal - dIdle
	if busy < 0 {
		busy = 0
	}
	return 100.0 * float64(busy) / float64(dTotal)
}

// readMemInfo returns total and available memory in MB from /proc/meminfo.
func readMemInfo() (totalMB, availMB float64) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}
	var totalKB, availKB float64
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		v, perr := strconv.ParseFloat(fields[1], 64)
		if perr != nil {
			continue
		}
		switch fields[0] {
		case "MemTotal:":
			totalKB = v
		case "MemAvailable:":
			availKB = v
		}
	}
	return totalKB / 1024.0, availKB / 1024.0
}

// readLoadAvg returns the 1/5/15-minute load averages from /proc/loadavg.
func readLoadAvg() [3]float64 {
	var la [3]float64
	data, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return la
	}
	fields := strings.Fields(string(data))
	for i := 0; i < 3 && i < len(fields); i++ {
		la[i], _ = strconv.ParseFloat(fields[i], 64)
	}
	return la
}

// diskStat returns total, used, and unprivileged-available space in GiB for the
// filesystem backing path. Free uses Bavail to match diskSpaceCheck's semantics.
func diskStat(path string) (totalGB, usedGB, freeGB float64) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(path, &st); err != nil {
		return 0, 0, 0
	}
	bsize := uint64(st.Bsize)
	totalBytes := st.Blocks * bsize
	freeBytes := st.Bavail * bsize
	usedBytes := (st.Blocks - st.Bfree) * bsize
	const gb = 1 << 30
	return float64(totalBytes) / gb, float64(usedBytes) / gb, float64(freeBytes) / gb
}

// defaultGPUStats shells nvidia-smi for per-GPU stats with a 3s timeout, plus a
// best-effort per-process query. It returns (nil, false) when the binary is
// absent (CPU host) or the query fails.
func defaultGPUStats(ctx context.Context) ([]GPUStat, bool) {
	cctx, cancel := context.WithTimeout(ctx, gpuQueryTimeout)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi",
		"--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, false
	}
	gpus := parseGPUCSV(string(out))
	if len(gpus) == 0 {
		return nil, false
	}
	// Best-effort per-process rows. On WSL2 these are commonly empty or degraded
	// ([Not Found]/[N/A]); attach only for a single-GPU host where the flat list
	// maps unambiguously to a device. Multi-GPU degrades to device-total memory.
	if len(gpus) == 1 {
		if procs := gpuComputeApps(ctx); len(procs) > 0 {
			gpus[0].Processes = procs
		}
	}
	return gpus, true
}

// parseGPUCSV parses the 7-field per-GPU CSV. Malformed lines are skipped; an
// unparseable numeric field falls back to 0 without dropping the GPU (handles
// "[N/A]" values some drivers emit).
func parseGPUCSV(out string) []GPUStat {
	var gpus []GPUStat
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) < 7 {
			continue
		}
		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}
		gpus = append(gpus, GPUStat{
			Index:      parseIntField(fields[0]),
			Name:       fields[1],
			MemTotalMB: parseFloatField(fields[2]),
			MemUsedMB:  parseFloatField(fields[3]),
			MemFreeMB:  parseFloatField(fields[4]),
			UtilPct:    parseFloatField(fields[5]),
			TempC:      parseFloatField(fields[6]),
		})
	}
	return gpus
}

// gpuComputeApps runs the best-effort per-process query. Any error (including
// WSL2's lack of support) yields nil.
func gpuComputeApps(ctx context.Context) []GPUProc {
	cctx, cancel := context.WithTimeout(ctx, gpuQueryTimeout)
	defer cancel()
	out, err := exec.CommandContext(cctx, "nvidia-smi",
		"--query-compute-apps=pid,process_name,used_memory",
		"--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	var procs []GPUProc
	for _, line := range strings.Split(string(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) < 3 {
			continue
		}
		procs = append(procs, GPUProc{
			PID:   parseIntField(strings.TrimSpace(fields[0])),
			Name:  strings.TrimSpace(fields[1]),
			MemMB: parseFloatField(strings.TrimSpace(fields[2])),
		})
	}
	return procs
}

// parseFloatField parses a trimmed CSV numeric field, returning 0 on failure.
func parseFloatField(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return v
}

// parseIntField parses a trimmed CSV integer field, returning 0 on failure.
func parseIntField(s string) int {
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return v
}
