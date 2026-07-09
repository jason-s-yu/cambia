export interface Run {
	id: number;
	name: string;
	algorithm: string;
	status: string;
	best_metric_value: number | null;
	best_metric_iter: number | null;
	created_at: string;
	updated_at: string;
	/** The Go-owned current-process-state record (runs/<name>/process.json),
	 * present when the run has a supervised or dashboard-created process; absent
	 * for an external run_db-only run. */
	process?: ProcessState;
	/** Origin host of a remote (serving-harness) run; absent for a local run.
	 * A remote run renders read-only on this dashboard in v1. */
	host?: string;
	/** RFC3339 timestamp of the last successful pull for a remote run; absent
	 * for a local run or a remote run never yet synced. */
	last_sync_at?: string;
	/** True when a remote run's synced projection is older than 3 sync intervals
	 * (bounded-stale threshold). Always false for a local run. */
	stale?: boolean;
}

export interface RunDetail extends Run {
	config_yaml: string;
	notes?: string;
	tags?: string;
}

export interface EvalMetric {
	iteration: number;
	baseline: string;
	win_rate: number | null;
	ci_low: number | null;
	ci_high: number | null;
	games_played: number | null;
	adv_loss: number | null;
	strat_loss: number | null;
	timestamp: string;
}

export interface MeanImpPoint {
	iteration: number;
	mean_imp: number;
}

export interface Checkpoint {
	id: number;
	iteration: number;
	file_path: string;
	file_size_bytes: number | null;
	created_at: string;
	is_best: boolean;
}

export type ProcessStatus = 'created' | 'starting' | 'running' | 'stopping' | 'stopped' | 'crashed';

export interface ProcessState {
	name: string;
	status: ProcessStatus;
	algorithm: string;
	pid: number;
	pgid: number;
	config_path: string;
	created_at: string;
	started_at?: string;
	finished_at?: string;
	exit_code?: number;
	last_error?: string;
	/** Origin host for a remote (serving-harness) run; empty/absent for a local
	 * run. Stamped by the dashboard store so a remote run's status renders as a
	 * bounded-stale projection rather than a local pid probe. */
	host?: string;
}

export interface PreflightCheck {
	name: string;
	ok: boolean;
	detail: string;
}

export interface CreateRunRequest {
	name: string;
	template: string;
	algorithm?: string;
	overrides?: Record<string, string | number>;
	yaml?: string;
}

/** POST /training/runs -> 201 response body. */
export interface CreateRunResponse {
	run: RunDetail;
	process: ProcessState;
}

/** POST /training/runs/{name}/start|stop|resume -> 202 response body. */
export interface ProcessActionResponse {
	process: ProcessState;
}

/** 409 response body shape for preflight failures (start/resume) and create collisions. */
export interface PreflightFailedResponse {
	error: string;
	checks?: PreflightCheck[];
	override?: string;
}

export interface StartRunOptions {
	force?: boolean;
	min_free_vram_gb?: number;
	min_free_disk_gb?: number;
}

export interface StopRunOptions {
	force?: boolean;
}

export interface ResumeRunOptions {
	force?: boolean;
	min_free_vram_gb?: number;
}

export interface EvalJob {
	id: string;
	run: string;
	status: 'queued' | 'running' | 'succeeded' | 'failed';
	target: string;
	device: 'cpu' | 'cuda';
	games: number;
	argmax: boolean;
	log_path: string;
	started_at?: string;
	finished_at?: string;
	exit_code?: number;
	error?: string;
	tail?: string[];
}

export interface TriggerEvalRequest {
	epoch?: number;
	device?: 'cpu' | 'cuda';
	games?: number;
	argmax?: boolean;
	force?: boolean;
	min_free_vram_gb?: number;
	min_free_disk_gb?: number;
}

/** POST /training/runs/{name}/eval -> 202 response body. */
export interface TriggerEvalResponse {
	job: EvalJob;
}

/** GET /training/runs/{name}/eval -> 200 response body. */
export interface EvalJobsResponse {
	jobs: EvalJob[];
}

export interface GPUProc {
	pid: number;
	name: string;
	mem_mb: number;
}

export interface GPUStat {
	index: number;
	name: string;
	mem_total_mb: number;
	mem_used_mb: number;
	mem_free_mb: number;
	util_pct: number;
	temp_c: number;
	processes?: GPUProc[];
}

export interface ResourceSnapshot {
	timestamp: string;
	cpu_pct: number;
	per_core_pct?: number[];
	load_avg: [number, number, number];
	mem_total_mb: number;
	mem_used_mb: number;
	mem_avail_mb: number;
	disk_total_gb: number;
	disk_used_gb: number;
	disk_free_gb: number;
	gpus: GPUStat[];
	gpu_available: boolean;
}

export interface RunComparison {
	name: string;
	algorithm: string;
	best_metric_value?: number;
	best_metric_iter?: number;
	mean_imp: MeanImpPoint[];
	final_baselines: EvalMetric[];
}

/** GET /training/compare?runs=a,b,c -> 200 response body. */
export interface ComparisonResponse {
	runs: RunComparison[];
}
