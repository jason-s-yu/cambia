export interface Run {
	id: number;
	name: string;
	algorithm: string;
	status: string;
	best_metric_value: number | null;
	best_metric_iter: number | null;
	created_at: string;
	updated_at: string;
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
