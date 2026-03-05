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
