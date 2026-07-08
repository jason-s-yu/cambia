/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stores/trainingStore.ts
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import api from '@/lib/axios';
import type {
	Run,
	RunDetail,
	EvalMetric,
	MeanImpPoint,
	Checkpoint,
	ProcessState,
	PreflightCheck,
	CreateRunRequest,
	CreateRunResponse,
	ProcessActionResponse,
	StartRunOptions,
	StopRunOptions,
	ResumeRunOptions,
} from '@/types/training';

const LOG_BUFFER_CAP = 5000;

/** Pulls a PreflightCheck[] out of a 409 preflight_failed body, if present. */
function extractPreflightChecks(err: any): PreflightCheck[] | null {
	const checks = err?.response?.data?.checks;
	return Array.isArray(checks) ? checks : null;
}

interface TrainingState {
	runs: Run[];
	selectedRun: RunDetail | null;
	metrics: Record<string, EvalMetric[]>;
	meanImp: Record<string, MeanImpPoint[]>;
	checkpoints: Record<string, Checkpoint[]>;
	logBuffer: string[];
	filters: { algorithm: string; status: string };
	isLoading: boolean;
	processes: Record<string, ProcessState>;
	templates: string[];
	preflight: PreflightCheck[] | null;
}

interface TrainingActions {
	fetchRuns: () => Promise<void>;
	fetchRunDetail: (name: string) => Promise<void>;
	fetchMetrics: (name: string) => Promise<void>;
	fetchMeanImp: (name: string) => Promise<void>;
	fetchCheckpoints: (name: string) => Promise<void>;
	appendLogLine: (line: string) => void;
	appendLogBackfill: (lines: string[]) => void;
	clearLogBuffer: () => void;
	setFilter: (key: 'algorithm' | 'status', value: string) => void;
	createRun: (req: CreateRunRequest) => Promise<void>;
	startRun: (name: string, opts?: StartRunOptions) => Promise<void>;
	stopRun: (name: string, opts?: StopRunOptions) => Promise<void>;
	resumeRun: (name: string, opts?: ResumeRunOptions) => Promise<void>;
	fetchTemplates: () => Promise<void>;
}

export const useTrainingStore = create<TrainingState & TrainingActions>()(
	immer((set) => ({
		runs: [],
		selectedRun: null,
		metrics: {},
		meanImp: {},
		checkpoints: {},
		logBuffer: [],
		filters: { algorithm: '', status: '' },
		isLoading: false,
		processes: {},
		templates: [],
		preflight: null,

		fetchRuns: async () => {
			set((state) => { state.isLoading = true; });
			try {
				const res = await api.get<Run[]>('/training/runs');
				set((state) => {
					state.runs = res.data;
					// Hydrate the processes map from each run's embedded process
					// record so process-derived UI (status badges, start/stop
					// controls) reflects the list response without a separate fetch.
					for (const run of res.data) {
						if (run.process) {
							state.processes[run.name] = run.process;
						}
					}
					state.isLoading = false;
				});
			} catch (err: any) {
				console.error('Failed to fetch runs:', err);
				set((state) => { state.isLoading = false; });
			}
		},

		fetchRunDetail: async (name: string) => {
			set((state) => { state.isLoading = true; });
			try {
				const res = await api.get<RunDetail>(`/training/runs/${name}`);
				set((state) => {
					state.selectedRun = res.data;
					if (res.data.process) {
						state.processes[res.data.name] = res.data.process;
					}
					state.isLoading = false;
				});
			} catch (err: any) {
				console.error('Failed to fetch run detail:', err);
				set((state) => { state.isLoading = false; });
			}
		},

		fetchMetrics: async (name: string) => {
			try {
				const res = await api.get<EvalMetric[]>(`/training/runs/${name}/metrics`);
				set((state) => {
					state.metrics[name] = res.data;
				});
			} catch (err: any) {
				console.error('Failed to fetch metrics:', err);
			}
		},

		fetchMeanImp: async (name: string) => {
			try {
				const res = await api.get<MeanImpPoint[]>(`/training/runs/${name}/metrics`, {
					params: { aggregate: 'mean_imp' },
				});
				set((state) => {
					state.meanImp[name] = res.data;
				});
			} catch (err: any) {
				console.error('Failed to fetch mean_imp:', err);
			}
		},

		fetchCheckpoints: async (name: string) => {
			try {
				const res = await api.get<Checkpoint[]>(`/training/runs/${name}/checkpoints`);
				set((state) => {
					state.checkpoints[name] = res.data;
				});
			} catch (err: any) {
				console.error('Failed to fetch checkpoints:', err);
			}
		},

		createRun: async (req: CreateRunRequest) => {
			set((state) => { state.isLoading = true; });
			try {
				const res = await api.post<CreateRunResponse>('/training/runs', req);
				set((state) => {
					state.runs.push(res.data.run);
					state.processes[res.data.process.name] = res.data.process;
					state.preflight = null;
					state.isLoading = false;
				});
			} catch (err: any) {
				console.error('Failed to create run:', err);
				set((state) => {
					state.isLoading = false;
					state.preflight = extractPreflightChecks(err);
				});
			}
		},

		startRun: async (name: string, opts: StartRunOptions = {}) => {
			set((state) => { state.isLoading = true; });
			try {
				const res = await api.post<ProcessActionResponse>(`/training/runs/${name}/start`, opts);
				set((state) => {
					state.processes[name] = res.data.process;
					state.preflight = null;
					state.isLoading = false;
				});
			} catch (err: any) {
				console.error(`Failed to start run ${name}:`, err);
				set((state) => {
					state.isLoading = false;
					state.preflight = extractPreflightChecks(err);
				});
			}
		},

		stopRun: async (name: string, opts: StopRunOptions = {}) => {
			set((state) => { state.isLoading = true; });
			try {
				const res = await api.post<ProcessActionResponse>(`/training/runs/${name}/stop`, opts);
				set((state) => {
					state.processes[name] = res.data.process;
					state.preflight = null;
					state.isLoading = false;
				});
			} catch (err: any) {
				console.error(`Failed to stop run ${name}:`, err);
				set((state) => {
					state.isLoading = false;
					state.preflight = extractPreflightChecks(err);
				});
			}
		},

		resumeRun: async (name: string, opts: ResumeRunOptions = {}) => {
			set((state) => { state.isLoading = true; });
			try {
				const res = await api.post<ProcessActionResponse>(`/training/runs/${name}/resume`, opts);
				set((state) => {
					state.processes[name] = res.data.process;
					state.preflight = null;
					state.isLoading = false;
				});
			} catch (err: any) {
				console.error(`Failed to resume run ${name}:`, err);
				set((state) => {
					state.isLoading = false;
					state.preflight = extractPreflightChecks(err);
				});
			}
		},

		fetchTemplates: async () => {
			try {
				const res = await api.get<string[]>('/training/config/templates');
				set((state) => {
					state.templates = res.data;
				});
			} catch (err: any) {
				console.error('Failed to fetch templates:', err);
			}
		},

		appendLogLine: (line: string) => {
			set((state) => {
				state.logBuffer.push(line);
				while (state.logBuffer.length > LOG_BUFFER_CAP) {
					state.logBuffer.shift();
				}
			});
		},

		appendLogBackfill: (lines: string[]) => {
			set((state) => {
				state.logBuffer = [...lines, ...state.logBuffer];
				while (state.logBuffer.length > LOG_BUFFER_CAP) {
					state.logBuffer.shift();
				}
			});
		},

		clearLogBuffer: () => {
			set((state) => { state.logBuffer = []; });
		},

		setFilter: (key: 'algorithm' | 'status', value: string) => {
			set((state) => { state.filters[key] = value; });
		},
	}))
);
