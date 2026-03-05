/* eslint-disable @typescript-eslint/no-explicit-any */
// src/stores/trainingStore.ts
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import api from '@/lib/axios';
import type { Run, RunDetail, EvalMetric, MeanImpPoint, Checkpoint } from '@/types/training';

const LOG_BUFFER_CAP = 5000;

interface TrainingState {
	runs: Run[];
	selectedRun: RunDetail | null;
	metrics: Record<string, EvalMetric[]>;
	meanImp: Record<string, MeanImpPoint[]>;
	checkpoints: Record<string, Checkpoint[]>;
	logBuffer: string[];
	filters: { algorithm: string; status: string };
	isLoading: boolean;
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

		fetchRuns: async () => {
			set((state) => { state.isLoading = true; });
			try {
				const res = await api.get<Run[]>('/training/runs');
				set((state) => {
					state.runs = res.data;
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
