// src/components/training/EvalControls.tsx
import React, { useEffect, useRef, useState } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import type { EvalJob, TriggerEvalRequest } from '@/types/training';

const POLL_INTERVAL_MS = 2000;
const LATEST = 'latest';
// Stable reference so `evalJobs[runName] ?? EMPTY_JOBS` doesn't create a new
// array (and re-trigger the effects below) on every render before the first
// fetch populates the store for this run.
const EMPTY_JOBS: EvalJob[] = [];

function formatElapsed(startedAt?: string, finishedAt?: string): string {
	if (!startedAt) return '--';
	const start = new Date(startedAt).getTime();
	const end = finishedAt ? new Date(finishedAt).getTime() : Date.now();
	const seconds = Math.max(0, Math.round((end - start) / 1000));
	if (seconds < 60) return `${seconds}s`;
	const minutes = Math.floor(seconds / 60);
	return `${minutes}m ${seconds % 60}s`;
}

function statusDotColor(status: EvalJob['status']): string {
	switch (status) {
		case 'succeeded':
			return 'bg-green-500';
		case 'failed':
			return 'bg-red-500';
		case 'running':
			return 'bg-blue-500';
		default:
			return 'bg-gray-400';
	}
}

interface EvalControlsProps {
	runName: string;
}

const EvalControls: React.FC<EvalControlsProps> = ({ runName }) => {
	const {
		checkpoints,
		evalJobs,
		preflight,
		triggerEval,
		fetchEvalJobs,
		fetchMetrics,
		fetchMeanImp,
	} = useTrainingStore();

	const [target, setTarget] = useState<string>(LATEST);
	const [device, setDevice] = useState<'cpu' | 'cuda'>('cpu');
	const [games, setGames] = useState('5000');
	const [argmax, setArgmax] = useState(false);
	const [force, setForce] = useState(false);
	const [isSubmitting, setIsSubmitting] = useState(false);

	const runCheckpoints = checkpoints[runName] ?? [];
	const jobs = evalJobs[runName] ?? EMPTY_JOBS;
	const hasActiveJob = jobs.some((j) => j.status === 'queued' || j.status === 'running');

	// Tracks each job's last-seen status so a queued/running -> succeeded
	// transition can be detected exactly once per job (see the refetch effect
	// below), independent of the store's 10s ListRuns cache (Phase 2 carry-
	// forward CF, addressed for eval per the sprint plan's S1T7 note).
	const prevStatuses = useRef<Record<string, EvalJob['status']>>({});

	useEffect(() => {
		fetchEvalJobs(runName);
	}, [runName, fetchEvalJobs]);

	useEffect(() => {
		if (!hasActiveJob) return;
		const id = setInterval(() => fetchEvalJobs(runName), POLL_INTERVAL_MS);
		return () => clearInterval(id);
	}, [hasActiveJob, runName, fetchEvalJobs]);

	useEffect(() => {
		const next: Record<string, EvalJob['status']> = {};
		for (const job of jobs) {
			const prev = prevStatuses.current[job.id];
			if (prev !== 'succeeded' && job.status === 'succeeded') {
				fetchMetrics(runName);
				fetchMeanImp(runName);
			}
			next[job.id] = job.status;
		}
		// Replace rather than mutate in place, so ids evicted from the server's
		// retained job history (maxEvalJobsPerRun) are pruned here too; otherwise
		// this ref grows unbounded for a long-lived run detail view.
		prevStatuses.current = next;
	}, [jobs, runName, fetchMetrics, fetchMeanImp]);

	const buildRequest = (overrides: Partial<TriggerEvalRequest> = {}): TriggerEvalRequest => {
		const req: TriggerEvalRequest = { device, argmax };
		if (target !== LATEST) {
			const epoch = Number(target);
			if (Number.isFinite(epoch)) req.epoch = epoch;
		}
		const gamesNum = Number(games);
		if (games.trim() !== '' && Number.isFinite(gamesNum)) req.games = gamesNum;
		if (force) req.force = true;
		return { ...req, ...overrides };
	};

	const handleSubmit = async (e: React.FormEvent) => {
		e.preventDefault();
		setIsSubmitting(true);
		try {
			await triggerEval(runName, buildRequest());
		} finally {
			setIsSubmitting(false);
		}
	};

	const handleRetryWithForce = async () => {
		setForce(true);
		setIsSubmitting(true);
		try {
			await triggerEval(runName, buildRequest({ force: true }));
		} finally {
			setIsSubmitting(false);
		}
	};

	const hasFailingCheck = (preflight ?? []).some((c) => !c.ok);
	const sortedCheckpoints = [...runCheckpoints].sort((a, b) => b.iteration - a.iteration);

	return (
		<div className="space-y-4">
			<form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3">
				<div>
					<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
						Checkpoint
					</label>
					<select
						value={target}
						onChange={(e) => setTarget(e.target.value)}
						disabled={isSubmitting}
						className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
					>
						<option value={LATEST}>Latest</option>
						{sortedCheckpoints.map((cp) => (
							<option key={cp.id} value={cp.iteration}>
								Iter {cp.iteration}
								{cp.is_best ? ' (best)' : ''}
							</option>
						))}
					</select>
				</div>

				<div>
					<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
						Device
					</label>
					<select
						value={device}
						onChange={(e) => setDevice(e.target.value as 'cpu' | 'cuda')}
						disabled={isSubmitting}
						className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
					>
						<option value="cpu">cpu</option>
						<option value="cuda">cuda</option>
					</select>
				</div>

				<div>
					<label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
						Games
					</label>
					<input
						type="number"
						min={1}
						value={games}
						onChange={(e) => setGames(e.target.value)}
						disabled={isSubmitting}
						placeholder="5000"
						className="w-28 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm dark:bg-gray-700 dark:text-white"
					/>
				</div>

				<label className="flex items-center gap-1.5 text-sm text-gray-600 dark:text-gray-400 pb-2">
					<input
						type="checkbox"
						checked={argmax}
						onChange={(e) => setArgmax(e.target.checked)}
						className="rounded border-gray-300 dark:border-gray-600"
					/>
					Argmax
				</label>

				<label className="flex items-center gap-1.5 text-sm text-gray-600 dark:text-gray-400 pb-2">
					<input
						type="checkbox"
						checked={force}
						onChange={(e) => setForce(e.target.checked)}
						className="rounded border-gray-300 dark:border-gray-600"
					/>
					Force
				</label>

				<button
					type="submit"
					disabled={isSubmitting}
					className="px-3 py-1.5 rounded text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
				>
					{isSubmitting ? 'Triggering...' : 'Run eval'}
				</button>
			</form>

			{preflight && preflight.length > 0 && (
				<div className="border border-gray-200 dark:border-gray-700 rounded-md p-3 space-y-2">
					<h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
						Preflight checks
					</h4>
					<ul className="space-y-1">
						{preflight.map((check) => (
							<li key={check.name} className="flex items-start gap-2 text-sm">
								<span
									className={`inline-block w-2 h-2 mt-1.5 rounded-full flex-shrink-0 ${
										check.ok ? 'bg-green-500' : 'bg-red-500'
									}`}
								/>
								<span className="text-gray-700 dark:text-gray-300">
									<span className="font-medium">{check.name}</span>: {check.detail}
								</span>
							</li>
						))}
					</ul>
					{hasFailingCheck && (
						<button
							onClick={handleRetryWithForce}
							disabled={isSubmitting}
							className="px-3 py-1 rounded text-xs font-medium bg-yellow-600 hover:bg-yellow-700 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
						>
							Retry with force
						</button>
					)}
				</div>
			)}

			<div className="space-y-2">
				<h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Eval jobs</h4>
				{jobs.length === 0 && (
					<p className="text-sm text-gray-500 dark:text-gray-400">No eval jobs yet.</p>
				)}
				{jobs.map((job) => (
					<div
						key={job.id}
						className="border border-gray-200 dark:border-gray-700 rounded-md p-3 space-y-1"
					>
						<div className="flex flex-wrap items-center gap-2 text-sm">
							<span
								className={`inline-block w-2 h-2 rounded-full flex-shrink-0 ${statusDotColor(job.status)}`}
							/>
							<span className="font-medium text-gray-800 dark:text-gray-200">{job.status}</span>
							<span className="text-gray-500 dark:text-gray-400">
								{job.target} &middot; {job.device} &middot; {job.games} games
								{job.argmax ? ' · argmax' : ''}
							</span>
							<span className="text-gray-500 dark:text-gray-400 ml-auto">
								{formatElapsed(job.started_at, job.finished_at)}
							</span>
						</div>
						{job.error && <p className="text-sm text-red-600 dark:text-red-400">{job.error}</p>}
						{job.tail && job.tail.length > 0 && (
							<pre className="mt-1 max-h-40 overflow-y-auto text-xs font-mono bg-gray-50 dark:bg-gray-900 text-gray-700 dark:text-gray-300 p-2 rounded">
								{job.tail.join('\n')}
							</pre>
						)}
					</div>
				))}
			</div>
		</div>
	);
};

export default EvalControls;
