// src/pages/TrainingPage.tsx
import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { formatDistanceToNow, parseISO } from 'date-fns';
import { useTrainingStore } from '@/stores/trainingStore';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import type { Run } from '@/types/training';

const ALGORITHM_OPTIONS = ['', 'deep_cfr', 'rebel', 'gtcfr', 'sog', 'ppo'];
const STATUS_OPTIONS = ['', 'running', 'stopped', 'completed', 'failed', 'created'];

const STATUS_BADGE: Record<string, string> = {
	running: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
	completed: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
	stopped: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
	failed: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
	created: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
};

function statusBadge(status: string) {
	const cls = STATUS_BADGE[status] ?? STATUS_BADGE.created;
	return (
		<span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${cls}`}>
			{status}
		</span>
	);
}

const TrainingPage: React.FC = () => {
	const navigate = useNavigate();
	const { runs, filters, isLoading, fetchRuns, setFilter } = useTrainingStore();

	useEffect(() => {
		fetchRuns();
		const interval = setInterval(fetchRuns, 10_000);
		return () => clearInterval(interval);
	}, [fetchRuns]);

	const filtered = runs.filter((r: Run) => {
		if (filters.algorithm && r.algorithm !== filters.algorithm) return false;
		if (filters.status && r.status !== filters.status) return false;
		return true;
	});

	return (
		<div>
			<h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-6">
				Training Runs
			</h1>

			{/* Filter bar */}
			<div className="flex flex-wrap gap-4 mb-4">
				<select
					value={filters.algorithm}
					onChange={(e) => setFilter('algorithm', e.target.value)}
					className="rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm px-3 py-1.5"
				>
					<option value="">All Algorithms</option>
					{ALGORITHM_OPTIONS.filter(Boolean).map((a) => (
						<option key={a} value={a}>{a}</option>
					))}
				</select>
				<select
					value={filters.status}
					onChange={(e) => setFilter('status', e.target.value)}
					className="rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm px-3 py-1.5"
				>
					<option value="">All Statuses</option>
					{STATUS_OPTIONS.filter(Boolean).map((s) => (
						<option key={s} value={s}>{s}</option>
					))}
				</select>
			</div>

			{isLoading && runs.length === 0 ? (
				<div className="flex justify-center py-12">
					<LoadingSpinner />
				</div>
			) : (
				<div className="overflow-x-auto bg-white dark:bg-gray-800 rounded-lg shadow">
					<table className="min-w-full text-sm">
						<thead>
							<tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-400">
								<th className="py-3 px-4 font-medium">Name</th>
								<th className="py-3 px-4 font-medium">Algorithm</th>
								<th className="py-3 px-4 font-medium">Status</th>
								<th className="py-3 px-4 font-medium">Best Metric</th>
								<th className="py-3 px-4 font-medium">Last Updated</th>
							</tr>
						</thead>
						<tbody>
							{filtered.length === 0 ? (
								<tr>
									<td colSpan={5} className="py-8 text-center text-gray-500 dark:text-gray-400">
										No runs found.
									</td>
								</tr>
							) : (
								filtered.map((run) => (
									<tr
										key={run.id}
										onClick={() => navigate(`/training/${run.name}`)}
										className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-750 cursor-pointer transition-colors text-gray-800 dark:text-gray-200"
									>
										<td className="py-3 px-4 font-medium">{run.name}</td>
										<td className="py-3 px-4">
											<span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
												{run.algorithm}
											</span>
										</td>
										<td className="py-3 px-4">{statusBadge(run.status)}</td>
										<td className="py-3 px-4 font-mono">
											{run.best_metric_value != null
												? `${(run.best_metric_value * 100).toFixed(1)}%`
												: '--'}
											{run.best_metric_iter != null && (
												<span className="text-gray-400 ml-1 text-xs">
													@{run.best_metric_iter}
												</span>
											)}
										</td>
										<td className="py-3 px-4 text-gray-500 dark:text-gray-400">
											{formatDistanceToNow(parseISO(run.updated_at), { addSuffix: true })}
										</td>
									</tr>
								))
							)}
						</tbody>
					</table>
				</div>
			)}
		</div>
	);
};

export default TrainingPage;
