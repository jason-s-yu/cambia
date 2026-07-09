// src/pages/TrainingPage.tsx
import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { formatDistanceToNow, parseISO } from 'date-fns';
import { useTrainingStore } from '@/stores/trainingStore';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import StatusBadge from '@/components/training/StatusBadge';
import HostBadge from '@/components/training/HostBadge';
import SyncStatus from '@/components/training/SyncStatus';
import CreateRunModal from '@/components/training/CreateRunModal';
import ResourceMonitor from '@/components/training/ResourceMonitor';
import type { Run, ProcessStatus } from '@/types/training';

const STATUS_OPTIONS = ['', 'running', 'stopped', 'completed', 'failed', 'created'];

const STATUS_BADGE: Record<string, string> = {
	running: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
	completed: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
	stopped: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
	failed: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
	created: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
};

const PROCESS_STATUSES: ProcessStatus[] = [
	'created', 'starting', 'running', 'stopping', 'stopped', 'crashed',
];

// Run.status carries the legacy run_db vocabulary (running/stopped/completed/
// failed/created). Once a process.json exists for the run (tracked live in
// the store's `processes` map, populated after create/start/stop/resume),
// prefer its ProcessStatus rendered via the shared StatusBadge. Runs with no
// live process entry yet (legacy rows, or a status outside the ProcessStatus
// union) fall back to the original inline badge.
function toProcessStatus(status: string): ProcessStatus | null {
	if ((PROCESS_STATUSES as string[]).includes(status)) return status as ProcessStatus;
	if (status === 'completed') return 'stopped';
	if (status === 'failed') return 'crashed';
	return null;
}

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
	const { runs, filters, isLoading, processes, fetchRuns, setFilter } = useTrainingStore();
	const [isCreateOpen, setIsCreateOpen] = useState(false);

	const algorithmOptions = useMemo(
		() => [...new Set(runs.map((r) => r.algorithm))].sort(),
		[runs],
	);

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
			<ResourceMonitor />

			<div className="flex items-center justify-between mb-6">
				<h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
					Training Runs
				</h1>
				<button
					onClick={() => setIsCreateOpen(true)}
					className="px-3 py-1.5 rounded text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white transition-colors"
				>
					New run
				</button>
			</div>

			<CreateRunModal
				isOpen={isCreateOpen}
				onClose={() => setIsCreateOpen(false)}
				onCreated={(name) => navigate(`/training/${name}`)}
			/>

			{/* Filter bar */}
			<div className="flex flex-wrap gap-4 mb-4">
				<select
					value={filters.algorithm}
					onChange={(e) => setFilter('algorithm', e.target.value)}
					className="rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 text-sm px-3 py-1.5"
				>
					<option value="">All Algorithms</option>
					{algorithmOptions.map((a) => (
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
										<td className="py-3 px-4 font-medium">
											<div className="flex items-center gap-2">
												<span>{run.name}</span>
												<HostBadge host={run.host} />
											</div>
										</td>
										<td className="py-3 px-4">
											<span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
												{run.algorithm}
											</span>
										</td>
										<td className="py-3 px-4">
											<div className="flex flex-col gap-1">
												{processes[run.name] ? (
													<StatusBadge status={processes[run.name].status} />
												) : toProcessStatus(run.status) ? (
													<StatusBadge status={toProcessStatus(run.status) as ProcessStatus} />
												) : (
													statusBadge(run.status)
												)}
												<SyncStatus host={run.host} lastSyncAt={run.last_sync_at} stale={run.stale} />
											</div>
										</td>
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
