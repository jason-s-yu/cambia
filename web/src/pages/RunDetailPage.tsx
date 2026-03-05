// src/pages/RunDetailPage.tsx
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { formatDistanceToNow, parseISO } from 'date-fns';
import { useTrainingStore } from '@/stores/trainingStore';
import { useTrainingSocket } from '@/hooks/useTrainingSocket';
import MetricsChart from '@/components/training/MetricsChart';
import CheckpointTable from '@/components/training/CheckpointTable';
import LogViewer from '@/components/training/LogViewer';
import LoadingSpinner from '@/components/common/LoadingSpinner';

type Tab = 'metrics' | 'checkpoints' | 'logs';

const STATUS_BADGE: Record<string, string> = {
	running: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
	completed: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200',
	stopped: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200',
	failed: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
	created: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
};

const RunDetailPage: React.FC = () => {
	const { runName } = useParams<{ runName: string }>();
	const [activeTab, setActiveTab] = useState<Tab>('metrics');

	const {
		selectedRun,
		metrics,
		meanImp,
		checkpoints,
		isLoading,
		fetchRunDetail,
		fetchMetrics,
		fetchMeanImp,
		fetchCheckpoints,
	} = useTrainingStore();

	const { connected } = useTrainingSocket(runName);

	useEffect(() => {
		if (!runName) return;
		fetchRunDetail(runName);
		fetchMetrics(runName);
		fetchMeanImp(runName);
		fetchCheckpoints(runName);
	}, [runName, fetchRunDetail, fetchMetrics, fetchMeanImp, fetchCheckpoints]);

	if (isLoading && !selectedRun) {
		return (
			<div className="flex justify-center py-12">
				<LoadingSpinner />
			</div>
		);
	}

	if (!selectedRun) {
		return (
			<div className="text-gray-500 dark:text-gray-400 text-center py-12">
				Run not found.
			</div>
		);
	}

	const run = selectedRun;
	const runMetrics = runName ? metrics[runName] ?? [] : [];
	const runMeanImp = runName ? meanImp[runName] ?? [] : [];
	const runCheckpoints = runName ? checkpoints[runName] ?? [] : [];

	const statusCls = STATUS_BADGE[run.status] ?? STATUS_BADGE.created;

	const tabs: { key: Tab; label: string }[] = [
		{ key: 'metrics', label: 'Metrics' },
		{ key: 'checkpoints', label: 'Checkpoints' },
		{ key: 'logs', label: 'Logs' },
	];

	return (
		<div>
			{/* Header */}
			<div className="mb-6">
				<div className="flex items-center gap-3 mb-2">
					<h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
						{run.name}
					</h1>
					<span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
						{run.algorithm}
					</span>
					<span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${statusCls}`}>
						{run.status}
					</span>
				</div>
				<div className="text-sm text-gray-500 dark:text-gray-400 flex gap-4">
					<span>Created {formatDistanceToNow(parseISO(run.created_at), { addSuffix: true })}</span>
					<span>Updated {formatDistanceToNow(parseISO(run.updated_at), { addSuffix: true })}</span>
				</div>
			</div>

			{/* Tab bar */}
			<div className="border-b border-gray-200 dark:border-gray-700 mb-6">
				<nav className="flex gap-6">
					{tabs.map((tab) => (
						<button
							key={tab.key}
							onClick={() => setActiveTab(tab.key)}
							className={`pb-2 text-sm font-medium border-b-2 transition-colors ${
								activeTab === tab.key
									? 'border-blue-600 text-blue-600 dark:border-blue-400 dark:text-blue-400'
									: 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'
							}`}
						>
							{tab.label}
						</button>
					))}
				</nav>
			</div>

			{/* Tab content */}
			{activeTab === 'metrics' && (
				<MetricsChart metrics={runMetrics} meanImp={runMeanImp} />
			)}
			{activeTab === 'checkpoints' && (
				<CheckpointTable checkpoints={runCheckpoints} />
			)}
			{activeTab === 'logs' && (
				<LogViewer connected={connected} />
			)}
		</div>
	);
};

export default RunDetailPage;
