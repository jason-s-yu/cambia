// src/pages/RunDetailPage.tsx
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { formatDistanceToNow, parseISO } from 'date-fns';
import { useTrainingStore } from '@/stores/trainingStore';
import { useTrainingSocket } from '@/hooks/useTrainingSocket';
import PageContainer from '@/components/common/PageContainer';
import MetricsChart from '@/components/training/MetricsChart';
import CheckpointTable from '@/components/training/CheckpointTable';
import LogViewer from '@/components/training/LogViewer';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import StatusBadge from '@/components/training/StatusBadge';
import HostBadge from '@/components/training/HostBadge';
import SyncStatus from '@/components/training/SyncStatus';
import ProcessControls from '@/components/training/ProcessControls';
import EvalControls from '@/components/training/EvalControls';
import type { ProcessStatus, ProcessState } from '@/types/training';

type Tab = 'metrics' | 'checkpoints' | 'logs' | 'evaluate';

const PROCESS_STATUSES: ProcessStatus[] = [
	'created', 'starting', 'running', 'stopping', 'stopped', 'crashed',
];

// Run.status carries the legacy run_db vocabulary (running/stopped/completed/
// failed/created). Map it onto the ProcessStatus union so the shared
// StatusBadge and ProcessControls (which gate their actions on
// ProcessStatus) can render a sensible best guess before any process action
// has populated the store's live `processes` map for this run.
function toProcessStatus(status: string): ProcessStatus {
	if ((PROCESS_STATUSES as string[]).includes(status)) return status as ProcessStatus;
	if (status === 'completed') return 'stopped';
	if (status === 'failed') return 'crashed';
	return 'created';
}

const RunDetailPage: React.FC = () => {
	const { runName } = useParams<{ runName: string }>();
	const [activeTab, setActiveTab] = useState<Tab>('metrics');

	const {
		selectedRun,
		metrics,
		meanImp,
		checkpoints,
		processes,
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
			<PageContainer>
				<div className="flex justify-center py-12">
					<LoadingSpinner />
				</div>
			</PageContainer>
		);
	}

	if (!selectedRun) {
		return (
			<PageContainer>
				<div className="text-gray-500 dark:text-gray-400 text-center py-12">
					Run not found.
				</div>
			</PageContainer>
		);
	}

	const run = selectedRun;
	const runMetrics = runName ? metrics[runName] ?? [] : [];
	const runMeanImp = runName ? meanImp[runName] ?? [] : [];
	const runCheckpoints = runName ? checkpoints[runName] ?? [] : [];

	// Prefer the live process.json-backed state (populated after any
	// create/start/stop/resume action); fall back to a best-guess ProcessState
	// derived from the run_db row so ProcessControls has something to gate on
	// before the first action of this session.
	const liveProcess = processes[run.name];
	const fallbackProcessState: ProcessState = {
		name: run.name,
		status: toProcessStatus(run.status),
		algorithm: run.algorithm,
		pid: 0,
		pgid: 0,
		config_path: '',
		created_at: run.created_at,
	};
	const processState = liveProcess ?? fallbackProcessState;

	// A remote (serving-harness) run is read-only unless its origin matches the
	// dashboard's configured harness proxy (remote_controllable), in which case
	// stop/resume are forwarded to the runner (cambia-295 v1.1). The host comes
	// from the run detail or a stamped process record.
	const isRemote = Boolean(run.host || processState.host);
	const remoteControllable = Boolean(run.remote_controllable);

	const tabs: { key: Tab; label: string }[] = [
		{ key: 'metrics', label: 'Metrics' },
		{ key: 'checkpoints', label: 'Checkpoints' },
		{ key: 'logs', label: 'Logs' },
		{ key: 'evaluate', label: 'Evaluate' },
	];

	return (
		<PageContainer>
			{/* Header */}
			<div className="mb-6">
				<div className="flex items-center gap-3 mb-2 flex-wrap">
					<h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
						{run.name}
					</h1>
					<span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-indigo-100 text-indigo-800 dark:bg-indigo-900 dark:text-indigo-200">
						{run.algorithm}
					</span>
					<StatusBadge status={processState.status} />
					<HostBadge host={run.host} />
					<SyncStatus host={run.host} lastSyncAt={run.last_sync_at} stale={run.stale} />
				</div>
				<div className="text-sm text-gray-500 dark:text-gray-400 flex gap-4">
					<span>Created {formatDistanceToNow(parseISO(run.created_at), { addSuffix: true })}</span>
					<span>Updated {formatDistanceToNow(parseISO(run.updated_at), { addSuffix: true })}</span>
				</div>
			</div>

			{/* Process controls: mounted outside the tab switcher so start/stop/
			    resume and any preflight-block panel stay visible regardless of
			    the active tab. A remote run whose origin matches the configured
			    harness proxy gets forwarded stop/resume controls; a remote run
			    from an unconfigured origin stays read-only (its controls run on
			    the origin host). */}
			<div className="mb-6 bg-white dark:bg-gray-800 rounded-lg shadow p-4">
				{isRemote && !remoteControllable ? (
					<p className="text-sm text-gray-500 dark:text-gray-400">
						Remote run on {run.host}: read-only on this dashboard. Process controls
						(start, stop, resume) run on the origin host.
					</p>
				) : (
					<ProcessControls
						processState={processState}
						remote={isRemote && remoteControllable}
						onChanged={() => runName && fetchRunDetail(runName)}
					/>
				)}
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
			{activeTab === 'evaluate' && runName && (
				<EvalControls runName={runName} />
			)}
		</PageContainer>
	);
};

export default RunDetailPage;
