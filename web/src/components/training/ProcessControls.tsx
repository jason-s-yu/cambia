// src/components/training/ProcessControls.tsx
import React, { useState } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import type { ProcessState } from '@/types/training';

type LastAction = 'start' | 'stop' | 'resume' | null;

interface ProcessControlsProps {
	processState: ProcessState;
	onChanged?: () => void;
}

const ProcessControls: React.FC<ProcessControlsProps> = ({ processState, onChanged }) => {
	const { startRun, stopRun, resumeRun, preflight, processes } = useTrainingStore();
	const [force, setForce] = useState(false);
	const [lastAction, setLastAction] = useState<LastAction>(null);
	const [isBusy, setIsBusy] = useState(false);

	// The store keeps a live process map keyed by name, updated in place after
	// every start/stop/resume response; prefer it over the possibly-stale prop
	// so the gating below reflects the outcome of an action immediately.
	const live = processes[processState.name] ?? processState;

	const runAction = async (action: Exclude<LastAction, null>, fn: () => Promise<void>) => {
		setIsBusy(true);
		setLastAction(action);
		try {
			await fn();
		} finally {
			setIsBusy(false);
			onChanged?.();
		}
	};

	const handleStart = () => runAction('start', () => startRun(live.name, { force }));
	const handleStop = () => runAction('stop', () => stopRun(live.name, { force }));
	const handleResume = () => runAction('resume', () => resumeRun(live.name, { force }));

	const handleRetryWithForce = () => {
		setForce(true);
		if (lastAction === 'start') runAction('start', () => startRun(live.name, { force: true }));
		else if (lastAction === 'resume') runAction('resume', () => resumeRun(live.name, { force: true }));
		else if (lastAction === 'stop') runAction('stop', () => stopRun(live.name, { force: true }));
	};

	const canStart = live.status === 'created';
	const canStop = live.status === 'running' || live.status === 'starting';
	const canResume = live.status === 'stopped' || live.status === 'crashed';
	const hasFailingCheck = (preflight ?? []).some((c) => !c.ok);

	return (
		<div className="space-y-3">
			<div className="flex flex-wrap items-center gap-3">
				{canStart && (
					<button
						onClick={handleStart}
						disabled={isBusy}
						className="px-3 py-1.5 rounded text-sm font-medium bg-blue-600 hover:bg-blue-700 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
					>
						{isBusy && lastAction === 'start' ? 'Starting...' : 'Start'}
					</button>
				)}
				{canStop && (
					<button
						onClick={handleStop}
						disabled={isBusy}
						className="px-3 py-1.5 rounded text-sm font-medium bg-red-600 hover:bg-red-700 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
					>
						{isBusy && lastAction === 'stop' ? 'Stopping...' : 'Stop'}
					</button>
				)}
				{canResume && (
					<button
						onClick={handleResume}
						disabled={isBusy}
						className="px-3 py-1.5 rounded text-sm font-medium bg-green-600 hover:bg-green-700 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
					>
						{isBusy && lastAction === 'resume' ? 'Resuming...' : 'Resume'}
					</button>
				)}
				{!canStart && !canStop && !canResume && (
					<span className="text-sm text-gray-500 dark:text-gray-400">
						No actions available while {live.status}.
					</span>
				)}

				<label className="flex items-center gap-1.5 text-sm text-gray-600 dark:text-gray-400 ml-2">
					<input
						type="checkbox"
						checked={force}
						onChange={(e) => setForce(e.target.checked)}
						className="rounded border-gray-300 dark:border-gray-600"
					/>
					Force
				</label>
			</div>

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
							disabled={isBusy}
							className="px-3 py-1 rounded text-xs font-medium bg-yellow-600 hover:bg-yellow-700 text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
						>
							Retry with force
						</button>
					)}
				</div>
			)}
		</div>
	);
};

export default ProcessControls;
