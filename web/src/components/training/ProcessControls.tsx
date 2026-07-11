// src/components/training/ProcessControls.tsx
import React, { useState } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import type { ProcessState } from '@/types/training';
import Modal from '@/components/common/Modal';

type LastAction = 'start' | 'stop' | 'resume' | null;

interface ProcessControlsProps {
	processState: ProcessState;
	onChanged?: () => void;
	/** True when this run is a remote (serving-harness) run controllable via the
	 * dashboard's harness proxy: start/stop/resume are forwarded to the runner and
	 * the dashboard reflects the result on the next sync, so the controls show a
	 * "requested / pending sync" note instead of an immediate state flip. */
	remote?: boolean;
}

const ProcessControls: React.FC<ProcessControlsProps> = ({ processState, onChanged, remote = false }) => {
	const { startRun, stopRun, resumeRun, preflight, processes, actionError } = useTrainingStore();
	const [force, setForce] = useState(false);
	const [lastAction, setLastAction] = useState<LastAction>(null);
	const [isBusy, setIsBusy] = useState(false);
	// A stop is destructive (it can cancel a multi-day training run, local or
	// remote), so it is gated behind an explicit confirmation. pendingStopForce
	// carries the force flag the confirmed stop will use (null = modal closed).
	const [pendingStopForce, setPendingStopForce] = useState<boolean | null>(null);

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
	const handleResume = () => runAction('resume', () => resumeRun(live.name, { force }));

	// Stop routes through the confirmation modal (both local and remote); the
	// confirmed stop uses pendingStopForce.
	const requestStop = (stopForce: boolean) => setPendingStopForce(stopForce);
	const cancelStop = () => setPendingStopForce(null);
	const confirmStop = () => {
		const f = pendingStopForce ?? false;
		setPendingStopForce(null);
		setForce(f);
		runAction('stop', () => stopRun(live.name, { force: f }));
	};

	const handleRetryWithForce = () => {
		if (lastAction === 'start') { setForce(true); runAction('start', () => startRun(live.name, { force: true })); }
		else if (lastAction === 'resume') { setForce(true); runAction('resume', () => resumeRun(live.name, { force: true })); }
		else if (lastAction === 'stop') { requestStop(true); }
	};

	const canStart = live.status === 'created';
	const canStop = live.status === 'running' || live.status === 'starting';
	const canResume = live.status === 'stopped' || live.status === 'crashed';
	const hasFailingCheck = (preflight ?? []).some((c) => !c.ok);

	return (
		<div className="space-y-3">
			{remote && (
				<p className="text-xs text-purple-700 dark:text-purple-300">
					Remote run: actions are forwarded to the origin host. This dashboard
					reflects the result on the next sync (~60s).
				</p>
			)}

			<div className="flex flex-wrap items-center gap-3">
				{canStart && !remote && (
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
						onClick={() => requestStop(force)}
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

			{actionError && (
				<div className="border border-red-300 dark:border-red-800 bg-red-50 dark:bg-red-950 rounded-md p-2 text-sm text-red-700 dark:text-red-300">
					{actionError}
				</div>
			)}

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

			<Modal isOpen={pendingStopForce !== null} onClose={cancelStop} title="Stop this run?">
				<p className="text-sm text-gray-600 dark:text-gray-300">
					{remote
						? `This forwards a ${pendingStopForce ? 'force-kill (SIGKILL)' : 'graceful stop (SIGINT)'} to the origin host and interrupts the training run.`
						: `This ${pendingStopForce ? 'force-kills (SIGKILL)' : 'gracefully stops (SIGINT)'} the training process.`}
					{' '}It can interrupt a multi-day run. Resuming continues from the last checkpoint.
				</p>
				<div className="mt-5 flex justify-end gap-2">
					<button
						onClick={cancelStop}
						className="px-3 py-1.5 rounded text-sm font-medium bg-gray-200 hover:bg-gray-300 text-gray-800 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-100 transition-colors"
					>
						Cancel
					</button>
					<button
						onClick={confirmStop}
						className="px-3 py-1.5 rounded text-sm font-medium bg-red-600 hover:bg-red-700 text-white transition-colors"
					>
						{pendingStopForce ? 'Force stop' : 'Stop'}
					</button>
				</div>
			</Modal>
		</div>
	);
};

export default ProcessControls;
