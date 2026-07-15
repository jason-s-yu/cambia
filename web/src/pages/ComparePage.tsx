// src/pages/ComparePage.tsx
import React, { useEffect, useState } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';
import ComparisonChart from '@/components/training/ComparisonChart';
import PageContainer from '@/components/common/PageContainer';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import type { RunComparison } from '@/types/training';

const MIN_RUNS = 2;
const MAX_RUNS = 6;

function formatPct(value: number | null | undefined): string {
	return value != null ? `${(value * 100).toFixed(1)}%` : '--';
}

function isNoData(r: RunComparison): boolean {
	return r.mean_imp.length === 0 && r.final_baselines.length === 0 && r.best_metric_value == null;
}

const ComparePage: React.FC = () => {
	const { runs, comparison, isLoading, fetchRuns, fetchComparison } = useTrainingStore();
	const [selected, setSelected] = useState<string[]>([]);

	useEffect(() => {
		fetchRuns();
	}, [fetchRuns]);

	const toggle = (name: string) => {
		setSelected((prev) => {
			if (prev.includes(name)) return prev.filter((n) => n !== name);
			if (prev.length >= MAX_RUNS) return prev;
			return [...prev, name];
		});
	};

	const canCompare = selected.length >= MIN_RUNS && selected.length <= MAX_RUNS;

	useEffect(() => {
		if (canCompare) {
			fetchComparison(selected);
		}
	}, [selected, canCompare, fetchComparison]);

	const comparedRuns = comparison?.runs ?? [];

	const baselineNames = Array.from(
		new Set(comparedRuns.flatMap((r) => r.final_baselines.map((fb) => fb.baseline))),
	).sort();

	return (
		<PageContainer>
			<div className="mb-6">
				<h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
					Compare Runs
				</h1>
				<p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
					Select 2 to 6 runs to overlay their mean_imp trajectories and compare final baseline win rates.
				</p>
			</div>

			<div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6">
				<div className="flex items-center justify-between mb-3">
					<h2 className="text-sm font-semibold text-gray-700 dark:text-gray-300">
						Runs ({selected.length}/{MAX_RUNS})
					</h2>
				</div>
				{isLoading && runs.length === 0 ? (
					<div className="flex justify-center py-6">
						<LoadingSpinner size="sm" />
					</div>
				) : runs.length === 0 ? (
					<p className="text-gray-500 dark:text-gray-400 text-sm">No runs found.</p>
				) : (
					<div className="flex flex-wrap gap-2 max-h-48 overflow-y-auto">
						{runs.map((r) => {
							const checked = selected.includes(r.name);
							const disabled = !checked && selected.length >= MAX_RUNS;
							return (
								<label
									key={r.name}
									className={`inline-flex items-center gap-2 px-2.5 py-1.5 rounded border text-sm transition-colors ${
										checked
											? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200'
											: 'border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300'
									} ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer hover:border-blue-400'}`}
								>
									<input
										type="checkbox"
										checked={checked}
										disabled={disabled}
										onChange={() => toggle(r.name)}
										className="accent-blue-600"
									/>
									{r.name}
									<span className="text-xs text-gray-400">{r.algorithm}</span>
								</label>
							);
						})}
					</div>
				)}
				{selected.length > 0 && selected.length < MIN_RUNS && (
					<p className="text-xs text-amber-600 dark:text-amber-400 mt-2">
						Select at least {MIN_RUNS} runs to compare.
					</p>
				)}
			</div>

			{canCompare && (
				<>
					<div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 mb-6">
						<h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
							mean_imp Trajectories
						</h3>
						<ComparisonChart runs={comparedRuns} />
					</div>

					<div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4 overflow-x-auto">
						<h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
							Final Metrics
						</h3>
						<table className="min-w-full text-sm">
							<thead>
								<tr className="border-b border-gray-200 dark:border-gray-700 text-left text-gray-600 dark:text-gray-400">
									<th className="py-2 px-3 font-medium">Metric</th>
									{comparedRuns.map((r) => (
										<th key={r.name} className="py-2 px-3 font-medium">
											{r.name}
											{isNoData(r) && (
												<span className="ml-1 text-xs font-normal text-gray-400">(no data)</span>
											)}
										</th>
									))}
								</tr>
							</thead>
							<tbody>
								<tr className="border-b border-gray-100 dark:border-gray-700">
									<td className="py-2 px-3 text-gray-500 dark:text-gray-400">Algorithm</td>
									{comparedRuns.map((r) => (
										<td key={r.name} className="py-2 px-3">{r.algorithm || '--'}</td>
									))}
								</tr>
								<tr className="border-b border-gray-100 dark:border-gray-700">
									<td className="py-2 px-3 text-gray-500 dark:text-gray-400">Best Metric</td>
									{comparedRuns.map((r) => (
										<td key={r.name} className="py-2 px-3 font-mono">
											{formatPct(r.best_metric_value)}
											{r.best_metric_iter != null && (
												<span className="text-gray-400 ml-1 text-xs">@{r.best_metric_iter}</span>
											)}
										</td>
									))}
								</tr>
								{baselineNames.map((bl) => (
									<tr key={bl} className="border-b border-gray-100 dark:border-gray-700">
										<td className="py-2 px-3 text-gray-500 dark:text-gray-400">{bl}</td>
										{comparedRuns.map((r) => {
											const m = r.final_baselines.find((fb) => fb.baseline === bl);
											return (
												<td key={r.name} className="py-2 px-3 font-mono">
													{formatPct(m?.win_rate)}
												</td>
											);
										})}
									</tr>
								))}
								{baselineNames.length === 0 && (
									<tr>
										<td
											colSpan={comparedRuns.length + 1}
											className="py-4 px-3 text-center text-gray-500 dark:text-gray-400"
										>
											No baseline eval data available for the selected runs.
										</td>
									</tr>
								)}
							</tbody>
						</table>
					</div>
				</>
			)}
		</PageContainer>
	);
};

export default ComparePage;
