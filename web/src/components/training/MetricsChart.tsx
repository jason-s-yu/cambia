// src/components/training/MetricsChart.tsx
import React, { useMemo, useRef, useEffect, useState } from 'react';
import UplotReact from 'uplot-react';
import type uPlot from 'uplot';
import type { EvalMetric, MeanImpPoint } from '@/types/training';

interface MetricsChartProps {
	metrics: EvalMetric[];
	meanImp: MeanImpPoint[];
}

const BASELINE_COLORS: Record<string, string> = {
	random_no_cambia: '#3b82f6',   // blue
	random_late_cambia: '#22c55e', // green
	imperfect_greedy: '#f97316',   // orange
	memory_heuristic: '#ef4444',   // red
	aggressive_snap: '#a855f7',    // purple
	random: '#6b7280',             // gray
	greedy: '#ec4899',             // pink
};

function isDarkMode(): boolean {
	return document.documentElement.classList.contains('dark');
}

const MetricsChart: React.FC<MetricsChartProps> = ({ metrics, meanImp }) => {
	const containerRef = useRef<HTMLDivElement>(null);
	const [width, setWidth] = useState(800);

	useEffect(() => {
		if (!containerRef.current) return;
		const observer = new ResizeObserver((entries) => {
			for (const entry of entries) {
				setWidth(entry.contentRect.width);
			}
		});
		observer.observe(containerRef.current);
		return () => observer.disconnect();
	}, []);

	// Convergence chart data
	const convergenceData = useMemo(() => {
		const dark = isDarkMode();
		const axisColor = dark ? '#9ca3af' : '#6b7280';
		const gridColor = dark ? '#374151' : '#e5e7eb';

		// Group metrics by baseline
		const byBaseline = new Map<string, Map<number, number>>();
		for (const m of metrics) {
			if (m.win_rate == null) continue;
			if (!byBaseline.has(m.baseline)) byBaseline.set(m.baseline, new Map());
			byBaseline.get(m.baseline)!.set(m.iteration, m.win_rate * 100);
		}

		// mean_imp as its own series
		const meanImpMap = new Map<number, number>();
		for (const p of meanImp) {
			meanImpMap.set(p.iteration, p.mean_imp * 100);
		}

		// Collect all iterations
		const iterSet = new Set<number>();
		for (const map of byBaseline.values()) for (const k of map.keys()) iterSet.add(k);
		for (const k of meanImpMap.keys()) iterSet.add(k);
		const iterations = Array.from(iterSet).sort((a, b) => a - b);

		if (iterations.length === 0) {
			return { opts: null, data: null };
		}

		const baselines = Array.from(byBaseline.keys()).sort();
		const series: uPlot.Series[] = [{ label: 'Iteration' }];
		const dataArrays: (number | null)[][] = [iterations.map(Number)];

		for (const bl of baselines) {
			const map = byBaseline.get(bl)!;
			series.push({
				label: bl,
				stroke: BASELINE_COLORS[bl] ?? '#64748b',
				width: 1.5,
			});
			dataArrays.push(iterations.map((it) => map.get(it) ?? null));
		}

		// mean_imp series (bold)
		series.push({
			label: 'mean_imp',
			stroke: dark ? '#f9fafb' : '#111827',
			width: 3,
		});
		dataArrays.push(iterations.map((it) => meanImpMap.get(it) ?? null));

		const opts: uPlot.Options = {
			width,
			height: 350,
			scales: { x: { time: false }, y: { auto: true } },
			series,
			axes: [
				{ label: 'Iteration', stroke: axisColor, grid: { stroke: gridColor } },
				{ label: 'Win Rate (%)', stroke: axisColor, grid: { stroke: gridColor } },
			],
			cursor: { drag: { x: true, y: false } },
		};

		return { opts, data: dataArrays as uPlot.AlignedData };
	}, [metrics, meanImp, width]);

	// Loss chart data
	const lossData = useMemo(() => {
		const dark = isDarkMode();
		const axisColor = dark ? '#9ca3af' : '#6b7280';
		const gridColor = dark ? '#374151' : '#e5e7eb';

		// Deduplicate: one row per iteration
		const iterMap = new Map<number, { adv: number | null; strat: number | null }>();
		for (const m of metrics) {
			if (!iterMap.has(m.iteration)) {
				iterMap.set(m.iteration, { adv: m.adv_loss, strat: m.strat_loss });
			}
		}

		const iterations = Array.from(iterMap.keys()).sort((a, b) => a - b);
		if (iterations.length === 0) {
			return { opts: null, data: null };
		}

		const advLoss = iterations.map((it) => iterMap.get(it)!.adv);
		const stratLoss = iterations.map((it) => iterMap.get(it)!.strat);

		// Check if there's any non-null data
		const hasAdv = advLoss.some((v) => v != null);
		const hasStrat = stratLoss.some((v) => v != null);
		if (!hasAdv && !hasStrat) return { opts: null, data: null };

		const series: uPlot.Series[] = [{ label: 'Iteration' }];
		const dataArrays: (number | null)[][] = [iterations.map(Number)];

		if (hasAdv) {
			series.push({ label: 'Advantage Loss', stroke: '#3b82f6', width: 1.5 });
			dataArrays.push(advLoss);
		}
		if (hasStrat) {
			series.push({ label: 'Strategy Loss', stroke: '#f97316', width: 1.5 });
			dataArrays.push(stratLoss);
		}

		const opts: uPlot.Options = {
			width,
			height: 300,
			scales: { x: { time: false }, y: { auto: true } },
			series,
			axes: [
				{ label: 'Iteration', stroke: axisColor, grid: { stroke: gridColor } },
				{ label: 'Loss', stroke: axisColor, grid: { stroke: gridColor } },
			],
			cursor: { drag: { x: true, y: false } },
		};

		return { opts, data: dataArrays as uPlot.AlignedData };
	}, [metrics, width]);

	return (
		<div ref={containerRef} className="space-y-6">
			<div>
				<h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
					Convergence
				</h3>
				{convergenceData.opts && convergenceData.data ? (
					<UplotReact options={convergenceData.opts} data={convergenceData.data} />
				) : (
					<p className="text-gray-500 dark:text-gray-400 text-sm">No win-rate data available.</p>
				)}
			</div>
			<div>
				<h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-2">
					Training Loss
				</h3>
				{lossData.opts && lossData.data ? (
					<UplotReact options={lossData.opts} data={lossData.data} />
				) : (
					<p className="text-gray-500 dark:text-gray-400 text-sm">No loss data available.</p>
				)}
			</div>
		</div>
	);
};

export default MetricsChart;
