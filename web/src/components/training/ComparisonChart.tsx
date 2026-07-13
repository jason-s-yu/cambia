// src/components/training/ComparisonChart.tsx
import React, { useMemo, useRef, useEffect, useState } from 'react';
import UplotReact from 'uplot-react';
import type uPlot from 'uplot';
import type { RunComparison } from '@/types/training';

interface ComparisonChartProps {
	runs: RunComparison[];
}

// Fixed palette cycled by selection order so a run keeps the same color across
// re-renders as long as its position in the selection is stable.
const RUN_COLORS = [
	'#3b82f6', // blue
	'#f97316', // orange
	'#22c55e', // green
	'#ef4444', // red
	'#a855f7', // purple
	'#06b6d4', // cyan
];

function isDarkMode(): boolean {
	return document.documentElement.getAttribute('data-theme') === 'dark';
}

const ComparisonChart: React.FC<ComparisonChartProps> = ({ runs }) => {
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

	const chartData = useMemo(() => {
		const dark = isDarkMode();
		const axisColor = dark ? '#9ca3af' : '#6b7280';
		const gridColor = dark ? '#374151' : '#e5e7eb';

		// One mean_imp point-map per run, keyed by iteration.
		const byRun = runs.map((r) => {
			const map = new Map<number, number>();
			for (const p of r.mean_imp) {
				map.set(p.iteration, p.mean_imp * 100);
			}
			return map;
		});

		// Union of all iterations across all runs (unknown/no-data runs
		// contribute nothing here, which is fine: their series is all null).
		const iterSet = new Set<number>();
		for (const map of byRun) for (const k of map.keys()) iterSet.add(k);
		const iterations = Array.from(iterSet).sort((a, b) => a - b);

		if (iterations.length === 0) {
			return { opts: null, data: null };
		}

		const series: uPlot.Series[] = [{ label: 'Iteration' }];
		const dataArrays: (number | null)[][] = [iterations.map(Number)];

		runs.forEach((r, i) => {
			series.push({
				label: r.name,
				stroke: RUN_COLORS[i % RUN_COLORS.length],
				width: 2,
			});
			const map = byRun[i];
			dataArrays.push(iterations.map((it) => map.get(it) ?? null));
		});

		const opts: uPlot.Options = {
			width,
			height: 380,
			scales: { x: { time: false }, y: { auto: true } },
			series,
			axes: [
				{ label: 'Iteration', stroke: axisColor, grid: { stroke: gridColor } },
				{ label: 'mean_imp (%)', stroke: axisColor, grid: { stroke: gridColor } },
			],
			cursor: { drag: { x: true, y: false } },
		};

		return { opts, data: dataArrays as uPlot.AlignedData };
	}, [runs, width]);

	return (
		<div ref={containerRef}>
			{chartData.opts && chartData.data ? (
				<UplotReact options={chartData.opts} data={chartData.data} />
			) : (
				<p className="text-gray-500 dark:text-gray-400 text-sm py-8 text-center">
					No mean_imp data available for the selected runs.
				</p>
			)}
		</div>
	);
};

export default ComparisonChart;
