// src/components/training/ResourceMonitor.tsx
import React, { useMemo, useState } from 'react';
import UplotReact from 'uplot-react';
import type uPlot from 'uplot';
import { useResourceSocket } from '@/hooks/useResourceSocket';
import type { GPUStat, ResourceSnapshot } from '@/types/training';

function formatGB(mb: number): string {
	return `${(mb / 1024).toFixed(1)} GB`;
}

function pressureColor(pct: number): string {
	if (pct >= 90) return 'bg-red-500';
	if (pct >= 70) return 'bg-yellow-500';
	return 'bg-green-500';
}

function tempColor(tempC: number): string {
	if (tempC >= 85) return 'text-red-500';
	if (tempC >= 70) return 'text-yellow-500';
	return 'text-gray-600 dark:text-gray-300';
}

function gpuMemPctHistory(history: ResourceSnapshot[], index: number): (number | null)[] {
	return history.map((snap) => {
		const gpu = snap.gpus.find((g) => g.index === index);
		if (!gpu || gpu.mem_total_mb <= 0) return null;
		return (gpu.mem_used_mb / gpu.mem_total_mb) * 100;
	});
}

interface SparklineProps {
	data: (number | null)[];
	color: string;
}

// Minimal fixed-size sparkline: no axes/legend/cursor, just a trend line over
// the local history ring (bounded live window, not a historical range store).
const Sparkline: React.FC<SparklineProps> = ({ data, color }) => {
	const opts = useMemo<uPlot.Options | null>(() => {
		if (data.length < 2) return null;
		return {
			width: 160,
			height: 36,
			scales: { x: { time: false } },
			series: [{}, { stroke: color, width: 1.5, points: { show: false } }],
			axes: [{ show: false }, { show: false }],
			legend: { show: false },
			cursor: { show: false },
		};
	}, [data.length, color]);

	const aligned = useMemo<uPlot.AlignedData | null>(() => {
		if (data.length < 2) return null;
		return [data.map((_, i) => i), data] as uPlot.AlignedData;
	}, [data]);

	if (!opts || !aligned) {
		return (
			<div className="h-9 flex items-center text-xs text-gray-400 dark:text-gray-500">
				Gathering samples...
			</div>
		);
	}
	return <UplotReact options={opts} data={aligned} />;
};

interface StatTileProps {
	label: string;
	value: string;
	sub?: string;
	spark?: (number | null)[];
	sparkColor?: string;
}

const StatTile: React.FC<StatTileProps> = ({ label, value, sub, spark, sparkColor }) => (
	<div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
		<div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{label}</div>
		<div className="text-lg font-semibold text-gray-900 dark:text-gray-100">{value}</div>
		{sub && <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">{sub}</div>}
		{spark && <Sparkline data={spark} color={sparkColor ?? '#3b82f6'} />}
	</div>
);

const GPUCard: React.FC<{ gpu: GPUStat; history: ResourceSnapshot[] }> = ({ gpu, history }) => {
	const memPct = gpu.mem_total_mb > 0 ? (gpu.mem_used_mb / gpu.mem_total_mb) * 100 : 0;
	// WSL2 degrades per-process GPU stats to a "[Not Found]" stub; filter it out
	// and lean on device-total mem_used/mem_total for pressure instead.
	const procs = (gpu.processes ?? []).filter((p) => p.name !== '[Not Found]');

	return (
		<div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3">
			<div className="flex items-center justify-between mb-2">
				<div className="text-sm font-medium text-gray-800 dark:text-gray-200">
					GPU {gpu.index}: {gpu.name}
				</div>
				<span className={`text-xs font-mono ${tempColor(gpu.temp_c)}`}>
					{gpu.temp_c.toFixed(0)}°C
				</span>
			</div>

			<div className="mb-2">
				<div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-0.5">
					<span>VRAM</span>
					<span>
						{formatGB(gpu.mem_used_mb)} / {formatGB(gpu.mem_total_mb)} ({memPct.toFixed(0)}%)
					</span>
				</div>
				<div className="w-full h-2 rounded bg-gray-200 dark:bg-gray-700 overflow-hidden">
					<div
						className={`h-full ${pressureColor(memPct)} transition-all`}
						style={{ width: `${Math.min(100, memPct)}%` }}
					/>
				</div>
			</div>

			<div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
				<span>Utilization</span>
				<span className="font-mono">{gpu.util_pct.toFixed(0)}%</span>
			</div>

			<Sparkline data={gpuMemPctHistory(history, gpu.index)} color="#a855f7" />

			{procs.length > 0 && (
				<div className="mt-2 text-xs text-gray-500 dark:text-gray-400 space-y-0.5">
					{procs.map((p) => (
						<div key={p.pid} className="flex justify-between font-mono">
							<span className="truncate mr-2">{p.name}</span>
							<span>{p.mem_mb.toFixed(0)} MB</span>
						</div>
					))}
				</div>
			)}
		</div>
	);
};

const ResourceMonitor: React.FC = () => {
	const { connected, snapshot, history } = useResourceSocket();
	const [open, setOpen] = useState(true);

	const cpuHistory = useMemo(() => history.map((h) => h.cpu_pct), [history]);
	const memHistory = useMemo(
		() => history.map((h) => (h.mem_total_mb > 0 ? (h.mem_used_mb / h.mem_total_mb) * 100 : null)),
		[history],
	);

	return (
		<div className="bg-white dark:bg-gray-800 rounded-lg shadow mb-6">
			<button
				onClick={() => setOpen((o) => !o)}
				className="w-full flex items-center justify-between px-4 py-3 text-left"
			>
				<div className="flex items-center gap-2">
					<span
						className={`inline-block w-2 h-2 rounded-full ${
							connected ? 'bg-green-500' : 'bg-red-500'
						}`}
					/>
					<h2 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
						Resource Monitor
					</h2>
				</div>
				<span className="text-gray-400 text-xs">{open ? '▲ collapse' : '▼ expand'}</span>
			</button>

			{open && (
				<div className="px-4 pb-4">
					{!snapshot ? (
						<p className="text-sm text-gray-500 dark:text-gray-400">
							Waiting for resource data...
						</p>
					) : (
						<>
							<div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
								<StatTile
									label="CPU"
									value={`${snapshot.cpu_pct.toFixed(1)}%`}
									sub={`load ${snapshot.load_avg.map((v) => v.toFixed(2)).join(' / ')}`}
									spark={cpuHistory}
									sparkColor="#3b82f6"
								/>
								<StatTile
									label="Memory"
									value={`${formatGB(snapshot.mem_used_mb)} / ${formatGB(snapshot.mem_total_mb)}`}
									sub={`${formatGB(snapshot.mem_avail_mb)} available`}
									spark={memHistory}
									sparkColor="#22c55e"
								/>
								<StatTile
									label="Disk"
									value={`${snapshot.disk_used_gb.toFixed(1)} / ${snapshot.disk_total_gb.toFixed(1)} GB`}
									sub={`${snapshot.disk_free_gb.toFixed(1)} GB free`}
								/>
								<StatTile
									label="Load Average"
									value={snapshot.load_avg.map((v) => v.toFixed(2)).join(' / ')}
									sub="1m / 5m / 15m"
								/>
							</div>

							<div>
								<h3 className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
									GPU
								</h3>
								{!snapshot.gpu_available || snapshot.gpus.length === 0 ? (
									<p className="text-sm text-gray-500 dark:text-gray-400">
										No GPU detected on this host.
									</p>
								) : (
									<div className="grid grid-cols-1 md:grid-cols-2 gap-3">
										{snapshot.gpus.map((gpu) => (
											<GPUCard key={gpu.index} gpu={gpu} history={history} />
										))}
									</div>
								)}
							</div>
						</>
					)}
				</div>
			)}
		</div>
	);
};

export default ResourceMonitor;
