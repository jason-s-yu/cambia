// src/components/training/LogViewer.tsx
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useTrainingStore } from '@/stores/trainingStore';

interface LogViewerProps {
	connected: boolean;
}

const AUTO_SCROLL_THRESHOLD = 40; // pixels

const LogViewer: React.FC<LogViewerProps> = ({ connected }) => {
	const logBuffer = useTrainingStore((s) => s.logBuffer);
	const containerRef = useRef<HTMLDivElement>(null);
	const [autoScroll, setAutoScroll] = useState(true);

	const handleScroll = useCallback(() => {
		const el = containerRef.current;
		if (!el) return;
		const isNearBottom =
			el.scrollTop + el.clientHeight >= el.scrollHeight - AUTO_SCROLL_THRESHOLD;
		setAutoScroll(isNearBottom);
	}, []);

	useEffect(() => {
		if (autoScroll && containerRef.current) {
			containerRef.current.scrollTop = containerRef.current.scrollHeight;
		}
	}, [logBuffer, autoScroll]);

	const jumpToBottom = () => {
		if (containerRef.current) {
			containerRef.current.scrollTop = containerRef.current.scrollHeight;
			setAutoScroll(true);
		}
	};

	return (
		<div className="relative">
			{/* Connection indicator */}
			<div className="absolute top-2 right-2 z-10 flex items-center gap-1.5">
				<span
					className={`inline-block w-2 h-2 rounded-full ${
						connected ? 'bg-green-500' : 'bg-red-500'
					}`}
				/>
				<span className="text-xs text-gray-400">
					{connected ? 'Connected' : 'Disconnected'}
				</span>
			</div>

			<div
				ref={containerRef}
				onScroll={handleScroll}
				className="bg-gray-900 rounded-lg h-96 overflow-y-scroll font-mono text-sm p-4 pr-8"
			>
				{logBuffer.length === 0 ? (
					<div className="text-gray-500">No log output yet...</div>
				) : (
					logBuffer.map((line, i) => (
						<div key={i} className="text-green-400 whitespace-pre-wrap break-all leading-5">
							{line}
						</div>
					))
				)}
			</div>

			{/* Jump to bottom */}
			{!autoScroll && (
				<button
					onClick={jumpToBottom}
					className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white text-xs px-3 py-1.5 rounded-full shadow-lg transition-colors"
				>
					Jump to bottom
				</button>
			)}
		</div>
	);
};

export default LogViewer;
