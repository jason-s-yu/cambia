// src/components/game/GameTable.tsx
import React from 'react';

interface GameTableProps {
	opponents: React.ReactNode[];
	selfSlot: React.ReactNode;
	centerSlot: React.ReactNode;
}

const getOpponentPositions = (count: number): { left: number; top: number }[] => {
	return Array.from({ length: count }, (_, i) => {
		const theta = Math.PI - ((i + 1) * Math.PI / (count + 1));
		const left = 50 + 40 * Math.cos(theta);
		const top = 50 - 36 * Math.sin(theta);
		return { left, top };
	});
};

const GameTable: React.FC<GameTableProps> = ({ opponents, selfSlot, centerSlot }) => {
	const positions = getOpponentPositions(opponents.length);

	return (
		<div className="relative w-full max-w-4xl aspect-[4/3] bg-green-700 dark:bg-green-900 border-4 border-yellow-700/80 dark:border-yellow-600/60 rounded-xl shadow-2xl overflow-hidden">
			{opponents.map((opponent, i) => {
				const pos = positions[i];
				return (
					<div
						key={i}
						className="absolute -translate-x-1/2 -translate-y-1/2"
						style={{ left: `${pos.left}%`, top: `${pos.top}%` }}
					>
						{opponent}
					</div>
				);
			})}

			<div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
				{centerSlot}
			</div>

			<div className="absolute bottom-3 left-1/2 -translate-x-1/2 w-full flex justify-center px-2">
				{selfSlot}
			</div>
		</div>
	);
};

export default GameTable;
